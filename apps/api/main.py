from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from ml.pipeline.integrated import run_integrated

ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = ROOT / "data" / "uploads"
OUTPUT_DIR = ROOT / "data" / "outputs" / "dashboard"
CHECKPOINT = ROOT / "models" / "behavior" / "videomae" / "videomae_combined_v1.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Wildlife Protection AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

EVENTS: list[dict[str, Any]] = []

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@app.get("/v1/health")
def health():
    return {
        "status": "ok",
        "service": "wildlife-dashboard",
        "pipeline": "MegaDetector V6 + ByteTrack + SpeciesNet + VideoMAE + Risk Engine",
        "device": "cpu",
        "checkpoint_present": CHECKPOINT.exists(),
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/v1/events")
def events():
    return {"items": EVENTS, "count": len(EVENTS)}


@app.get("/v1/events/{event_id}")
def event(event_id: str):
    for item in EVENTS:
        if item["risk_event_id"] == event_id:
            return item
    raise HTTPException(404, "Event not found")


@app.post("/v1/inference/video")
async def inference_video(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(400, "Supported video formats: MP4, AVI, MOV, MKV")

    job_id = str(uuid4())
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    job_dir = OUTPUT_DIR / job_id

    with input_path.open("wb") as destination:
        shutil.copyfileobj(file.file, destination)

    if not CHECKPOINT.exists():
        raise HTTPException(500, f"VideoMAE checkpoint not found: {CHECKPOINT}")

    try:
        result = run_integrated(
            input_path,
            job_dir,
            sample_every=3,
            species_samples=16,
            behavior_checkpoint=CHECKPOINT,
        )
    except Exception as exc:
        raise HTTPException(500, f"Pipeline failed: {type(exc).__name__}: {exc}") from exc

    # Convert local artifact paths into browser URLs.
    result["job_id"] = job_id
    result["filename"] = file.filename
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["outputs"]["annotated_video_url"] = f"/outputs/{job_id}/video/annotated.mp4"
    if result["outputs"].get("evidence"):
        result["outputs"]["evidence_url"] = f"/outputs/{job_id}/evidence.jpg"

    for event in result.get("risk_events", []):
        event["job_id"] = job_id
        event["evidence_uri"] = result["outputs"].get("evidence_url")
        EVENTS.insert(0, event)

    return result


@app.post("/v1/events/{event_id}/review")
def review(event_id: str, decision: str):
    allowed = {"TRUE_POSITIVE", "FALSE_POSITIVE", "NEEDS_REVIEW", "REVIEWED"}
    if decision not in allowed:
        raise HTTPException(400, f"decision must be one of {sorted(allowed)}")
    for item in EVENTS:
        if item["risk_event_id"] == event_id:
            item["review"] = decision
            item["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            return item
    raise HTTPException(404, "Event not found")


DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wildlife Protection AI</title>
<style>
*{box-sizing:border-box}body{margin:0;font-family:Inter,Segoe UI,Arial,sans-serif;background:#07110d;color:#edf7f0}
header{padding:24px 5%;border-bottom:1px solid #21352b;background:#0a1711;display:flex;justify-content:space-between;align-items:center}
h1{margin:0;font-size:25px}header p{margin:5px 0 0;color:#8fa99a;font-size:13px}
.badge{border:1px solid #315c46;border-radius:999px;padding:8px 12px;color:#9fe0b7;font-size:12px}
main{width:min(1200px,90%);margin:28px auto}
.panel{background:#0c1b14;border:1px solid #20392b;border-radius:16px;padding:22px;margin-bottom:20px}
.upload{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
input[type=file]{padding:12px;border:1px dashed #426451;border-radius:10px;background:#08130e;color:#cfe1d6;flex:1}
button{background:#2e9d62;color:white;border:0;border-radius:10px;padding:12px 20px;font-weight:700;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}
.status{margin-top:14px;color:#9db7a8;min-height:20px}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.card{padding:18px;border-radius:13px;background:#0a1711;border:1px solid #213a2c}.label{color:#7e9989;font-size:12px;text-transform:uppercase}.value{font-size:20px;font-weight:700;margin-top:8px}
.results{display:grid;grid-template-columns:1.3fr .7fr;gap:20px}.video{width:100%;border-radius:10px;background:#000;max-height:520px}
.evidence{width:100%;border-radius:10px;margin-top:10px}
.risk{font-size:28px;font-weight:800}.HIGH,.CRITICAL{color:#ff8c8c}.MEDIUM{color:#ffd27a}.LOW{color:#9fe0b7}.UNKNOWN{color:#aeb9b2}
table{width:100%;border-collapse:collapse;margin-top:10px}td,th{padding:10px;text-align:left;border-bottom:1px solid #20352a;font-size:13px}
.factor{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1b2d24;color:#b6c9bd}
.hidden{display:none}.error{color:#ff9696}
@media(max-width:800px){.grid,.results{grid-template-columns:1fr 1fr}.results{grid-template-columns:1fr}} 
</style>
</head>
<body>
<header>
<div><h1>Wildlife Protection AI</h1><p>AI-based video surveillance & conservation decision support</p></div>
<div class="badge">LOCAL DEMO • CPU</div>
</header>
<main>
<section class="panel">
<h2>Analyze Wildlife Video</h2>
<p style="color:#8fa99a">Upload a surveillance clip. The system runs detection → tracking → species → behaviour → risk assessment.</p>
<div class="upload">
<input id="file" type="file" accept=".mp4,.avi,.mov,.mkv">
<button id="run" onclick="analyze()">Analyze Video</button>
</div>
<div id="status" class="status">Ready.</div>
</section>

<section id="dashboard" class="hidden">
<div class="grid">
<div class="card"><div class="label">Animals tracked</div><div id="tracks" class="value">—</div></div>
<div class="card"><div class="label">Species</div><div id="species" class="value">—</div></div>
<div class="card"><div class="label">Behaviour</div><div id="behavior" class="value">—</div></div>
<div class="card"><div class="label">Risk</div><div id="risk" class="value">—</div></div>
</div>

<div class="results" style="margin-top:20px">
<div class="panel">
<h3>Processed Evidence Video</h3>
<video id="video" class="video" controls></video>
</div>
<div class="panel">
<h3>Event Evidence</h3>
<img id="evidence" class="evidence" alt="AI evidence frame">
<div id="confidence" style="margin-top:14px;color:#9db7a8"></div>
</div>
</div>

<div class="panel">
<h3>Pipeline Result</h3>
<table><thead><tr><th>Track</th><th>Species</th><th>Behaviour</th><th>Confidence</th><th>Risk</th><th>Human</th></tr></thead>
<tbody id="rows"></tbody></table>
</div>

<div class="panel">
<h3>Risk Factors</h3>
<div id="factors"></div>
</div>
</section>
</main>
<script>
function esc(x){return String(x??"—").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]))}
async function analyze(){
 const input=document.getElementById('file'), btn=document.getElementById('run'), status=document.getElementById('status');
 if(!input.files.length){status.innerHTML='<span class="error">Choose a video first.</span>';return}
 btn.disabled=true; status.textContent='Running the complete pipeline. CPU inference may take several minutes for a long clip...';
 document.getElementById('dashboard').classList.add('hidden');
 const fd=new FormData();fd.append('file',input.files[0]);
 try{
   const res=await fetch('/v1/inference/video',{method:'POST',body:fd});
   const data=await res.json(); if(!res.ok) throw new Error(data.detail||'Pipeline failed');
   render(data); status.textContent='Analysis complete.';
 }catch(e){status.innerHTML='<span class="error">'+esc(e.message)+'</span>'}
 finally{btn.disabled=false}
}
function render(d){
 document.getElementById('dashboard').classList.remove('hidden');
 const summary=d.summary||{}, species=Object.values(d.species||{}), behaviors=Object.values(d.behavior||{}), events=d.risk_events||[];
 document.getElementById('tracks').textContent=(summary.unique_track_ids||[]).length;
 document.getElementById('species').textContent=species[0]?.species||'UNKNOWN';
 document.getElementById('behavior').textContent=behaviors[0]?.behaviour||'UNKNOWN';
 const risk=events[0]?.risk||{}; const level=risk.risk_level||'UNKNOWN';
 document.getElementById('risk').textContent=level;document.getElementById('risk').className='value '+level;
 document.getElementById('video').src=d.outputs.annotated_video_url;
 if(d.outputs.evidence_url){document.getElementById('evidence').src=d.outputs.evidence_url}
 document.getElementById('confidence').textContent='Behaviour confidence: '+((behaviors[0]?.confidence||0)*100).toFixed(1)+'% • Species confidence: '+((species[0]?.confidence||0)*100).toFixed(1)+'%';
 document.getElementById('rows').innerHTML=events.map(e=>'<tr><td>'+esc(e.track_id)+'</td><td>'+esc(e.species)+'</td><td>'+esc(e.behaviour)+'</td><td>'+((e.behaviour_confidence||0)*100).toFixed(1)+'%</td><td class="'+esc(e.risk?.risk_level)+'">'+esc(e.risk?.risk_level)+'</td><td>'+((e.human_present)?'YES':'NO')+'</td></tr>').join('');
 document.getElementById('factors').innerHTML=(risk.factors||[]).map(f=>'<div class="factor"><span>'+esc(f.name)+'</span><span>'+esc(f.contribution)+'</span></div>').join('');
}
</script>
</body></html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.api.main:app", host="127.0.0.1", port=8000, reload=False)
