"use client";

import { useEffect, useState } from "react";

type Event = {
  risk_event_id: string;
  risk_level: string;
  species: string;
  behaviour: string;
  camera_id: string;
  zone_id: string;
  risk_score: number;
  human_present: boolean;
  created_at: string;
  evidence_uri?: string | null;
  review?: string;
};

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [events, setEvents] = useState<Event[]>([]);
  const [status, setStatus] = useState("Loading");

  async function load() {
    try {
      const res = await fetch(\`\${API}/v1/events\`, { cache: "no-store" });
      const data = await res.json();
      setEvents(data.items || []);
      setStatus("Connected");
    } catch {
      setStatus("API unavailable");
    }
  }

  async function review(id: string, decision: string) {
    await fetch(\`\${API}/v1/events/\${id}/review?decision=\${decision}\`, { method: "POST" });
    await load();
  }

  useEffect(() => { load(); }, []);

  return (
    <main style={{ minHeight: "100vh", background: "#07110d", color: "#eaf4ee", padding: 32, fontFamily: "system-ui" }}>
      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 28 }}>
          <div>
            <h1 style={{ margin: 0 }}>Wildlife Protection AI</h1>
            <p style={{ color: "#9db2a5" }}>Ranger decision-support dashboard</p>
          </div>
          <span style={{ padding: "8px 12px", borderRadius: 20, background: "#10251a" }}>{status}</span>
        </header>

        <section style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
          {["CRITICAL", "HIGH", "MEDIUM", "ALL"].map(level => (
            <div key={level} style={{ background: "#0d1b14", border: "1px solid #1d3528", borderRadius: 12, padding: 18 }}>
              <div style={{ color: "#9db2a5", fontSize: 12 }}>{level}</div>
              <strong style={{ fontSize: 28 }}>
                {level === "ALL" ? events.length : events.filter(e => e.risk_level === level).length}
              </strong>
            </div>
          ))}
        </section>

        <section style={{ background: "#0d1b14", border: "1px solid #1d3528", borderRadius: 14, overflow: "hidden" }}>
          <div style={{ padding: 18, borderBottom: "1px solid #1d3528" }}>
            <h2 style={{ margin: 0 }}>Recent Alerts</h2>
          </div>
          {events.length === 0 ? (
            <p style={{ padding: 24, color: "#9db2a5" }}>No events available. Start the inference worker or submit a recorded clip.</p>
          ) : events.map(e => (
            <article key={e.risk_event_id} style={{ padding: 18, borderBottom: "1px solid #1d3528", display: "grid", gridTemplateColumns: "110px 1fr auto", gap: 16, alignItems: "center" }}>
              <strong>{e.risk_level}</strong>
              <div>
                <div><strong>{e.species}</strong> · {e.behaviour}</div>
                <div style={{ color: "#9db2a5", fontSize: 13 }}>{e.camera_id} · {e.zone_id} · score {e.risk_score.toFixed(1)} · human {e.human_present ? "yes" : "no"}</div>
                <div style={{ color: "#6f8a7a", fontSize: 12 }}>{new Date(e.created_at).toLocaleString()}</div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button onClick={() => review(e.risk_event_id, "TRUE_POSITIVE")}>Confirm</button>
                <button onClick={() => review(e.risk_event_id, "FALSE_POSITIVE")}>False</button>
              </div>
            </article>
          ))}
        </section>

        <footer style={{ marginTop: 24, color: "#6f8a7a", fontSize: 12 }}>
          AI output is decision support. Rangers remain the final decision-makers.
        </footer>
      </div>
    </main>
  );
}
