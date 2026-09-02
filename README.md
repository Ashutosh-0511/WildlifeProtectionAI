# Wildlife Protection AI

AI-based wildlife video surveillance: **detection → tracking → species → behaviour → conservation risk → ranger alerting**.

## V1 local demonstration

The fastest complete demonstration uses the already-tested CPU pipeline:

**MegaDetector V6 → ByteTrack → SpeciesNet → VideoMAE → Risk Engine → Local Ranger Dashboard**

The dashboard is intentionally local for the current review. ResNet18+LSTM, Animal Kingdom training, advanced research evaluation, PostgreSQL, cloud deployment and Vercel are future integrations.

## Run the dashboard

From the repository root with the project virtual environment activated:

```powershell
python -m pip install -r requirements.txt
python -m scripts.run_dashboard
```

Open **http://127.0.0.1:8000**

Upload a short wildlife MP4 and click **Analyze Video**.

The browser dashboard shows:
- processed evidence video
- number of tracked animals
- SpeciesNet species result
- VideoMAE behaviour result
- confidence
- LOW/MEDIUM/HIGH/CRITICAL risk
- human presence
- risk factors
- evidence frame

The complete result is also written to `data/outputs/dashboard/<job_id>/pipeline.json`.

## Existing command-line pipeline

```powershell
python -m scripts.run_pipeline --input data\raw\test_video\test.mp4 --output-dir data\outputs\full_pipeline --behavior-checkpoint models\behavior\videomae\videomae_combined_v1.pt
```

## Architecture

```
Video
  ↓
MegaDetector V6
  ↓
ByteTrack
  ↓
per-track crops
  ├── SpeciesNet → species
  └── VideoMAE → behaviour
             ↓
       Risk Engine
             ↓
      Ranger Dashboard
```

## Research roadmap

The future research version can add Animal Kingdom transfer learning, ResNet18+LSTM comparison, Indian target-domain fine-tuning, camera/location holdout evaluation, calibration, PostgreSQL persistence and a separate inference worker.

## Important limitation

The V1 VideoMAE checkpoint is a pretrained demonstration model and its behaviour labels should not be presented as scientifically validated Indian-wildlife behaviour recognition. Risk is decision support for a human ranger, not an autonomous enforcement decision.

## Deployment decision

GitHub remains the source of truth. Vercel can host a future lightweight web frontend, while heavy Python video inference should run separately on a machine/server capable of sustained model execution.
