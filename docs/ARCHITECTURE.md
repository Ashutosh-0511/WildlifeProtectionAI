# Architecture

Video ingestion → MegaDetector → ByteTrack → SpeciesNet → ResNet18+LSTM → transparent risk engine → alert engine → FastAPI/PostgreSQL/object storage → Next.js/Vercel ranger dashboard.

Vercel hosts the web UI. Long-running/GPU inference runs on a separate worker host. This keeps video inference independent of serverless execution limits.
