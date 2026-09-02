# Deployment

GitHub is the source of truth. Vercel should use apps/web as the root directory and deploy the Next.js ranger dashboard. The API URL is configured through NEXT_PUBLIC_API_BASE_URL.

MegaDetector, ByteTrack, SpeciesNet and ResNet+LSTM run on a separate GPU-capable worker. Never commit credentials, raw wildlife video, sensitive coordinates or model weights.
