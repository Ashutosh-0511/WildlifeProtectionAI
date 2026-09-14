# Gemini Video Behaviour Analysis

The project can optionally send the uploaded wildlife video to Gemini for a second, temporal behaviour analysis. The Gemini result is stored as structured JSON and can be consumed by the dashboard/API.

## Setup

Install the Google GenAI SDK:

```bash
pip install google-genai
```

Set the API key in the shell rather than committing it:

PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
```

Optional model override:

```powershell
$env:GEMINI_MODEL="YOUR_CURRENT_FREE_GEMINI_MODEL"
```

The code has a default model value, but using `GEMINI_MODEL` is recommended so the project can follow the currently available model in Google AI Studio without changing code.

## Standalone test

```bash
python -m scripts.run_gemini_behavior --input path\to\lion.mp4
```

The default output is:

`data/outputs/gemini/<video-name>.json`

## Dashboard integration design

The recommended architecture is:

1. Existing local pipeline performs detection/tracking/species analysis.
2. The same original uploaded video is sent once to Gemini for temporal behaviour/risk analysis.
3. Gemini returns structured JSON using a response schema, not free-form text parsing.
4. The JSON is saved beside the existing job artifacts.
5. The dashboard displays Gemini behaviour/risk as a separate evidence source.
6. Existing local risk logic remains available as a fallback if Gemini is disabled, unavailable, rate-limited, or returns insufficient evidence.

Do not put the Gemini API key in frontend JavaScript or commit it to Git. The key must remain server-side.

## Important limitation

Gemini is a general multimodal model, not a validated wildlife ethology classifier. Its behaviour descriptions should be treated as decision support and should not silently overwrite a specialist model's output. For high-stakes conservation deployment, benchmark Gemini against labelled wildlife clips and retain human review for high-risk alerts.
