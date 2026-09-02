# Setup

## Current local baseline

The project is currently developed on Windows with the user's CPU-only virtual environment. The tested baseline is the existing `.venv`. Do not replace it while the CPU pipeline is being validated.

## Python environment

From the repository root:

```powershell
.venv\Scripts\Activate.ps1
python --version
```

Verify that the same Python owns pip:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip --version
```

The environment must contain the tested dependencies:

```powershell
python -m pip install --upgrade "setuptools<81"
python -m pip install soundfile
python -m pip install PytorchWildlife supervision speciesnet
```

SpeciesNet installation can be verified using its documented CLI:

```powershell
python -m speciesnet.scripts.run_model --help
```

The current SpeciesNet repository documents `pip install speciesnet` and the `run_model` CLI with `--folders` and `--predictions_json`. It also notes that the current CLI processes still images; our pipeline therefore extracts tracked animal crops before calling SpeciesNet.

## CVAT

Use the official CVAT Community Docker Compose deployment for annotation. Keep raw wildlife video outside Git.

## First perception test

A working test video is expected at:

`data/raw/test_video/test.mp4`

The validated command is:

```powershell
python scripts/run_video.py --input data/raw/test_video/test.mp4
```

The validated result should produce an annotated video plus JSON detection/track manifests under `data/outputs/video_test/`.

## SpeciesNet test

Once track crops have been created:

```powershell
python -m scripts.run_speciesnet --folder data/outputs/video_test/crops
```

SpeciesNet writes predictions to the configured JSON path. Do not commit model caches or prediction media to Git.
