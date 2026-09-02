# Windows + NVIDIA GPU setup

Your development laptop reports an NVIDIA GeForce GTX 1650 4GB and driver 591.91. Use a separate Python 3.11 virtual environment for this project. The current environment is Python 3.13 with CPU-only Torch and is not our target environment.

## 1. Create the clean environment

From the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools==80.9.0 wheel
```

If `py -3.11` is unavailable, install Python 3.11.x and retry.

## 2. Install CUDA-enabled PyTorch

For the pinned baseline we use the official CUDA 12.6 wheels:

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

PyTorch publishes this exact Windows/Linux CUDA 12.6 command in its previous-version installation documentation.

## 3. Install project dependencies

```powershell
python -m pip install -r requirements.txt
python -m pip install -r apps/api/requirements.txt
```

## 4. Verify GPU access

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Expected:

```text
CUDA available: True
GPU: NVIDIA GeForce GTX 1650
```

## 5. Verify MegaDetector

```powershell
python -c "from PytorchWildlife.models import detection as d; m=d.MegaDetectorV6(device='cuda', version='MDV6-yolov9-c'); print('MEGADETECTOR_GPU_OK')"
```

For a more licensing-explicit MIT model variant, use the `MegaDetectorV6MIT` class and one of the MIT model versions documented by PyTorch-Wildlife.

## 6. Test one image

```powershell
python scripts/test_megadetector.py data/raw/test/animal.jpg --device cuda --version MDV6-yolov9-c
```

## 7. Full video test

```powershell
python scripts/run_pipeline.py --video data/raw/demo.mp4 --sample-every 3 --max-frames 300
```

## 8. Do not install the CUDA Toolkit solely because `nvidia-smi` reports CUDA 13.1

The NVIDIA driver version and the CUDA runtime packaged with PyTorch are separate. The driver can support newer CUDA runtimes than the one bundled with the PyTorch wheel.
