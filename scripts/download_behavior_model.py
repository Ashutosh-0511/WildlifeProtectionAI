from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "imageomics/x3d-kabr-kinetics"
# The repository currently exposes the fine-tuned checkpoint used by the project.
FILENAME = "x3d-l-kabr-kinetics-lr5e-2-ep120-best-epoch.zip"


def main() -> None:
    # Kept as a reproducibility helper for the legacy X3D-KABR experiment.
    # V1 uses models/behavior/videomae/videomae_combined_v1.pt, which is a
    # separate local checkpoint and should be obtained from the project's
    # model source rather than committed to Git.
    out = Path("models/behavior/x3d-kabr")
    out.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        repo_type="model",
        local_dir=out,
    )
    print(path)


if __name__ == "__main__":
    main()
