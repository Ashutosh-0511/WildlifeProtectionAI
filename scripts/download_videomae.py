from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "sakifkhan98/cattle-vision-framework"
FILENAME = "videomae_combined_v1.pt"


def main() -> None:
    out = Path("models/behavior/videomae")
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
