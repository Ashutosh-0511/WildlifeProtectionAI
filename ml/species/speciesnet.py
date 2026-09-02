from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class SpeciesNetAdapter:
    """Adapter for Google's open-source SpeciesNet command-line runtime."""

    def __init__(self, country: str = "IND"):
        self.country = country

    @staticmethod
    def check_installation() -> None:
        """Compatibility check: import the public SpeciesNet package directly."""
        try:
            import speciesnet  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"SpeciesNet package cannot be imported: {exc}") from exc

    def classify_folder(self, image_folder: Path, output_json: Path) -> dict[str, Any]:
        image_folder = image_folder.resolve()
        output_json = output_json.resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        self.check_installation()
        cmd = [
            sys.executable,
            "-m",
            "speciesnet.scripts.run_model",
            "--folders",
            str(image_folder),
            "--predictions_json",
            str(output_json),
            "--country",
            self.country,
            "--bypass_prompts",
        ]
        result = subprocess.run(cmd, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"SpeciesNet failed with exit code {result.returncode}. "
                "Run the command above directly to inspect the SpeciesNet error."
            )
        if not output_json.exists():
            raise RuntimeError(f"SpeciesNet completed but did not create {output_json}")
        return json.loads(output_json.read_text(encoding="utf-8"))
