#!/usr/bin/env python3
"""Copy the six main-text vector figures into JAMIA's separate figure files."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from output_safety import prepare_generation

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FIGURES = (
    ("F1_overview.pdf", "Figure_01.pdf"),
    ("F3_external_comparison.pdf", "Figure_02.pdf"),
    ("F4_aligned_architecture.pdf", "Figure_03.pdf"),
    ("F3_protocol_objective.pdf", "Figure_04.pdf"),
    ("F9_p2_loho.pdf", "Figure_05.pdf"),
    ("F8_p1_landmark_use_case.pdf", "Figure_06.pdf"),
)
SOURCE_DIR = ROOT / "results" / "paper"
TARGET_DIR = ROOT / "paper" / "submission" / "figures"
MANIFEST = TARGET_DIR / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive the current submission figure package before replacing it")
    args = parser.parse_args()
    missing = [str(SOURCE_DIR / source) for source, _ in SOURCE_FIGURES
               if not (SOURCE_DIR / source).is_file()]
    if missing:
        raise FileNotFoundError("missing main-text figure(s): " + ", ".join(missing))
    outputs = [TARGET_DIR / target for _, target in SOURCE_FIGURES] + [MANIFEST]
    prepare_generation(ROOT, outputs, refresh=args.refresh_generated,
                       label="jamia_submission_figures")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    figures = []
    for source_name, target_name in SOURCE_FIGURES:
        source = SOURCE_DIR / source_name
        target = TARGET_DIR / target_name
        shutil.copy2(source, target)
        figures.append({
            "file": target_name,
            "source": str(source.relative_to(ROOT)),
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
            "format": "PDF (vector source copied without raster conversion)",
        })
    MANIFEST.write_text(json.dumps({
        "description": "JAMIA main-text figure files, numbered in manuscript order.",
        "figures": figures,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"figure_count": len(figures), "manifest": str(MANIFEST.relative_to(ROOT))},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
