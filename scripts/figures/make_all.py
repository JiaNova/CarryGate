#!/usr/bin/env python
"""Regenerate every figure in the manuscript, in order.

Each module reads only committed result CSVs and the saved prediction arrays, so
the figures cannot drift from the numbers in the text.
"""

import runpy
import sys
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULES = ["fig_overview", "fig_clinical", "fig_external", "fig_capacity",
           "fig_protocol", "fig_per_variable"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-generated", action="store_true",
                        help="archive existing figure PDFs and PNGs before replacing them")
    parser.parse_args()
    sys.path.insert(0, str(HERE))
    for name in MODULES:
        print(f"{name}:")
        runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
