"""Deployment entry point.

Streamlit Community Cloud runs the repo-root main file. The real dashboard
lives under a directory whose name contains a space and a dot, so it cannot be
imported as a package; load it by path instead.
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "trading.quant mvp", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

runpy.run_path(os.path.join(SRC, "visualization", "dashboard.py"), run_name="__main__")
