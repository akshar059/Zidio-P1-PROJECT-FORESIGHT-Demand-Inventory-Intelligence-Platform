"""
PROJECT FORESIGHT / RETAILAI: Enterprise Demand & Inventory Intelligence System
Main Direct Entry Point for Streamlit Dashboard Application.
"""

import os
import sys
from pathlib import Path

# Explicitly set active working directory to project root
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

dashboard_path = ROOT_DIR / "src" / "dashboard" / "app.py"

# Directly execute the dashboard application in the active process
with open(dashboard_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(dashboard_path), "exec")
    exec(code, globals())
