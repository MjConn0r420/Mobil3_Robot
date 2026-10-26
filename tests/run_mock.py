"""Runs the REAL Webots controller file against the mock Webots API (no Webots needed)."""
import os, sys, runpy
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, ".."))
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(here, "mock_webots"))
os.environ["NAV_LOG_PREFIX"] = "mock"
runpy.run_path(os.path.join(root, "controllers", "nav_controller", "nav_controller.py"), run_name="__main__")
