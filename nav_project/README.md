# Perception, Motion Planning and Control for a Mobile Robot (Webots + Python)

Perception (LiDAR + camera fusion) -> A*/Dijkstra planning -> PID path tracking. No ROS/Gazebo.

## STATUS - read first
* The Webots world and controller have NOT been run in real Webots (only against a mock of the Webots API, `tests/`).
  Expect to debug: HSV thresholds in `navlib/perception.py`, camera/LiDAR conventions, `worlds/arena.wbt`.
* Everything in `results/` was produced by the offline Python simulator, NOT Webots, and comes from a MIXED state:
  the survey route was changed (it now returns to the start before planning) after most of these were generated.
  Re-run `python experiments/run_all.py` (~6-8 min) before using any number or figure.
* No written report is included yet.

## Layout
navlib/          shared library (world, sensors, perception/fusion, planning, control, simulator)
experiments/     run_all.py reproduces every figure/table in results/
worlds/          generate_world.py -> arena.wbt (same layout as the simulator)
controllers/nav_controller/nav_controller.py   Webots controller (self-calibrates LiDAR order and IMU yaw)
analysis/plot_webots_run.py   planned-vs-actual figure + performance table from Webots logs
tests/run_mock.py             runs the real controller against a fake Webots API (no Webots needed)

## Usage
pip install -r requirements.txt
python experiments/run_all.py              # offline results
python worlds/generate_world.py            # regenerate arena.wbt
# Webots (R2023b+): open worlds/arena.wbt and press play; the world's basicTimeStep must stay 32 ms.
# Copy/point the controller folder so the world finds controllers/nav_controller.
python analysis/plot_webots_run.py         # after a Webots run (logs/webots_*.csv|npz)
python tests/run_mock.py && python analysis/plot_webots_run.py --prefix mock
