"""Reproduces every figure and table in results/.   Usage:  python experiments/run_all.py   (~6-8 min)"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import exp_control, exp_perception, exp_planning, exp_integration

for name, mod in [("control tuning", exp_control), ("perception", exp_perception),
                  ("planning", exp_planning), ("integration + sweeps", exp_integration)]:
    t0 = time.time()
    print(f"=== {name} ===", flush=True)
    mod.run()
    print(f"    done in {time.time() - t0:.0f} s", flush=True)
print("all results written to results/figures and results/tables")
