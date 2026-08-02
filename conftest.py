# Empty on purpose. pytest auto-discovers conftest.py and adds its
# directory (the project root) to sys.path — this is what makes
# `from perception.perception_agent import PerceptionAgent` and
# `from reasoning.baseline_arm import ...` resolvable when running
# `pytest` directly (as opposed to `python -m pytest`, which already
# adds the current directory to sys.path on its own).
