from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    app = Path(__file__).with_name("app.py")
    repo = app.parents[2]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(repo), os.environ.get("PYTHONPATH")]))}
    raise SystemExit(subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]],
                                    env=environment, cwd=str(repo)).returncode)
