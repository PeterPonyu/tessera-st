"""Public-release identity/prediction verification; not scientific validity."""
from pathlib import Path
import subprocess
import sys

CAP = Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(CAP / 'verify_release.py')], check=True)
