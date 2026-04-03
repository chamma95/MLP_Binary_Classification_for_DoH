import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_DIR = PROJECT_ROOT / ".venv"

req_file = PROJECT_ROOT / "requirements.txt"
if not req_file.exists():
    print("requirements.txt not found. Please create it with needed packages.")
    sys.exit(1)

if not VENV_DIR.exists():
    print("Creating virtual environment at .venv...")
    subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
else:
    print("Virtual environment already exists at .venv")

if os.name == "nt":
    python_bin = VENV_DIR / "Scripts" / "python.exe"
    pip_bin = VENV_DIR / "Scripts" / "pip.exe"
else:
    python_bin = VENV_DIR / "bin" / "python"
    pip_bin = VENV_DIR / "bin" / "pip"

print("Upgrading pip...")
subprocess.check_call([str(pip_bin), "install", "--upgrade", "pip"])
print("Installing requirements...")
subprocess.check_call([str(pip_bin), "install", "-r", str(req_file)])
print("Setup complete! Activate with:")
if os.name == "nt":
    print("    .venv\\Scripts\\Activate.ps1")
else:
    print("    source .venv/bin/activate")
