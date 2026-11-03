import importlib.util, sys, tempfile, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def have(mod):
    return importlib.util.find_spec(mod) is not None


def tmpdir():
    d = Path(tempfile.mkdtemp(prefix="cyptdi_test_"))
    return d


def cleanup(d):
    shutil.rmtree(d, ignore_errors=True)
