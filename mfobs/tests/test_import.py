import os
import subprocess as sp
from pathlib import Path


def test_import():
    """Test that Modflow-obs is installed, and can be imported
    (from another location besides the repo top-level, which contains the
    'mfobs' folder)."""
    os.system("python -c 'import mfobs'")
    results = sp.check_call(["python", "-c", "import mfobs"], cwd=Path('..'))
