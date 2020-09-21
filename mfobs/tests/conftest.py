import os
import shutil
from pathlib import Path
import pytest


@pytest.fixture(scope="session")
def project_root_path():
    """Root folder for the project (with setup.py)"""
    filepath = os.path.split(os.path.abspath(__file__))[0]
    return Path(os.path.normpath(os.path.join(filepath, '../../')))


@pytest.fixture(scope="session")
def test_data_path(project_root_path):
    """Root folder for the project (with setup.py),
    two levels up from the location of this file.
    """
    return Path(project_root_path, 'mfobs', 'tests', 'data')


@pytest.fixture(scope="session", autouse=True)
def test_output_folder(project_root_path):
    """(Re)make an output folder for the tests 
    at the begining of each test session."""
    folder = project_root_path / 'mfobs/tests/output'
    reset = True
    if reset:
        if folder.is_dir():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    return folder