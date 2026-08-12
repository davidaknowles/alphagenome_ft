from pathlib import Path


def test_v0data_venv_launchers_export_python_runtime_path() -> None:
    missing = []
    for path in Path("scripts/v0data").rglob("*.sbatch"):
        script = path.read_text()
        if "/venv/" in script and "/bin/python" in script:
            if "LD_LIBRARY_PATH=" not in script:
                missing.append(str(path))

    assert missing == []
