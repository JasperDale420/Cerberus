from pathlib import Path


def _requirement_names() -> set[str]:
    requirements_path = Path(__file__).resolve().parents[2] / "requirements.txt"
    names: set[str] = set()
    for raw_line in requirements_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        package_part = line.split("#", 1)[0].strip()
        name = package_part.split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].strip()
        names.add(name.lower())
    return names


def _raw_requirements() -> list[str]:
    requirements_path = Path(__file__).resolve().parents[2] / "requirements.txt"
    return [line.strip() for line in requirements_path.read_text().splitlines() if line.strip()]


def test_requirements_include_pydantic_settings_runtime_dependency() -> None:
    assert "pydantic-settings" in _requirement_names()


def test_requirements_include_pyarrow_runtime_dependency() -> None:
    assert "pyarrow" in _requirement_names()
