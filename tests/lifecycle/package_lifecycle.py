"""Package lifecycle tests for issue #22.

These tests are deliberately **not** part of the fast suite: the file is named
``package_lifecycle.py`` (not matching ``test_*.py``/``*_test.py``) and lives
under ``tests/lifecycle/``,
so ``pytest`` with the default ``testpaths = ["tests"]`` does not collect it.
Run them explicitly:

    pytest tests/lifecycle/package_lifecycle.py -q

They build/consume wheels and create throw-away virtual environments, so they
are slow and need network access for dependencies (numpy/scikit-learn, and
TensorFlow for the Keras extra). The hermetic job uses a locally built wheel
(``DNNGA_WHEEL_DIR=dist``); the index job is parameterised by
``DNNGA_INDEX_URL``/``DNNGA_PREVIOUS_VERSION``/``DNNGA_VERSION`` and is meant
for TestPyPI first, then PyPI after promotion.

What is asserted:
- clean install -> import/console script/CLI smoke -> uninstall leaves nothing
  behind (no module, no dist-info, no console script, no editable ``.pth``);
- editable install and uninstall cleanup;
- hash-pinned install (``--require-hashes``);
- upgrade from a previous published version, when one exists (the history
  compatibility assertion is deferred to #24, which owns
  ``RunHistory.from_json``/schema versioning);
- optional Keras extra smoke on a single Linux/3.12 cell.

The upgrade step is skipped until a previous release exists (issue #22's
first-release bypass); there is no synthetic previous wheel by design.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path

import pytest

pytestmark = pytest.mark.lifecycle

ROOT = Path(__file__).resolve().parents[2]
WHEEL_DIR = Path(os.environ.get("DNNGA_WHEEL_DIR", ROOT / "dist"))
INDEX_URL = os.environ.get("DNNGA_INDEX_URL", "")
PREVIOUS_VERSION = os.environ.get("DNNGA_PREVIOUS_VERSION", "")
CANDIDATE_VERSION = os.environ.get("DNNGA_VERSION", "")
TEST_KERAS = os.environ.get("DNNGA_TEST_KERAS") == "1"

INSTALL_TIMEOUT = 900
PIP_ENV = {
    **os.environ,
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_INPUT": "1",
}


def _run(cmd, cwd=None, timeout=INSTALL_TIMEOUT, env=None):
    """Run a command, failing with captured output on a non-zero exit."""
    result = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd is not None else None,
        env={**PIP_ENV, **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"command failed ({result.returncode}): {' '.join(str(p) for p in cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result


def _make_venv(tmp_path: Path) -> tuple[Path, Path]:
    """Create a clean venv; return (python, bin_dir)."""
    env_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    if os.name == "nt":
        python = env_dir / "Scripts" / "python.exe"
        bin_dir = env_dir / "Scripts"
    else:
        python = env_dir / "bin" / "python"
        bin_dir = env_dir / "bin"
    assert python.exists(), f"venv python not found at {python}"
    _run([python, "-m", "pip", "--version"])
    return python, bin_dir


def _pip(python: Path, *args: str, timeout: int = INSTALL_TIMEOUT):
    return _run([python, "-m", "pip", *args], timeout=timeout)


def _purelib(python: Path) -> Path:
    out = _run(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"]
    ).stdout.strip()
    return Path(out)


def _install_deps(python: Path, *, keras: bool = False) -> None:
    """Install runtime deps from PyPI (they are not part of the wheelhouse)."""
    _pip(python, "install", "--upgrade", "pip", "setuptools", "wheel")
    _pip(python, "install", "numpy>=1.22", "scikit-learn>=1.1")
    if keras:
        _pip(python, "install", "tensorflow>=2.16")


def _wheel() -> Path:
    wheels = sorted(WHEEL_DIR.glob("dnnga-*.whl"))
    assert wheels, f"no dnnga wheel in {WHEEL_DIR}; build with `python -m build`"
    return wheels[-1]


def _wheel_version(wheel: Path) -> str:
    match = re.match(r"dnnga-([^-]+)-", wheel.name)
    assert match, f"unexpected wheel name: {wheel.name}"
    return match.group(1)


def _install_local_wheel(python: Path, wheel: Path, extras: str = "sklearn") -> None:
    """Install the local wheel without resolving dependencies (pre-installed)."""
    requirement = f"dnnga[{extras}]" if extras else "dnnga"
    _pip(
        python,
        "install",
        "--no-index",
        "--find-links",
        str(WHEEL_DIR),
        "--no-deps",
        requirement,
    )


def _install_from_index(python: Path, requirement: str) -> None:
    """Install from the configured index with retry/backoff (TestPyPI lag).

    Dependencies are pre-installed, so ``--no-deps`` avoids TestPyPI index
    shadowing entirely (no ``--extra-index-url``).
    """
    attempts = [10, 20, 40, 80, 160]
    last = None
    for delay in [0, *attempts]:
        if delay:
            time.sleep(delay)
        try:
            _pip(
                python,
                "install",
                "--no-deps",
                "--index-url",
                INDEX_URL,
                requirement,
            )
            return
        except AssertionError as exc:
            last = exc
            text = str(exc)
            if "No matching distribution" not in text and "404" not in text:
                raise
    raise AssertionError(f"could not install {requirement} from {INDEX_URL}: {last}")


def _smoke(python: Path, bin_dir: Path, expected_version: str) -> None:
    """Import, metadata, console script, CLI smoke and pip check."""
    _run(
        [
            python,
            "-c",
            (
                "import dnnga, importlib.metadata as m;"
                f"assert dnnga.__version__ == {expected_version!r}, dnnga.__version__;"
                f"assert m.version('dnnga') == {expected_version!r}, m.version('dnnga');"
                "print('dnnga', dnnga.__version__)"
            ),
        ]
    )

    script = bin_dir / ("dnnga.exe" if os.name == "nt" else "dnnga")
    assert script.exists(), f"console script missing: {script}"
    _run([script, "--version"])

    _run(
        [
            python,
            "-m",
            "dnnga",
            "--mode",
            "synthetic",
            "--population",
            "2",
            "--tournament-size",
            "2",
            "--elitism-count",
            "0",
            "--generations",
            "1",
            "--seed",
            "0",
        ]
    )
    _pip(python, "check")


def _assert_uninstalled(python: Path, bin_dir: Path) -> None:
    """The environment must be indistinguishable from a clean one."""
    result = subprocess.run(
        [str(python), "-c", "import dnnga"],
        capture_output=True,
        text=True,
        env=PIP_ENV,
    )
    assert result.returncode != 0, "'import dnnga' succeeded after uninstall"
    assert "ModuleNotFoundError" in result.stderr or "No module named" in result.stderr

    names = _run(
        [
            python,
            "-c",
            (
                "import importlib.metadata as m;"
                "print(sorted(d.metadata['Name'] for d in m.distributions()))"
            ),
        ]
    ).stdout
    assert "'dnnga'" not in names and '"dnnga"' not in names, names

    purelib = _purelib(python)
    leftovers = sorted(
        p.name
        for p in purelib.iterdir()
        if p.name == "dnnga"
        or p.name.startswith("dnnga-")
        or ("dnnga" in p.name and p.name.startswith("__editable__"))
    )
    assert leftovers == [], f"leftovers in {purelib}: {leftovers}"

    pth = [p for p in purelib.glob("*.pth") if "dnnga" in p.read_text(errors="ignore")]
    assert pth == [], f"stale .pth files: {pth}"

    script = bin_dir / ("dnnga.exe" if os.name == "nt" else "dnnga")
    assert not script.exists(), f"console script left behind: {script}"


# --------------------------------------------------------------------- tests
def test_local_wheel_install_smoke_uninstall(tmp_path):
    """Install -> smoke -> uninstall from a locally built wheel."""
    python, bin_dir = _make_venv(tmp_path)
    wheel = _wheel()
    _install_deps(python)
    _install_local_wheel(python, wheel)
    _smoke(python, bin_dir, _wheel_version(wheel))

    # Extras metadata is declared even though deps were pre-installed.
    requires = _run(
        [python, "-c", "import importlib.metadata as m; print(m.requires('dnnga'))"]
    ).stdout
    assert "scikit-learn" in requires and (
        "extra == 'sklearn'" in requires or 'extra == "sklearn"' in requires
    ), requires

    _pip(python, "uninstall", "-y", "dnnga")
    _assert_uninstalled(python, bin_dir)


def test_editable_install_uninstall(tmp_path):
    """Editable install -> import -> uninstall must not leave a .pth/finder."""
    source = tmp_path / "src_copy"
    source.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    if (ROOT / "README.md").exists():
        shutil.copy2(ROOT / "README.md", source / "README.md")
    shutil.copytree(ROOT / "src" / "dnnga", source / "src" / "dnnga")

    python, bin_dir = _make_venv(tmp_path)
    _install_deps(python)
    _pip(python, "install", "--no-build-isolation", "-e", str(source))
    _run(
        [
            python,
            "-c",
            (
                "import dnnga, importlib.metadata as m;"
                "assert m.version('dnnga') == dnnga.__version__;"
                "print('editable', dnnga.__version__)"
            ),
        ]
    )
    _pip(python, "uninstall", "-y", "dnnga")
    _assert_uninstalled(python, bin_dir)


def test_hash_pinned_install(tmp_path):
    """--require-hashes install of the wheel from a pinned requirements file."""
    python, bin_dir = _make_venv(tmp_path)
    wheel = _wheel()
    _install_deps(python)

    hash_output = _run([python, "-m", "pip", "hash", str(wheel)]).stdout.strip()
    hash_line = next(
        line.strip()
        for line in reversed(hash_output.splitlines())
        if line.strip().startswith("--hash=")
    )
    assert hash_line.startswith("--hash=sha256:"), hash_output
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(
        f"dnnga=={_wheel_version(wheel)} {hash_line}\n", encoding="utf-8"
    )

    _pip(
        python,
        "install",
        "--no-index",
        "--find-links",
        str(WHEEL_DIR),
        "--no-deps",
        "--require-hashes",
        "-r",
        str(requirements),
    )
    _run([python, "-c", "import dnnga; print('pinned', dnnga.__version__)"])
    _pip(python, "uninstall", "-y", "dnnga")
    _assert_uninstalled(python, bin_dir)


@pytest.mark.skipif(
    not (INDEX_URL and PREVIOUS_VERSION and CANDIDATE_VERSION),
    reason="set DNNGA_INDEX_URL, DNNGA_PREVIOUS_VERSION and DNNGA_VERSION",
)
def test_published_upgrade_lifecycle(tmp_path):
    """previous -> upgrade -> candidate -> uninstall from the configured index.

    The history-compatibility assertion is deferred to #24 (RunHistory schema
    versioning); this test verifies that the upgrade keeps the package usable.
    """
    python, bin_dir = _make_venv(tmp_path)
    _install_deps(python)

    _install_from_index(python, f"dnnga[sklearn]=={PREVIOUS_VERSION}")
    _smoke(python, bin_dir, PREVIOUS_VERSION)

    _pip(
        python,
        "install",
        "--upgrade",
        "--no-deps",
        "--index-url",
        INDEX_URL,
        f"dnnga[sklearn]=={CANDIDATE_VERSION}",
    )
    _smoke(python, bin_dir, CANDIDATE_VERSION)

    _pip(python, "uninstall", "-y", "dnnga")
    _assert_uninstalled(python, bin_dir)


@pytest.mark.skipif(not TEST_KERAS, reason="set DNNGA_TEST_KERAS=1")
def test_keras_extra_smoke(tmp_path):
    """Install the wheel with [keras] and build one tiny Keras model."""
    python, bin_dir = _make_venv(tmp_path)
    wheel = _wheel()
    _install_deps(python, keras=True)
    _install_local_wheel(python, wheel, extras="keras")
    _run(
        [
            python,
            "-c",
            (
                "import numpy as np\n"
                "from dnnga import Architecture, TrainingData\n"
                "from dnnga.backends import KerasMLPBackend\n"
                "data = TrainingData(np.zeros((4, 2)), [0, 1, 0, 1], np.zeros((2, 2)), [0, 1])\n"
                "backend = KerasMLPBackend()\n"
                "model = backend.build(Architecture((3,)), data)\n"
                "assert model.count_params() > 0\n"
                "print('keras ok', model.count_params())\n"
            ),
        ]
    )
    _pip(python, "uninstall", "-y", "dnnga")
    _assert_uninstalled(python, bin_dir)
