"""``python -m dnnga`` runs the command-line interface."""

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised via a subprocess test
    raise SystemExit(main())
