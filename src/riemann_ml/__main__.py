"""Entry point to execute the riemann-ml CLI via ``python -m riemann_ml``."""

from __future__ import annotations

from riemann_ml.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
