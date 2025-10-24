"""Integration tests for the CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from riemann_ml.cli import app


def test_cli_show_config_runs():
    runner = CliRunner()
    result = runner.invoke(app, ["show-config", "--name", "fvm"])
    assert result.exit_code == 0
    assert "fvm" in result.stdout
