"""Tests for the CLI surface (the fast smoke path, no torch)."""

from __future__ import annotations

from typer.testing import CliRunner

from gnn_fraud import __version__
from gnn_fraud.cli import app

runner = CliRunner()


def test_info_runs_and_reports_version() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help exits with code 0 and prints usage
    assert "Usage" in result.stdout
