"""Tests for the CLI module."""

import pytest
from click.testing import CliRunner
from raw2ultra.cli import cli


def test_cli_version():
    """Test that the CLI shows version information."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert '0.1.0' in result.output


def test_cli_help():
    """Test that the CLI shows help information."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'raw2ultra' in result.output


def test_info_command():
    """Test the info command."""
    runner = CliRunner()
    result = runner.invoke(cli, ['info'])
    assert result.exit_code == 0
    assert 'raw2ultra version 0.1.0' in result.output


def test_verbose_flag():
    """Test the verbose flag."""
    runner = CliRunner()
    result = runner.invoke(cli, ['--verbose', 'info'])
    assert result.exit_code == 0
    assert 'raw2ultra v0.1.0' in result.output 