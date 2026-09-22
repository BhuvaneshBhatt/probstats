"""Static release and source-hygiene contracts."""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src" / "probstats"


def _source_trees():
    for path in SOURCE.rglob("*.py"):
        yield path, ast.parse(path.read_text())


def test_gplv3_release_metadata_and_license():
    project = (ROOT / "pyproject.toml").read_text()
    license_text = (ROOT / "LICENSE").read_text()
    assert 'license = "GPL-3.0-only"' in project
    assert 'license-files = ["LICENSE"]' in project
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3, 29 June 2007" in license_text


def test_pypi_workflow_builds_smokes_and_uses_trusted_publishing():
    workflow = (ROOT / ".github" / "workflows" / "publish-pypi.yml").read_text()
    assert "python -m build" in workflow
    assert "Smoke-test wheel" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow


def test_source_distribution_manifest_retains_release_assets():
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "recursive-include .github *.yml *.yaml" in manifest
    assert "include CHANGELOG.md" in manifest
    assert "recursive-include docs *.md" in manifest
    assert "recursive-include tools *.py" in manifest


def test_readme_has_no_relative_markdown_links():
    readme = (ROOT / "README.md").read_text()
    links = re.findall(r"\[[^]]+\]\(([^)]+)\)", readme)
    relative = [
        link for link in links if not link.startswith(("http://", "https://", "#"))
    ]
    assert relative == []


def test_source_has_no_broad_exception_handlers():
    for path, tree in _source_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name):
                assert node.type.id != "Exception", (
                    f"broad exception handler in {path}:{node.lineno}"
                )


def test_source_uses_exceptions_instead_of_assert_statements():
    for path, tree in _source_trees():
        asserts = [
            node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)
        ]
        assert asserts == [], f"production assert in {path}: {asserts}"
