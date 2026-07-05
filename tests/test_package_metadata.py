"""Tests for package metadata consistency."""

from importlib.metadata import metadata

from packaging.specifiers import SpecifierSet


def test_requires_python_matches_source_syntax():
    """The declared Python floor must match what the source can import.

    isoext.utils uses `int | list[int]` and isoext.sdf uses `list[float]`
    in signatures that are evaluated at import time, so the package cannot
    import on Python < 3.10. Claiming an older floor breaks installs.
    """
    spec = SpecifierSet(metadata("isoext")["Requires-Python"])

    assert not spec.contains("3.8")
    assert not spec.contains("3.9")
    assert spec.contains("3.10")
    assert spec.contains("3.12")
