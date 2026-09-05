"""Tests for package metadata consistency."""

from importlib.metadata import metadata

import pytest
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
    # No upper bound: the extension targets the stable ABI and the test
    # suite runs on 3.13.
    assert spec.contains("3.13")
    assert spec.contains("3.14")


def test_viewer_is_reachable_as_attribute():
    """isoext.viewer is available right after `import isoext`."""
    import isoext

    assert callable(isoext.viewer.embed)
    with pytest.raises(AttributeError):
        getattr(isoext, "no_such_module")
