"""Shared fixtures for pbg-emitters tests."""

import pytest

from bigraph_schema import allocate_core


@pytest.fixture
def core():
    """A fresh process-bigraph Core for each test.

    Uses ``bigraph_schema.allocate_core()`` — the same factory
    process-bigraph itself uses to build its default core in tests.
    """
    return allocate_core()
