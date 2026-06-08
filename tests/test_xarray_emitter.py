"""Unit tests for pbg_emitters.xarray_emitter."""

import pytest

# Skip the entire file if the [xarray] extra isn't installed.
pytest.importorskip("xarray")
pytest.importorskip("zarr")


def test_base_module_imports():
    from pbg_emitters.xarray_emitter._base import (
        BufferedEmitter, StoragePartition, BlockingExecutor,
    )
    assert BufferedEmitter is not None
    assert StoragePartition is not None
    assert BlockingExecutor is not None


def test_storage_partition_dataclass():
    from pbg_emitters.xarray_emitter._base import StoragePartition

    p = StoragePartition(
        experiment_id="exp1", variant=2, lineage_seed=7, agent_id="01"
    )
    assert p.generation == 2  # len(agent_id)
    assert p.parent.agent_id == "0"


def test_buffered_emitter_inherits_pbg_emitter():
    from pbg_emitters.xarray_emitter._base import BufferedEmitter
    from process_bigraph.emitter import Emitter
    assert issubclass(BufferedEmitter, Emitter)


def test_all_submodules_import():
    """Smoke check: every vendored sub-module imports without error."""
    for name in (
        "transducer", "view", "storage", "writer", "zarr_writer",
        "emit_path", "emit_predicate", "utils",
    ):
        __import__(f"pbg_emitters.xarray_emitter.{name}")


def test_xarray_emitter_imports():
    from pbg_emitters.xarray_emitter import XArrayEmitter
    from process_bigraph.emitter import Emitter
    assert issubclass(XArrayEmitter, Emitter)


def test_xarray_emitter_config_schema_has_expected_keys():
    from pbg_emitters.xarray_emitter import XArrayEmitter
    for key in (
        "emit", "out_uri", "transducer", "view", "writer",
        "metadata", "metadata_keys", "metadata_validators",
        "output_metadata", "debug",
    ):
        assert key in XArrayEmitter.config_schema, f"missing: {key}"


def test_metadata_validators_failure_raises(minimal_xarray_config, core):
    from pbg_emitters.xarray_emitter import XArrayEmitter

    cfg = {
        **minimal_xarray_config,
        "metadata": {"single_daughters": False},
        "metadata_validators": {"single_daughters": True},
    }
    with pytest.raises(ValueError, match="single_daughters"):
        XArrayEmitter(config=cfg, core=core)


def test_empty_metadata_validators_no_op(minimal_xarray_config, core):
    """Empty validators dict => no validation error from the validator path."""
    from pbg_emitters.xarray_emitter import XArrayEmitter

    cfg = {**minimal_xarray_config, "metadata_validators": {}}
    # If construction raised a ValueError from a validator, the test would fail.
    # With empty validators, construction must succeed.
    emitter = XArrayEmitter(config=cfg, core=core)
    assert emitter is not None
    emitter.close()  # tidy up the writer
