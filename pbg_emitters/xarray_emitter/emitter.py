"""Generic XArrayEmitter — Zarr-via-Xarray writer for process-bigraph composites.

Vendored from vivarium-collective/vEcoli@b25ca24 (PR #414 head). Re-rooted
onto process_bigraph.emitter.Emitter via the BufferedEmitter base in _base.

The vivarium emit() two-channel handshake collapses into __init__ (one-shot
configuration: build partition, allocate transducer, open writer store)
plus update(state) (per-tick history). finalize() becomes close(success).
metadata_keys and metadata_validators are now config knobs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from pprint import pp
from typing import Any

from process_bigraph.emitter import Emitter

from ._base import BufferedEmitter, StoragePartition
from .transducer import XarrayTransducer
from .storage import XarrayStoragePartition
from .writer import AsyncBufferWriter
from .utils import emitter_arg_error


class XArrayEmitter(BufferedEmitter):
    """Generic XArrayEmitter. See module docstring for the lifecycle mapping."""

    config_schema = {
        **Emitter.config_schema,
        "out_uri":             {"_type": "string", "_default": ""},
        "transducer":          {"_type": "map", "_default": {}},
        "view":                {"_type": "list", "_default": []},
        "writer":              {"_type": "map", "_default": {}},
        "metadata":            {"_type": "map", "_default": {}},
        "metadata_keys":       {"_type": "list[string]", "_default": []},
        "metadata_validators": {"_type": "map", "_default": {}},
        "output_metadata":     {"_type": "map", "_default": {}},
        "debug":               {"_type": "boolean", "_default": False},
    }

    def __init__(self, config: dict[str, Any], core: Any) -> None:
        self.validate_config(config)
        self.debug: bool = bool(config.get("debug", False))
        self._metadata_keys: list[str] = list(config.get("metadata_keys") or [])
        self._metadata_validators: dict[str, Any] = dict(
            config.get("metadata_validators") or {}
        )
        self._closed: bool = False

        # Unconditionally build the transducer and writer. Tests that only
        # exercise the validator path should supply a minimum-valid transducer
        # config (see the `minimal_xarray_config` fixture in tests/conftest.py).
        self.transducer = XarrayTransducer(config, debug=self.debug)
        self.writer = AsyncBufferWriter.dispatch(config["writer"])

        # Call the BufferedEmitter base __init__ AFTER setting up attributes
        # (per the upstream warning that __init__ must be called at the end).
        BufferedEmitter.__init__(self, config, core)

        # vivarium's "configuration" emit happens here at construction time
        # when metadata is available. validate_metadata is called before
        # transducer.alloc so a validator mismatch raises ValueError early.
        metadata = dict(config.get("metadata") or {})
        if metadata:
            self.validate_metadata(metadata)
            partition = self.extract_partition(metadata)
            extracted_meta = self.extract_metadata(metadata)
            coords = config.get("output_metadata") or {}
            self.transducer.alloc(
                partition=partition, metadata=extracted_meta, coords=coords,
            )
            self.writer.open_store(self.transducer.buffer)

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> None:
        for key in ("transducer", "view", "writer"):
            if key not in config:
                raise KeyError(emitter_arg_error(
                    cls, "Missing argument", f'"{key}": ...'
                ))
        match config.get("debug", False):
            case bool():
                pass
            case debug:
                raise TypeError(emitter_arg_error(
                    cls, "Invalid argument", f'"debug": {debug}'
                ))

    def validate_metadata(self, metadata: dict[str, Any]) -> None:
        """Check config['metadata_validators'] against the supplied metadata."""
        for key, expected in self._metadata_validators.items():
            actual = metadata.get(key)
            if bool(actual) != bool(expected):
                raise ValueError(
                    f"\n  Metadata field unsupported by {type(self).__name__}:"
                    f'\n    {{"{key}": {actual}}}'
                )

    def extract_partition(self, metadata: dict[str, Any]) -> XarrayStoragePartition:
        return XarrayStoragePartition.cast(
            BufferedEmitter.extract_partition(self, metadata)
        )

    def extract_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Return the subset of `metadata` named by `self._metadata_keys`."""
        keys = self._metadata_keys
        if not keys:
            selected = dict(metadata)
        else:
            selected = {k: metadata[k] for k in keys if k in metadata}
        # Reduce to JSON-friendly types.
        for k, v in list(selected.items()):
            match v:
                case Path():
                    selected[k] = str(v)
                case datetime():
                    selected[k] = str(v.astimezone())
        if self.debug:
            hline = "-" * 79
            print(f"\nMetadata:\n{hline}")
            pp(selected)
            print(hline)
        return selected

    @property
    def partition(self) -> XarrayStoragePartition:
        assert self.transducer is not None
        return self.transducer.buffer.partition

    def flush(self, *, final: bool = False) -> None:
        if self.transducer is None or self.writer is None:
            return
        if self.writer._buffer is None:
            # store not yet opened (no metadata was provided at construction)
            return
        self.writer.write(self.transducer, final=final)

    def update(self, state: dict[str, Any]) -> dict:
        """Buffer one history row via the transducer; flush when buffer fills.

        The ``state`` dict must be shaped to match the underlying vEcoli/vivarium
        storage layout that the transducer expects: a top-level ``"agents"`` key
        keyed by ``agent_id`` (matching ``config["metadata"]["agent_id"]``) plus a
        top-level ``"time"`` key. Example::

            emitter.update({
                "time": 1.0,
                "agents": {"1": {"listeners": {"global_time": 1.0}}},
            })

        This nesting is inherited from PR #414's ``XarrayTransducer.write()``
        implementation (see ``transducer.py``); it's the cost of vendoring the
        upstream transducer/view machinery unchanged.
        """
        if self.transducer is None:
            return {}
        if not self.transducer.step(state):
            self.flush()
            self.transducer.shift()
            assert self.transducer.step(state)
        return {}

    def close(self, success: bool = False) -> None:
        """Flush final batch, finalize the buffered base, close the writer."""
        if self._closed:
            return
        self.flush(final=True)
        if self.writer is not None and self.writer._buffer is not None:
            if success:
                self.writer.mark_success()
            self.writer.close()
        self.finalized = True
        self._closed = True

    def _finalize(self, *, success: bool) -> None:
        """Adapter for the BufferedEmitter abstract method."""
        self.close(success=success)

    def query(self, paths=None, query=None) -> Any:
        """Open the written Zarr store and return an xarray DataTree."""
        if not self._closed:
            self.flush(final=False)
        import xarray as xr
        assert self.writer is not None
        tree = xr.open_datatree(self.writer.out_uri, engine="zarr")
        select = paths if paths is not None else query
        if isinstance(select, list):
            tree = tree[select] if hasattr(tree, "__getitem__") else tree
        return tree

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
