"""Generic XArrayEmitter for process-bigraph composites.

Vendored from vivarium-collective/vEcoli@b25ca24 (PR #414 head). Re-rooted
onto process_bigraph.emitter.Emitter. vEcoli-specific metadata keys and
validator checks are config-driven; a downstream builder can reproduce
vEcoli's exact behavior.
"""

try:
    import xarray  # noqa: F401
    import zarr    # noqa: F401
    import zarrs   # noqa: F401
except ImportError as e:
    raise ImportError(
        f"pbg_emitters.xarray_emitter requires the [xarray] extra "
        f"(pip install 'pbg-emitters[xarray]'). (missing: {e.name})"
    ) from e

from pbg_emitters.xarray_emitter.emitter import XArrayEmitter

__all__ = ["XArrayEmitter"]
