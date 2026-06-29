"""Back-compat: the v2ecoli colony/lineage behavior survives as opt-in config.

Before Task 2, XArrayEmitter *hardwired* the v2ecoli colony envelope — it
always reached into ``data["agents"][agent_id]`` and always derived the
storage layout from the lineage-string ``agent_id``. Task 2 makes the generic
"flat" Step the default and preserves that colony behavior behind
``strategy="colony"`` + ``emit_root=["agents", <agent_id>]``.

This test drives an emitter in colony mode with the legacy
``{"agents": {<id>: {...}}, "global_time": t}`` emit shape and asserts the
produced zarr partition still uses the lineage layout
(``generation == len(agent_id)``, a ``parent`` cell, a ``generation=N``
dynamic suffix).
"""

import pytest

pytest.importorskip("xarray")
pytest.importorskip("zarr")

import xarray as xr  # noqa: E402

from bigraph_schema import allocate_core  # noqa: E402


def _colony_config(store, agent_id):
    return {
        "emit": {},
        "out_uri": store,
        "strategy": "colony",
        "emit_root": ["agents", agent_id],
        "transducer": {
            "predicate": [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": 3},
        },
        # v2ecoli-style view: variables live under a per-agent ("listeners",) root.
        "view": [
            {
                "root": ("listeners",),
                "metadata": False,
                "variables": {"mass": [{"path": "listeners/mass", "dtype": "<f8"}]},
            }
        ],
        "writer": {
            "backend": "zarr",
            "store": store,
            "buffers_per_chunk": 1,
            "backend_config": {"format": 3},
        },
        "metadata": {
            "experiment_id": "colony-run",
            "variant": 0,
            "lineage_seed": 0,
            "agent_id": agent_id,
        },
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": {},
        "debug": False,
    }


def _drive_colony_generation(core, store, agent_id):
    """Run one colony generation into `store` with the legacy emit shape."""
    from pbg_emitters.xarray_emitter import XArrayEmitter

    emitter = XArrayEmitter(_colony_config(store, agent_id), core=core)
    part = emitter.partition
    # The colony partition reads `agent_id` from metadata and derives the
    # lineage layout from it — NOT the flat degenerate (generation == 1).
    assert part.agent_id == agent_id
    assert part.generation == len(agent_id)
    assert part.dynamic_suffix.endswith(f"generation={len(agent_id)}")

    for i in range(6):
        emitter.update({
            "global_time": float(i),
            # legacy v2ecoli envelope: data nested under agents.<id>
            "agents": {agent_id: {"listeners": {"mass": 10.0 + i}}},
        })
    try:
        emitter.close(success=True)
    except Exception:
        pass
    return part


def test_xarray_colony_partition_uses_lineage_layout(tmp_path):
    core = allocate_core()
    store = str(tmp_path / "colony.zarr")

    # Mother cell (generation 1), then its daughter (generation 2). The colony
    # writer ENFORCES that the daughter's parent store already exists — proof
    # the lineage layout is fully intact behind `strategy="colony"`.
    _drive_colony_generation(core, store, "0")
    daughter = _drive_colony_generation(core, store, "01")
    assert daughter.generation == 2
    assert daughter.parent.agent_id == "0"

    # The store carries both the mother (gen=1) and daughter (gen=2) variables.
    tree = xr.open_datatree(store, engine="zarr")
    datavar_names = {
        name for node in tree.subtree for name in node.data_vars
    }
    assert any("generation=1" in n for n in datavar_names), datavar_names
    assert any("generation=2" in n for n in datavar_names), datavar_names


def test_xarray_flat_is_the_default_strategy(tmp_path):
    """Sanity: omitting `strategy` yields the flat (generation==1) partition."""
    from pbg_emitters.xarray_emitter import XArrayEmitter

    core = allocate_core()
    store = str(tmp_path / "default.zarr")
    cfg = _colony_config(store, "01")
    cfg.pop("strategy")              # default
    cfg["emit_root"] = []            # flat reads the wired state directly
    # Flat view keyed by the flat port name.
    cfg["view"] = [
        {
            "root": (),
            "metadata": False,
            "variables": {"mass": [{"path": "mass", "dtype": "<f8"}]},
        }
    ]
    emitter = XArrayEmitter(cfg, core=core)
    part = emitter.partition
    assert part.generation == 1  # degenerate single partition, no lineage
    try:
        emitter.close()
    except Exception:
        pass
