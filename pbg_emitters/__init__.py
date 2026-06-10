"""pbg-emitters: focused emitter library for process-bigraph composites.

Each emitter lives behind its own optional-extras group; importing the
package never forces the heavy dependencies. Install only what you need::

    pip install 'pbg-emitters[sqlite]'    # SQLiteEmitter (stdlib only)
    pip install 'pbg-emitters[parquet]'   # ParquetEmitter (duckdb, polars, ...)
"""

# Per-agent emitter-lifecycle registry. Pure stdlib (no heavy deps), so it
# imports unconditionally regardless of which emitter extras are installed.
from pbg_emitters.lifecycle import (
    register_emitter,
    get_emitter,
    unregister_emitter,
    clear_registry,
    registered_agent_ids,
    finalize_emitter_for_agent,
)

try:
    from pbg_emitters.sqlite_emitter import (
        SQLiteEmitter,
        save_simulation_metadata,
        mark_simulation_finished,
        list_simulations,
        load_history,
        load_simulation_metadata,
    )
except ImportError:
    pass  # [sqlite] extra not installed

try:
    from pbg_emitters.parquet_emitter import (
        ParquetEmitter,
        # Bare DuckDB / parquet reader helpers re-exported for downstream readers:
        BlockingExecutor,
        create_duckdb_conn,
        named_idx,
        ndidx_to_duckdb_expr,
        ndlist_to_ndarray,
        list_columns,
        quote_columns,
        union_by_name,
        dataset_sql,
        field_metadata,
        config_value,
        plot_metadata,
        read_stacked_columns,
        num_cells,
        skip_n_gens,
        np_dtype,
        union_pl_dtypes,
        flatten_dict,
        _is_bookkeeping_field,
        _split_structured_arrays,
        json_to_parquet,
        pl_dtype_from_ndarray,
        open_arbitrary_sim_data,
        METADATA_PREFIX,
    )
except ImportError:
    pass  # [parquet] extra not installed

try:
    from pbg_emitters.xarray_emitter import XArrayEmitter  # noqa: F401
except ImportError:
    pass  # [xarray] extra not installed

try:
    from pbg_emitters.run_reader import (  # noqa: F401
        RunReader,
        RunRef,
        by_generation,
        IdNotInCatalog,
        CatalogUnavailable,
    )
except ImportError:
    pass  # polars not installed (unlikely — it's a base dep)
