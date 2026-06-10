"""RunReader: emitter-agnostic reader for stored simulation runs.

Returns a uniform per-tick series (with generation + time) for any
observable, over parquet / sqlite / xarray-zarr stores.

Imports of heavy dependencies (duckdb, xarray, zarr) are lazy: they happen
inside the backend branches so importing this module does not force extras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass
class RunRef:
    """Reference to a stored run.

    Attributes:
        store: Path to the run's store (hive dir | .db file | .zarr root).
        kind:  Backend kind — ``"parquet"``, ``"sqlite"``, or ``"xarray"``.
               Auto-detected from ``store`` if ``None``.
    """

    store: str
    kind: str | None = None


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------


def _detect_kind(store: str) -> str:
    """Infer backend from the store path."""
    p = Path(store)
    # SQLite: an existing file with .db / .sqlite extension, or just the ext
    if p.is_file() and p.suffix in {".db", ".sqlite"}:
        return "sqlite"
    if p.suffix in {".db", ".sqlite"}:
        return "sqlite"
    # XArray: .zarr directory (suffix or path ending)
    if p.suffix == ".zarr" or str(store).endswith(".zarr"):
        return "xarray"
    if p.is_dir() and (p / ".zgroup").exists():
        return "xarray"
    # Parquet: directory containing a 'history' hive sub-tree
    if p.is_dir() and (p / "history").exists():
        return "parquet"
    if p.is_dir() and any(p.glob("**/history")):
        return "parquet"
    # Default fallbacks
    if p.is_dir():
        return "parquet"
    return "sqlite"


# ---------------------------------------------------------------------------
# Cumulative-time helper
# ---------------------------------------------------------------------------


def _cumulative_time(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``abs_time`` column: gen-local ``time`` stitched into one axis.

    Each generation's time resets to 0.  ``abs_time`` appends each generation
    at ``prev_max_time + 1`` so the axis is strictly increasing across gens.

    Args:
        df: Must have ``generation`` (int) and ``time`` (float) columns.

    Returns:
        ``df`` with an additional ``abs_time`` (Float64) column.
    """
    if df.is_empty():
        return df.with_columns(pl.lit(0.0).cast(pl.Float64).alias("abs_time"))

    gens = sorted(df["generation"].unique().to_list())
    offset = 0.0
    offsets: dict = {}
    for g in gens:
        offsets[g] = offset
        gmax = df.filter(pl.col("generation") == g)["time"].max()
        offset += (gmax or 0.0) + 1.0

    gen_dtype = df.schema["generation"]
    off_df = pl.DataFrame({
        "generation": pl.Series(list(offsets.keys()), dtype=gen_dtype),
        "_off": pl.Series(list(offsets.values()), dtype=pl.Float64),
    })
    return (
        df.join(off_df, on="generation", how="left")
        .with_columns((pl.col("time") + pl.col("_off")).alias("abs_time"))
        .drop("_off")
    )


# ---------------------------------------------------------------------------
# by_generation helper
# ---------------------------------------------------------------------------


def by_generation(df: pl.DataFrame) -> dict[int, pl.DataFrame]:
    """Split a series DataFrame into per-generation slices.

    Args:
        df: DataFrame as returned by :py:meth:`RunReader.series` — must have
            a ``generation`` column.

    Returns:
        Mapping from generation integer to the corresponding subset of ``df``.
    """
    return {
        int(g): df.filter(pl.col("generation") == g)
        for g in sorted(df["generation"].unique().to_list())
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dig(state: dict, dotted_path: str):
    """Navigate a nested dict by dotted path.

    Example::

        _dig({"a": {"b": 1}}, "a.b")  # → 1

    Raises:
        KeyError: if any segment of ``dotted_path`` is missing.
    """
    parts = dotted_path.split(".")
    val = state
    for p in parts:
        if not isinstance(val, dict):
            raise KeyError(
                f"Cannot dig into {type(val).__name__!r} at key {p!r}"
            )
        if p not in val:
            raise KeyError(p)
        val = val[p]
    return val


def _flatten_paths(d: dict, prefix: str = "") -> list[str]:
    """Flatten a nested dict to sorted dotted leaf-key paths.

    Example::

        _flatten_paths({"a": {"b": 1}, "c": 2})  # → ["a.b", "c"]
    """
    paths: list[str] = []
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            paths.extend(_flatten_paths(v, full))
        else:
            paths.append(full)
    return paths


# ---------------------------------------------------------------------------
# RunReader
# ---------------------------------------------------------------------------

_PARQUET_ID_COLS = frozenset(
    {"experiment_id", "variant", "lineage_seed", "generation", "agent_id", "time"}
)
_SQLITE_ID_FIELDS = frozenset({"generation", "global_time"})


class RunReader:
    """Emitter-agnostic reader that returns a uniform per-tick series.

    Open with :py:meth:`open`, then call:

    * :py:meth:`observables` — list available observable ids,
    * :py:meth:`generations` — sorted generation indices,
    * :py:meth:`series` — polars DataFrame with ``generation``, ``time``,
      ``abs_time``, ``value``.

    Example::

        r = RunReader.open("/path/to/run")
        df = r.series("listeners.mass.cell_mass")
        by_gen = by_generation(df)
    """

    def __init__(self, ref: RunRef) -> None:
        self._ref = ref
        self._kind: str = ref.kind or _detect_kind(ref.store)

    @classmethod
    def open(cls, store: str | RunRef, kind: str | None = None) -> "RunReader":
        """Open a run store and return a :py:class:`RunReader`.

        Args:
            store: Path to the run store, or a :py:class:`RunRef`.
            kind:  Force backend — ``"sqlite"``, ``"parquet"``, or
                   ``"xarray"``.  Auto-detected if ``None``.
        """
        if isinstance(store, RunRef):
            return cls(store)
        return cls(RunRef(store=str(store), kind=kind))

    @property
    def kind(self) -> str:
        """Detected or forced backend kind."""
        return self._kind

    # ------------------------------------------------------------------
    # Public interface — dispatch on kind
    # ------------------------------------------------------------------

    def observables(self) -> list[str]:
        """Return sorted list of available observable ids for this run."""
        if self._kind == "sqlite":
            return self._sqlite_observables()
        if self._kind == "parquet":
            return self._parquet_observables()
        if self._kind == "xarray":
            return self._xarray_observables()
        raise ValueError(f"Unknown kind: {self._kind!r}")

    def generations(self) -> list[int]:
        """Return sorted list of generation indices present in the run."""
        if self._kind == "sqlite":
            return self._sqlite_generations()
        if self._kind == "parquet":
            return self._parquet_generations()
        if self._kind == "xarray":
            return self._xarray_generations()
        raise ValueError(f"Unknown kind: {self._kind!r}")

    def series(self, observable: str) -> pl.DataFrame:
        """Return a per-tick series for ``observable``.

        Args:
            observable: Backend-native observable id as listed by
                :py:meth:`observables` (e.g. ``"listeners.mass.cell_mass"``).

        Returns:
            Polars DataFrame with columns::

                generation : Int64   — generation index (first-class from backend)
                time       : Float64 — gen-local time (resets each generation)
                abs_time   : Float64 — cumulative time (stitched across gens)
                value      : Float64 — observable value

            Ordered by ``(generation, time)``.

        Raises:
            KeyError: If ``observable`` is not available in this run.
        """
        if self._kind == "sqlite":
            return self._sqlite_series(observable)
        if self._kind == "parquet":
            return self._parquet_series(observable)
        if self._kind == "xarray":
            return self._xarray_series(observable)
        raise ValueError(f"Unknown kind: {self._kind!r}")

    # ------------------------------------------------------------------
    # SQLite backend
    # ------------------------------------------------------------------

    def _sqlite_rows(self) -> list[tuple]:
        """Read all history rows: [(step, global_time, state_dict), ...]."""
        import sqlite3
        import json

        con = sqlite3.connect(self._ref.store)
        try:
            rows = con.execute(
                "SELECT step, global_time, state FROM history ORDER BY step"
            ).fetchall()
        finally:
            con.close()
        return [(step, gtime, json.loads(state)) for step, gtime, state in rows]

    def _sqlite_observables(self) -> list[str]:
        rows = self._sqlite_rows()
        if not rows:
            return []
        _, _, state = rows[0]
        obs_state = {k: v for k, v in state.items() if k not in _SQLITE_ID_FIELDS}
        return sorted(_flatten_paths(obs_state))

    def _sqlite_generations(self) -> list[int]:
        rows = self._sqlite_rows()
        gens = {
            state["generation"]
            for _, _, state in rows
            if "generation" in state
        }
        return sorted(int(g) for g in gens)

    def _sqlite_series(self, observable: str) -> pl.DataFrame:
        rows = self._sqlite_rows()
        if not rows:
            return pl.DataFrame({
                "generation": pl.Series([], dtype=pl.Int64),
                "time": pl.Series([], dtype=pl.Float64),
                "abs_time": pl.Series([], dtype=pl.Float64),
                "value": pl.Series([], dtype=pl.Float64),
            })
        # Validate — check first row's state for the observable
        _, _, first_state = rows[0]
        try:
            _dig(first_state, observable)
        except KeyError:
            raise KeyError(observable)

        data = []
        for _, gtime, state in rows:
            gen = state.get("generation")
            if gen is None:
                continue
            try:
                val = _dig(state, observable)
            except KeyError:
                continue
            data.append({
                "generation": int(gen),
                "time": float(gtime if gtime is not None else 0.0),
                "value": float(val),
            })
        df = pl.DataFrame(data).sort(["generation", "time"])
        return _cumulative_time(df)

    # ------------------------------------------------------------------
    # Parquet backend
    # ------------------------------------------------------------------

    def _parquet_conn_sql(self):
        """Return (duckdb_conn, history_sql) for this store."""
        # Lazy import — only when parquet backend is needed
        from pbg_emitters.parquet_emitter import create_duckdb_conn, dataset_sql

        p = Path(self._ref.store)
        out_dir = str(p.parent)
        exp_id = p.name
        conn = create_duckdb_conn()
        history_sql, _, _ = dataset_sql(out_dir, [exp_id])
        return conn, history_sql

    def _parquet_observables(self) -> list[str]:
        from pbg_emitters.parquet_emitter import list_columns

        conn, history_sql = self._parquet_conn_sql()
        all_cols = list_columns(conn, history_sql)
        obs_cols = [c for c in all_cols if c not in _PARQUET_ID_COLS]
        return sorted(c.replace("__", ".") for c in obs_cols)

    def _parquet_generations(self) -> list[int]:
        conn, history_sql = self._parquet_conn_sql()
        result = conn.sql(
            f"SELECT DISTINCT generation FROM ({history_sql}) ORDER BY generation"
        ).pl()
        return result["generation"].cast(pl.Int64).to_list()

    def _parquet_series(self, observable: str) -> pl.DataFrame:
        from pbg_emitters.parquet_emitter import (
            list_columns,
            read_stacked_columns,
            quote_columns,
        )

        col_name = observable.replace(".", "__")
        conn, history_sql = self._parquet_conn_sql()

        # Validate — check column exists in schema
        available = list_columns(conn, history_sql, col_name)
        if not available:
            raise KeyError(observable)

        quoted = quote_columns(col_name)
        df = read_stacked_columns(history_sql, [quoted], conn=conn)
        # df has columns: [col_name, experiment_id, variant, lineage_seed,
        #                  generation, agent_id, time]
        df = (
            df.select(["generation", "time", col_name])
            .rename({col_name: "value"})
            .with_columns(
                pl.col("generation").cast(pl.Int64),
                pl.col("time").cast(pl.Float64),
                pl.col("value").cast(pl.Float64),
            )
            .sort(["generation", "time"])
        )
        return _cumulative_time(df)

    # ------------------------------------------------------------------
    # XArray backend
    # ------------------------------------------------------------------

    def _xarray_open(self):
        """Open the DataTree from the zarr store (lazy import)."""
        import xarray as xr

        return xr.open_datatree(self._ref.store, engine="zarr")

    def _xarray_gen_info(self, dt) -> list[tuple[int, str, str]]:
        """Return list of (gen_int, time_coo_name, time_var_name) sorted by gen.

        Reads from root node coords named ``emitstep_gen=N``.
        """
        # Import constants lazily — only executed on xarray backend
        from pbg_emitters.xarray_emitter.storage import (
            TIME_COO_PREFIX,
            TIME_VAR_PREFIX,
        )

        root_ds = dt["/"].ds
        gen_info = []
        for coo_name in root_ds.coords:
            if coo_name.startswith(TIME_COO_PREFIX):
                # e.g. "emitstep_gen=1" → extract "1" after last "="
                gen = int(coo_name.rsplit("=", 1)[-1])
                time_var = f"{TIME_VAR_PREFIX}gen={gen}"
                gen_info.append((gen, coo_name, time_var))
        return sorted(gen_info, key=lambda x: x[0])

    def _xarray_observables(self) -> list[str]:
        dt = self._xarray_open()
        obs = []
        for path in dt.groups:
            if path == "/":
                continue
            node_ds = dt[path].ds
            has_gen_vars = any(
                v.startswith("generation=") for v in node_ds.data_vars
            )
            if has_gen_vars:
                obs.append(path.lstrip("/").replace("/", "."))
        return sorted(obs)

    def _xarray_generations(self) -> list[int]:
        dt = self._xarray_open()
        return [g for g, _, _ in self._xarray_gen_info(dt)]

    def _xarray_series(self, observable: str) -> pl.DataFrame:
        dt = self._xarray_open()
        node_path = "/" + observable.replace(".", "/")

        if node_path not in dt.groups:
            raise KeyError(observable)

        node_ds = dt[node_path].ds
        root_ds = dt["/"].ds
        gen_info = self._xarray_gen_info(dt)

        rows = []
        for gen, _coo_name, time_var in gen_info:
            var_name = f"generation={gen}"
            if var_name not in node_ds.data_vars:
                continue
            if time_var not in root_ds.data_vars:
                continue
            times = root_ds[time_var].values.ravel()
            values = node_ds[var_name].values.ravel()
            for t, v in zip(times, values):
                rows.append({
                    "generation": gen,
                    "time": float(t),
                    "value": float(v),
                })

        if not rows:
            # Observable node exists but no generation variables
            raise KeyError(observable)

        df = pl.DataFrame(rows).sort(["generation", "time"])
        return _cumulative_time(df)
