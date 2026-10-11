"""Run named DuckDB SQL queries against Parquet files or in-memory DataFrames.

SQL files contain blocks introduced by ``-- name: <query_name>``. Queries reference
the views ``loans_all`` (all cleaned loans, every status) and ``loans`` (resolved
modelling population with ``is_adverse`` and ``split``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^--\s*name:\s*([A-Za-z0-9_]+)\s*$", re.MULTILINE)


def parse_sql_file(path: str | Path) -> dict[str, str]:
    """Split a SQL file into ``{name: statement}`` using ``-- name:`` markers."""
    text = Path(path).read_text(encoding="utf-8")
    matches = list(_NAME_RE.finditer(text))
    if not matches:
        raise ValueError(f"No '-- name:' blocks found in {path}")
    queries: dict[str, str] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip().rstrip(";").strip()
        if m.group(1) in queries:
            raise ValueError(f"Duplicate query name '{m.group(1)}' in {path}")
        queries[m.group(1)] = body
    return queries


def connect(sources: Mapping[str, str | Path | pd.DataFrame]) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with one view per source (Parquet path or DataFrame)."""
    con = duckdb.connect(database=":memory:")
    for view, source in sources.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", view):
            raise ValueError(f"Invalid view name: {view!r}")
        if isinstance(source, pd.DataFrame):
            con.register(f"{view}_df", source)
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM {view}_df")
        else:
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Parquet source for view '{view}' not found: {path}")
            escaped = path.as_posix().replace("'", "''")
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_parquet('{escaped}')")
    return con


def run_sql_file(path: str | Path, con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Execute every named query in ``path`` and return the results."""
    results = {}
    for name, sql in parse_sql_file(path).items():
        logger.debug("Running %s:%s", Path(path).name, name)
        results[name] = con.execute(sql).df()
    return results


def run_all(sql_dir: str | Path, con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Run every ``*.sql`` file in ``sql_dir`` (sorted); keys are ``<file_stem>.<query>``."""
    out: dict[str, pd.DataFrame] = {}
    for sql_file in sorted(Path(sql_dir).glob("*.sql")):
        for name, df in run_sql_file(sql_file, con).items():
            out[f"{sql_file.stem}.{name}"] = df
    return out
