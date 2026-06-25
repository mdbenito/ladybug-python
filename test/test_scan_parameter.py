from collections.abc import Callable
from typing import Any

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from type_aliases import ConnDB

ScanObjectFactory = Callable[[], Any]


@pytest.mark.parametrize(
    ("make_scan_object", "parameter_name"),
    [
        (lambda: pd.DataFrame({"col1": [1, 2, 3]}), "df"),
        (lambda: pl.DataFrame({"col1": [1, 2, 3]}), "df"),
        (lambda: pa.table({"col1": [1, 2, 3]}), "tab"),
    ],
)
def test_scan_object_param_rejected_outside_scan_source(
    conn_db_empty: ConnDB,
    make_scan_object: ScanObjectFactory,
    parameter_name: str,
) -> None:
    conn, _ = conn_db_empty

    with pytest.raises(
        RuntimeError,
        match=rf"Unsupported parameter type .* for parameter \${parameter_name}.*LOAD FROM / COPY FROM",
    ):
        conn.execute(
            f"RETURN ${parameter_name}",
            {parameter_name: make_scan_object()},
        )


def test_unreferenced_scan_object_param_is_rejected(conn_db_empty: ConnDB) -> None:
    conn, _ = conn_db_empty
    df = pd.DataFrame({"col1": [1, 2, 3]})

    with pytest.raises(
        RuntimeError,
        match=r"Unsupported parameter type DataFrame for parameter \$df",
    ):
        conn.execute("RETURN 1", {"df": df})


def test_scan_object_param_with_regular_param_still_works(
    conn_db_empty: ConnDB,
) -> None:
    conn, _ = conn_db_empty
    df = pd.DataFrame({"col1": [1, 2, 3]})

    result = conn.execute(
        "LOAD FROM $df WHERE col1 = $x RETURN col1",
        {"df": df, "x": 2},
    )

    assert result.get_next() == [2]
    assert not result.has_next()


def test_copy_from_load_scan_object_param_still_works(conn_db_empty: ConnDB) -> None:
    conn, _ = conn_db_empty
    df = pd.DataFrame({"col1": [10, 20, 30]})

    conn.execute("CREATE NODE TABLE T(id INT64 PRIMARY KEY)")
    conn.execute("COPY T FROM (LOAD FROM $df RETURN col1 AS id)", {"df": df})
    result = conn.execute("MATCH (t:T) RETURN t.id ORDER BY t.id")

    assert result.get_next() == [10]
    assert result.get_next() == [20]
    assert result.get_next() == [30]
    assert not result.has_next()


def test_prepare_rejects_scan_object_param_outside_scan_source(
    conn_db_empty: ConnDB,
) -> None:
    conn, _ = conn_db_empty
    df = pd.DataFrame({"col1": [1, 2, 3]})

    with (
        pytest.warns(DeprecationWarning, match="separate prepare"),
        pytest.raises(
            RuntimeError,
            match=r"Unsupported parameter type DataFrame for parameter \$df",
        ),
    ):
        conn.prepare("RETURN $df", {"df": df})


def test_prepared_execute_rejects_unknown_scan_object_param(
    conn_db_empty: ConnDB,
) -> None:
    conn, _ = conn_db_empty
    df = pd.DataFrame({"col1": [1, 2, 3]})

    with pytest.warns(DeprecationWarning, match="separate prepare"):
        prepared = conn.prepare("RETURN $x")

    with pytest.raises(RuntimeError, match=r"does not use \$df"):
        conn.execute(prepared, {"x": 1, "df": df})
