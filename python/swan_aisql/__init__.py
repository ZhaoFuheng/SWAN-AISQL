"""SWAN AI-SQL for Python: ``pip install swan-aisql`` then ``swan_aisql.connect()``.

The wheel bundles the ``aisql`` DuckDB extension for one platform and one exact DuckDB build (the
``duckdb`` version pinned in the package metadata). ``connect()`` opens a DuckDB connection, loads the
extension and applies the AI settings; the connection it returns is an ordinary ``DuckDBPyConnection``.

    import swan_aisql
    con = swan_aisql.connect(endpoint="https://api.openai.com", api_key=os.environ["OPENAI_API_KEY"])
    con.sql("SELECT ai_filter('Is this review positive? ' || review) FROM reviews").show()

Settings can also come from the environment (``AI_PROXY_URL``, ``AI_API_KEY``, ``AI_MODEL``, ...; the
extension reads them when it loads) and every ``SET ai_*`` option is accepted as a keyword argument.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional

import duckdb

try:  # written by build_wheel.sh: the DuckDB build the bundled binary loads into
    from ._build import DUCKDB_ENGINE, DUCKDB_PLATFORM, DUCKDB_SOURCE_ID, DUCKDB_VERSION
except ImportError:  # source tree without a built wheel
    DUCKDB_ENGINE = DUCKDB_PLATFORM = DUCKDB_SOURCE_ID = DUCKDB_VERSION = None

__version__ = "0.1.0"
__all__ = ["connect", "load", "extension_path", "DUCKDB_VERSION", "DUCKDB_ENGINE", "DUCKDB_SOURCE_ID", "DUCKDB_PLATFORM"]

_EXTENSION = Path(__file__).with_name("aisql.duckdb_extension")
_SETTING_NAME = re.compile(r"^ai_[a-z0-9_]+$")
_OPENAI = "https://api.openai.com"


def extension_path() -> str:
    """Path of the bundled ``aisql.duckdb_extension``."""
    if not _EXTENSION.is_file():
        raise RuntimeError(
            "swan_aisql: no aisql.duckdb_extension next to this module. Install the wheel (pip install swan-aisql) "
            "or build one with python/build_wheel.sh from the SWAN-AISQL checkout."
        )
    return str(_EXTENSION)


def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value).replace("'", "''")
    return f"'{text}'"


def _check_engine(con: duckdb.DuckDBPyConnection) -> None:
    if DUCKDB_ENGINE is None:
        return
    engine, platform = con.execute(
        "SELECT v.library_version, p.platform FROM pragma_version() v, pragma_platform() p"
    ).fetchone()
    if engine != DUCKDB_ENGINE or platform != DUCKDB_PLATFORM:
        raise RuntimeError(
            f"swan_aisql {__version__} bundles the extension for DuckDB {DUCKDB_ENGINE} on {DUCKDB_PLATFORM} "
            f"(pip: duckdb=={DUCKDB_VERSION}); this process runs DuckDB {engine} on {platform}. "
            f"Install the pinned engine: pip install --pre 'duckdb=={_required_duckdb()}'"
        )


def _required_duckdb() -> str:
    try:
        from importlib.metadata import requires

        for req in requires("swan-aisql") or []:
            if req.startswith("duckdb"):
                return req.split("==", 1)[1].split(";")[0].strip()
    except Exception:  # noqa: BLE001 - best effort for the error message
        pass
    return DUCKDB_VERSION or "<pinned version>"


def load(con: duckdb.DuckDBPyConnection, **settings: Any) -> duckdb.DuckDBPyConnection:
    """Load the extension into an existing connection (opened with ``allow_unsigned_extensions``) and apply
    ``SET ai_<name> = value`` for every keyword argument."""
    _check_engine(con)
    con.load_extension(extension_path())
    for name, value in settings.items():
        if value is None:
            continue
        if not _SETTING_NAME.match(name):
            raise ValueError(f"swan_aisql: '{name}' is not an ai_* setting")
        con.execute(f"SET {name} = {_literal(value)}")
    return con


def connect(
    database: str = ":memory:",
    read_only: bool = False,
    config: Optional[Mapping[str, Any]] = None,
    *,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    price_input: Optional[float] = None,
    price_output: Optional[float] = None,
    price_cached: Optional[float] = None,
    **settings: Any,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with SWAN AI-SQL loaded.

    endpoint, model, api_key and reasoning_effort map to ``ai_endpoint`` / ``ai_model`` / ``ai_api_key`` /
    ``ai_reasoning_effort``; price_input / price_output / price_cached (USD per 1M tokens) let ``ai_usage().cost_usd``
    price a direct endpoint's calls; any other ``ai_*`` setting is passed through as a keyword (``ai_concurrency=50``).

    Defaults: ``endpoint`` falls back to ``AI_PROXY_URL`` (read by the extension itself) and, when that is unset
    but ``OPENAI_API_KEY`` is, to OpenAI directly (``https://api.openai.com``); ``api_key`` falls back to
    ``AI_API_KEY`` and then ``OPENAI_API_KEY``; ``reasoning_effort`` defaults to ``low`` against OpenAI (the same
    setting the repository's litellm proxy pins for the default model) and is otherwise left to the endpoint.
    """
    options = {"allow_unsigned_extensions": "true"}
    if config:
        options.update(config)
    con = duckdb.connect(database, read_only=read_only, config=options)

    env = os.environ
    if endpoint is None and not env.get("AI_PROXY_URL") and env.get("OPENAI_API_KEY"):
        endpoint = _OPENAI
    if api_key is None and not env.get("AI_API_KEY"):
        api_key = env.get("OPENAI_API_KEY")
    direct = (endpoint or env.get("AI_PROXY_URL", "")).rstrip("/") == _OPENAI
    if reasoning_effort is None and direct:
        reasoning_effort = "low"

    resolved: dict[str, Any] = {
        "ai_endpoint": endpoint,
        "ai_model": model,
        "ai_api_key": api_key,
        "ai_reasoning_effort": reasoning_effort,
        "ai_price_input": price_input,
        "ai_price_output": price_output,
        "ai_price_cached": price_cached,
    }
    resolved.update(settings)
    return load(con, **resolved)
