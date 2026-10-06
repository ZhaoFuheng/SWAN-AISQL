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

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

import duckdb

try:  # written by build_wheel.sh: the DuckDB build the bundled binary loads into
    from ._build import DUCKDB_ENGINE, DUCKDB_PLATFORM, DUCKDB_SOURCE_ID, DUCKDB_VERSION
except ImportError:  # source tree without a built wheel
    DUCKDB_ENGINE = DUCKDB_PLATFORM = DUCKDB_SOURCE_ID = DUCKDB_VERSION = None

__version__ = "0.1.1"
__all__ = [
    "connect",
    "load",
    "extension_path",
    "start_embed_server",
    "stop_embed_server",
    "DUCKDB_VERSION",
    "DUCKDB_ENGINE",
    "DUCKDB_SOURCE_ID",
    "DUCKDB_PLATFORM",
]

_EXTENSION = Path(__file__).with_name("aisql.duckdb_extension")
_SETTING_NAME = re.compile(r"^ai_[a-z0-9_]+$")
_OPENAI = "https://api.openai.com"
_DEFAULT_EMBED = "http://localhost:4002"  # the extension's own default (AI_EMBED_URL)
_HOME = Path(os.environ.get("SWAN_AISQL_HOME", Path.home() / ".cache" / "swan-aisql"))


def _embed_alive(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/", timeout=timeout) as resp:
            return resp.status == 200 and json.loads(resp.read() or b"{}").get("status") == "ok"
    except Exception:  # noqa: BLE001 - not listening, or not ours
        return False


def _embed_installed() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def start_embed_server(endpoint: Optional[str] = None, wait: float = 120.0, quiet: bool = False) -> bool:
    """Start the embeddings server (``pip install "swan-aisql[embed]"``) for a local ``endpoint`` unless one
    already answers there, and wait up to ``wait`` seconds for it to listen. The first start downloads the two
    models (about 1.2 GB into the Hugging Face cache) and can outlast ``wait``; the server keeps loading in the
    background and later connections find it. It outlives this process so the next one skips the model load;
    ``stop_embed_server()`` ends it. Returns True once it answers."""
    url = endpoint or os.environ.get("AI_EMBED_URL") or _DEFAULT_EMBED
    if _embed_alive(url):
        return True
    parsed = urlparse(url)
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        raise RuntimeError(f"swan_aisql: cannot start an embeddings server for a remote endpoint ({url})")
    if not _embed_installed():
        raise RuntimeError('swan_aisql: the embeddings server needs the extra: pip install "swan-aisql[embed]"')
    port = parsed.port or 4002
    _HOME.mkdir(parents=True, exist_ok=True)
    log_path = _HOME / f"embed_server_{port}.log"
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "swan_aisql.embed_server", str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    (_HOME / f"embed_server_{port}.pid").write_text(str(proc.pid))
    if not quiet:
        print(f"[swan_aisql] starting the embeddings server on :{port} (pid {proc.pid}, log {log_path})", file=sys.stderr)
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        if _embed_alive(url):
            return True
        if proc.poll() is not None:
            raise RuntimeError(f"swan_aisql: the embeddings server exited with code {proc.returncode}; see {log_path}")
        time.sleep(0.5)
    if not quiet:
        print(
            f"[swan_aisql] the embeddings server is still loading its models after {wait:.0f} s (first start "
            f"downloads them); queries until then order predicates by prompt cost. SET ai_embed_endpoint = "
            f"'{url}' on the connection once it is up, or reconnect.",
            file=sys.stderr,
        )
    return False


def stop_embed_server(endpoint: Optional[str] = None) -> bool:
    """Stop an embeddings server started by this package for ``endpoint`` (default: the default endpoint)."""
    url = endpoint or os.environ.get("AI_EMBED_URL") or _DEFAULT_EMBED
    port = urlparse(url).port or 4002
    pid_file = _HOME / f"embed_server_{port}.pid"
    if not pid_file.is_file():
        return False
    try:
        os.kill(int(pid_file.read_text().strip()), signal.SIGTERM)
    except (ProcessLookupError, ValueError):
        pid_file.unlink(missing_ok=True)
        return False
    pid_file.unlink(missing_ok=True)
    return True


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
    embed_server: Optional[bool] = None,
    embed_wait: float = 120.0,
    **settings: Any,
) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection with SWAN AI-SQL loaded.

    embed_server: start the local embeddings server when nothing answers on the embeddings endpoint
    (``ai_embed_endpoint`` keyword, else ``AI_EMBED_URL``, else localhost:4002). None (the default) does so when
    the ``embed`` extra is installed and the endpoint is local; True insists (raising without the extra); False
    never starts one. ``embed_wait`` bounds the wait for it to listen (see start_embed_server).

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

    embed_url = settings.get("ai_embed_endpoint") or env.get("AI_EMBED_URL") or _DEFAULT_EMBED
    if embed_server is True or (embed_server is None and _embed_installed() and not _embed_alive(embed_url)):
        if urlparse(embed_url).hostname in ("localhost", "127.0.0.1"):
            start_embed_server(embed_url, wait=embed_wait)
        elif embed_server is True:
            raise RuntimeError(f"swan_aisql: cannot start an embeddings server for a remote endpoint ({embed_url})")

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
