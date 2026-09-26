#!/usr/bin/env python3
"""aisql SERVER CACHE (layer 2): persistent cross-process LLM cache proxy (DuckDB-backed).

Sits between the aisql engine and any OpenAI-compatible upstream so repeated runs never re-pay
API usage. Layer 1 (the in-process local cache, `SET ai_local_cache`) dedups within a process;
this server dedups across processes, queries, and runs.

Running the benchmark repeatedly should not repeatedly pay OpenAI. This proxy caches successful
chat-completion responses in a single DuckDB database file, keyed by the SHA-256 of the full request
(model + messages + response_format/schema + temperature ...). On a cache HIT it replays the stored
response body + cost header and (by default) sleeps the originally measured latency, so repeat runs
cost $0 at OpenAI but keep realistic latency and still report the would-be cost.

Cache row:   hash(request) -> (request, output, status, headers, latency, cost, recorded_at)
The original request is stored too, so the cache is inspectable with plain SQL, e.g.
    duckdb serve/.llm_cache.duckdb "SELECT request, latency, cost FROM cache LIMIT 5"
A single DuckDB file scales to large benchmarks far better than one file per entry.

This is a GLOBAL, cross-run cache -- distinct from DuckDB's in-process, per-query response cache.
Chain them:
    aisql --(SET ai_endpoint=:4001)--> ai_cache_server --(CACHE_UPSTREAM=:4000)--> litellm --> provider

Env:
  CACHE_PROXY_PORT        (default 4001)   port this proxy listens on
  CACHE_UPSTREAM          (default http://127.0.0.1:4000)  the litellm proxy to forward misses to
  CACHE_DB                (default serve/.llm_cache.duckdb)  persistent DuckDB store file
  CACHE_SIMULATE_LATENCY  (default 1)      sleep the recorded latency on hits (0 = return instantly)
"""
import atexit, datetime, hashlib, json, os, sys, threading, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import duckdb
except ImportError:
    sys.exit("cache_proxy requires the duckdb python package:  pip install duckdb")

PORT      = int(os.environ.get("CACHE_PROXY_PORT", "4001"))
UPSTREAM  = os.environ.get("CACHE_UPSTREAM", "http://127.0.0.1:4000").rstrip("/")
# Embeddings are served by a separate backend (sentence-transformers); route /v1/embeddings there.
EMBED_UPSTREAM = os.environ.get("CACHE_EMBED_UPSTREAM", "http://127.0.0.1:4002").rstrip("/")
# TypeSafe System One (SET ai_typesafe): /v1/systemone goes straight to the TypeSafe API, cached
# like chat. It returns no cost header; cost = input tokens x CACHE_TYPESAFE_PRICE_INPUT ($/Mtok).
TYPESAFE_UPSTREAM = os.environ.get("CACHE_TYPESAFE_UPSTREAM", "https://api.typesafe.ai").rstrip("/")
TYPESAFE_PRICE_INPUT = float(os.environ.get("CACHE_TYPESAFE_PRICE_INPUT", "0.042"))
CACHE_DB  = os.environ.get("CACHE_DB",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), ".llm_cache.duckdb"))
SIMULATE  = os.environ.get("CACHE_SIMULATE_LATENCY", "1") not in ("0", "false", "")
KEEP_HEADERS = ("Content-Type", "x-litellm-response-cost")

# One DuckDB connection, serialized by a lock (DB ops are tiny; network dominates, and we never
# hold the lock across an upstream call). Autocommit + WAL make each insert durable across restarts.
_con = duckdb.connect(CACHE_DB)
_con.execute("""CREATE TABLE IF NOT EXISTS cache(
    key      VARCHAR PRIMARY KEY,   -- sha256 of the canonical request
    request  VARCHAR,               -- original request body (JSON)
    output   VARCHAR,               -- response body (JSON)
    status   INTEGER,
    headers  VARCHAR,               -- replayed response headers (JSON; incl. cost header)
    latency  DOUBLE,                -- measured upstream latency (seconds)
    cost     DOUBLE,                -- parsed x-litellm-response-cost (for SQL rollups)
    recorded_at TIMESTAMP           -- UTC time the upstream response was recorded
)""")
# Replayed latency reproduces the provider's load AT RECORDING TIME, so a cross-system latency
# comparison is only fair between entries recorded under similar conditions -- this column is what
# makes that checkable. Stores that predate it are migrated in place; their rows keep NULL (unknown).
_con.execute("ALTER TABLE cache ADD COLUMN IF NOT EXISTS recorded_at TIMESTAMP")
_db_lock = threading.Lock()
atexit.register(lambda: _con.close())
_stats = {"hits": 0, "misses": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0}


def _accumulate_tokens(body_str):
    # Sum chat prompt/completion tokens so the harness can read clean chat-only token deltas.
    # Embeddings never reach this proxy (routed straight to the embed server), so this is chat-only.
    try:
        u = (json.loads(body_str) or {}).get("usage") or {}
        pt = int(u.get("prompt_tokens", 0) or 0)
        ct = int(u.get("completion_tokens", 0) or 0)
    except (ValueError, AttributeError, TypeError):
        return
    with _db_lock:
        _stats["prompt_tokens"] += pt
        _stats["completion_tokens"] += ct


def redact_images(body_obj):
    # Never persist raw image bytes. Replace each base64 image data-URI with its content hash: the hash still
    # uniquely identifies the image, so identical image+prompt requests collide on the same cache key -- but
    # the keyed/stored body is tiny instead of hundreds of KB (a 2.9 GB DB of base64 blobs is what corrupted
    # the store under concurrent writes). The ORIGINAL body (real base64) is still forwarded to the upstream
    # model; only the cache key + the stored `request` column use this compact form.
    msgs = body_obj.get("messages") if isinstance(body_obj, dict) else None
    if not isinstance(msgs, list):
        return body_obj
    out, changed = [], False
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            new_c = []
            for part in c:
                url = (part.get("image_url") or {}).get("url", "") if isinstance(part, dict) else ""
                if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                    head, b64 = url.split(";base64,", 1)
                    part = {**part, "image_url": {**part["image_url"],
                            "url": f"{head};base64,sha256:{hashlib.sha256(b64.encode()).hexdigest()}"}}
                    changed = True
                new_c.append(part)
            # Cache-KEY canonicalization: drop explicit-prompt-caching control fields and the "\n"
            # breakpoint carrier, then fold an all-text content array back to a plain string. A
            # request split for provider prompt caching thus keys identically to its unsplit form,
            # so recorded (plain) replay entries still hit when ai_prefix_cache is on.
            cleaned = []
            for part in new_c:
                if isinstance(part, dict):
                    part = {k: v for k, v in part.items() if k != "prompt_cache_breakpoint"}
                    if part.get("type") == "text" and part.get("text") == "\n":
                        changed = True
                        continue  # the image-prefix breakpoint carrier: semantically empty
                cleaned.append(part)
            new_c = cleaned
            if new_c and all(isinstance(x, dict) and x.get("type") == "text" for x in new_c):
                m = {**m, "content": "".join(x.get("text", "") for x in new_c)}
                changed = True
            else:
                m = {**m, "content": new_c}
        out.append(m)
    body_obj = {**body_obj, "messages": out} if changed else body_obj
    if isinstance(body_obj, dict) and "prompt_cache_options" in body_obj:
        body_obj = {k: v for k, v in body_obj.items() if k != "prompt_cache_options"}
    return body_obj


def key_for(body_obj):
    canon = json.dumps(body_obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canon).hexdigest()


def load(k):
    with _db_lock:
        row = _con.execute("SELECT status, output, headers, latency FROM cache WHERE key = ?",
                           [k]).fetchone()
    if not row:
        return None
    return {"status": row[0], "body": row[1],
            "headers": json.loads(row[2]) if row[2] else {}, "latency": row[3]}


def typesafe_cost(output):
    """Cost of a System One response from its usage (input tokens only; output tokens are free)."""
    try:
        return json.loads(output)["usage"]["input_tokens"] / 1e6 * TYPESAFE_PRICE_INPUT
    except (ValueError, KeyError, TypeError):
        return None


def store(k, request_obj, status, output, headers, latency, systemone=False):
    try:
        cost = float(headers["x-litellm-response-cost"]) if headers.get("x-litellm-response-cost") else None
    except (TypeError, ValueError):
        cost = None
    if systemone:
        cost = typesafe_cost(output)
    with _db_lock:
        _con.execute("INSERT INTO cache (key, request, output, status, headers, latency, cost, recorded_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (key) DO NOTHING",
                     [k, json.dumps(request_obj), output, status, json.dumps(headers), latency, cost,
                      datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)])


ALIAS_CHAT = os.environ.get("CACHE_ALIAS_CHAT", "") == "1"


def chat_alias_hit(body_obj):
    """Bench-specific (CACHE_ALIAS_CHAT=1): serve a chat-completions request from the RECORDED
    Responses-API entry with the same prompt text, so two engines with different envelopes
    consume the SAME model sample instead of resampling nondeterministic verdicts. The alias key
    is the Responses body {input: <last user text>, model}; on a hit the stored Responses output
    is converted to chat format (message text + usage), and the entry's recorded latency/cost
    replay as usual. System-prompt content is intentionally ignored for the alias."""
    if not ALIAS_CHAT or not isinstance(body_obj, dict):
        return None
    msgs = body_obj.get("messages")
    if not isinstance(msgs, list):
        return None
    user = next((m.get("content") for m in reversed(msgs)
                 if isinstance(m, dict) and m.get("role") == "user"), None)
    if not isinstance(user, str):
        return None
    alias_key = key_for({"input": user, "model": body_obj.get("model")})
    hit = load(alias_key)
    if hit is None:
        return None
    try:
        resp = json.loads(hit["body"])
        text = next((it.get("content", [{}])[0].get("text", "") for it in resp.get("output", [])
                     if isinstance(it, dict) and it.get("type") == "message"), "")
        u = resp.get("usage", {}) or {}
        chat = {"id": "chatcmpl-alias", "object": "chat.completion", "model": resp.get("model"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": u.get("input_tokens", 0),
                          "completion_tokens": u.get("output_tokens", 0),
                          "total_tokens": u.get("total_tokens", 0)}}
    except Exception:
        return None
    return {"status": 200, "body": json.dumps(chat), "headers": hit.get("headers", {}),
            "latency": hit.get("latency", 0)}


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 keep-alive: the engine pools connections; every response already
    # carries Content-Length, and ThreadingHTTPServer gives each connection its own
    # daemon thread, so persistent connections are safe here.
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass

    def _send(self, status, body_bytes, headers):
        self.send_response(status)
        for k, v in (headers or {}).items():
            if v is not None:
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        try:
            self.wfile.write(body_bytes)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _forward(self, raw):
        upstream = (EMBED_UPSTREAM if self.path.endswith("/embeddings") else
                    TYPESAFE_UPSTREAM if self.path.endswith("/systemone") else UPSTREAM)
        req = urllib.request.Request(upstream + self.path, data=raw, method="POST")
        for h in ("Content-Type", "Authorization"):
            if self.headers.get(h):
                req.add_header(h, self.headers.get(h))
        req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.status, r.read().decode(), {k: v for k, v in r.headers.items()}
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(), {k: v for k, v in (e.headers or {}).items()}
        except Exception as e:  # noqa: BLE001 - relay as a gateway error
            return 502, json.dumps({"error": {"message": f"cache_proxy upstream error: {e}"}}), \
                {"Content-Type": "application/json"}

    def _sleep_until_replay(self, latency):
        # Reproduce the original latency from the CLIENT's perspective: reply at
        # (client_start + latency). This subtracts whatever time already elapsed in transit /
        # queueing / proxy handling, so the client observes exactly `latency`. Without a client
        # start header (X-Request-Start-Ms, ms since epoch), fall back to sleeping the full latency.
        start = self.headers.get("X-Request-Start-Ms")
        if start:
            try:
                delay = (float(start) / 1000.0 + latency) - time.time()
            except ValueError:
                delay = latency
        else:
            delay = latency
        if delay > 0:
            time.sleep(delay)

    def do_GET(self):
        if self.path == "/cache/stats":
            with _db_lock:
                n, tot_cost, tot_lat = _con.execute(
                    "SELECT count(*), coalesce(sum(cost),0), coalesce(sum(latency),0) FROM cache").fetchone()
            body = json.dumps({**_stats, "entries": n, "cached_cost_usd": round(tot_cost, 6),
                               "cached_latency_s": round(tot_lat, 2), "db": CACHE_DB,
                               "simulate_latency": SIMULATE, "upstream": UPSTREAM}).encode()
            self._send(200, body, {"Content-Type": "application/json"})
        else:
            self._send(200, b'{"status":"ok"}', {"Content-Type": "application/json"})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        cacheable = (self.path.endswith("/chat/completions") or self.path.endswith("/embeddings") or
                     self.path.endswith("/responses") or  # OpenAI Responses API (PLOP's client)
                     self.path.endswith("/systemone"))    # TypeSafe System One
        body_obj = None
        if cacheable:
            try:
                body_obj = json.loads(raw or b"{}")
            except ValueError:
                cacheable = False

        # Key + store on the image-redacted body (no base64 bytes persisted); forward the ORIGINAL raw.
        key_obj = redact_images(body_obj) if cacheable else None
        cache_key = key_for(key_obj) if cacheable else None
        if cacheable:
            hit = None
            if self.path.endswith("/chat/completions"):
                hit = chat_alias_hit(body_obj)
            if hit is None:
                hit = load(cache_key)
            if hit is not None:
                if SIMULATE and hit.get("latency", 0) > 0:
                    self._sleep_until_replay(hit["latency"])
                with _db_lock:
                    _stats["hits"] += 1
                if self.path.endswith("/chat/completions"):
                    _accumulate_tokens(hit["body"])
                self._send(hit["status"], hit["body"].encode(), hit.get("headers", {}))
                return

        t0 = time.time()
        status, body, hdrs = self._forward(raw)
        latency = time.time() - t0
        keep = {k: hdrs.get(k) for k in KEEP_HEADERS if hdrs.get(k)}
        if cacheable and 200 <= status < 300:
            store(cache_key, key_obj, status, body, keep, latency, systemone=self.path.endswith("/systemone"))
            with _db_lock:
                _stats["misses"] += 1
            if self.path.endswith("/chat/completions"):
                _accumulate_tokens(body)
        elif cacheable:
            with _db_lock:
                _stats["errors"] += 1  # errors/429 are relayed, never cached (so the client retries)
        self._send(status, body.encode(), keep)


if __name__ == "__main__":
    ThreadingHTTPServer.request_queue_size = 256
    ThreadingHTTPServer.daemon_threads = True
    print(f"cache_proxy on :{PORT} -> {UPSTREAM}  db={CACHE_DB}  simulate_latency={SIMULATE}",
          file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
