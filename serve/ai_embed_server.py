#!/usr/bin/env python3
"""aisql embeddings server: ONE dual-encoder model for text AND images (mlx-embeddings).

OpenAI-compatible POST /v1/embeddings. `input` may be a string, a list of strings, or a list
mixing strings and {"image": "<path-or-base64-data-uri>"} objects. Text and images land in the
SAME joint space (CLIP/SigLIP-class dual encoder), so a predicate's text can be compared against
a row's image - the zero-shot selectivity prior the reorder's MLP consumes.

Backends (auto-selected):
  mlx-embeddings (Apple Silicon; pip install mlx-embeddings pillow) - text + image
  sentence-transformers fallback - text only (image requests return an error entry)

Env:
  AI_EMBED_PORT   (default 4002)
  AI_EMBED_MODEL  (default mlx-community/clip-vit-base-patch32)
"""
import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("AI_EMBED_PORT", "4002"))
MODEL = os.environ.get("AI_EMBED_MODEL", "mlx-community/clip-vit-base-patch32")

backend = None


class MlxBackend:
    def __init__(self, model_name):
        from mlx_embeddings.utils import load
        from PIL import Image
        self.Image = Image
        self.model, self.processor = load(model_name)

    def embed_texts(self, texts):
        import mlx.core as mx
        inputs = self.processor(text=texts, return_tensors="np", padding=True, truncation=True)
        out = self.model(**{k: mx.array(v) for k, v in inputs.items()})
        emb = out.text_embeds if hasattr(out, "text_embeds") else out.embeddings
        return [list(map(float, e)) for e in emb.tolist()]

    def embed_image(self, ref):
        import mlx.core as mx
        if ref.startswith("data:"):
            img = self.Image.open(io.BytesIO(base64.b64decode(ref.split(",", 1)[1])))
        else:
            img = self.Image.open(ref)
        inputs = self.processor(images=img.convert("RGB"), return_tensors="np")
        out = self.model(pixel_values=mx.array(inputs["pixel_values"]))
        emb = out.image_embeds if hasattr(out, "image_embeds") else out.embeddings
        return list(map(float, emb.tolist()[0]))


class StBackend:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        return [list(map(float, e)) for e in self.model.encode(texts)]

    def embed_image(self, ref):
        raise RuntimeError("image embedding requires the mlx-embeddings backend")


def load_backend():
    global backend
    try:
        backend = MlxBackend(MODEL)
        print(f"[embed] mlx-embeddings backend, model={MODEL} (text+image)")
    except Exception as ex:  # noqa: BLE001
        fallback = os.environ.get("AI_EMBED_TEXT_FALLBACK", "sentence-transformers/all-MiniLM-L6-v2")
        backend = StBackend(fallback)
        print(f"[embed] mlx unavailable ({ex}); sentence-transformers fallback, model={fallback} (text only)")


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 keep-alive: the engine pools connections; every response already
    # carries Content-Length, and ThreadingHTTPServer gives each connection its own
    # daemon thread, so persistent connections are safe here.
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):
        pass

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"status": "ok", "model": MODEL})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(raw or b"{}")
        inp = req.get("input", [])
        if isinstance(inp, str):
            inp = [inp]
        data = []
        text_idx, texts = [], []
        for i, item in enumerate(inp):
            if isinstance(item, dict) and "image" in item:
                try:
                    data.append({"object": "embedding", "index": i, "embedding": backend.embed_image(item["image"])})
                except Exception as ex:  # noqa: BLE001
                    data.append({"object": "error", "index": i, "error": str(ex)[:200]})
            else:
                text_idx.append(i)
                texts.append(item if isinstance(item, str) else json.dumps(item))
        if texts:
            for i, emb in zip(text_idx, backend.embed_texts(texts)):
                data.append({"object": "embedding", "index": i, "embedding": emb})
        data.sort(key=lambda d: d["index"])
        self._send({"object": "list", "data": data, "usage": {"total_tokens": len(inp)}})


if __name__ == "__main__":
    load_backend()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    ThreadingHTTPServer.daemon_threads = True
    print(f"[embed] listening on :{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
