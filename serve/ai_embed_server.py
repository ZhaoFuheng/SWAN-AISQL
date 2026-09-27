#!/usr/bin/env python3
"""aisql embeddings server: a text encoder for text predicates plus a CLIP dual encoder for images.

OpenAI-compatible POST /v1/embeddings. `input` may be a string, a list of strings, or a list mixing
  "plain text"                     -> the text model (default all-MiniLM-L6-v2, 384-d, zero-padded to
                                      the image model's width so every feature has one size)
  {"image": "<path-or-data-uri>"}  -> the CLIP image tower (512-d)
  {"image_text": "predicate text"} -> the CLIP TEXT tower: the predicate half of an image leaf's
                                      feature, so its cosine with the image is in one joint space

Why two models. The selectivity MLP's feature is `embed(predicate) | embed(input) | cosine`. For text
leaves the input is a review, a comment, a description: MiniLM (256-token window, trained for sentence
similarity) separates rows; CLIP's text tower (77 tokens, contrastive) does not -- on agent_bench Q17
it made the learned order unstable (237 calls stable with MiniLM vs 230 / 1,461 run-to-run with CLIP)
and cost 20 s more per query. For image leaves only a dual encoder can relate the predicate text to
the pixels, and that comparison must happen in CLIP's space -- hence the marked "image_text" item.

Backends:
  sentence-transformers (default; every platform)  AI_EMBED_BACKEND=st
  mlx-embeddings (Apple Silicon only; pip install mlx-embeddings pillow)  AI_EMBED_BACKEND=mlx
      one CLIP model for everything (mlx-community/clip-vit-base-patch32); opt-in and untested
      against the recorded runs.

Env:
  AI_EMBED_PORT         (default 4002)
  AI_EMBED_BACKEND      st | mlx (default st)
  AI_EMBED_MODEL        text model (default sentence-transformers/all-MiniLM-L6-v2)
  AI_EMBED_IMAGE_MODEL  CLIP model for images + image predicates (default clip-ViT-B-32; "" = none,
                        image items then come back as error entries and the engine keeps its
                        neutral prior for image leaves)
  AI_EMBED_DEVICE       torch device (default cpu). torch auto-picks a GPU (mps/cuda) when one
                        exists, but for these models the CPU is fast enough per request AND a
                        discrete GPU powers down after ~10 s idle and costs ~0.8 s to wake on the
                        next request -- the selectivity features are requested in small bursts, so
                        that wake-up landed on the critical path of every wave.
"""
import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("AI_EMBED_PORT", "4002"))
BACKEND = os.environ.get("AI_EMBED_BACKEND", "st")
DEVICE = os.environ.get("AI_EMBED_DEVICE", "cpu")
TEXT_MODEL = os.environ.get("AI_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL = os.environ.get("AI_EMBED_IMAGE_MODEL", "clip-ViT-B-32")
MLX_MODEL = os.environ.get("AI_EMBED_MODEL", "mlx-community/clip-vit-base-patch32")
CLIP_MAX_TOKENS = 77  # CLIP's text tower has 77 position embeddings; longer inputs raise

backend = None
backend_desc = {"backend": "none"}  # what GET / reports: the models that actually loaded


def _open_image(ref):
    from PIL import Image
    if ref.startswith("data:"):
        img = Image.open(io.BytesIO(base64.b64decode(ref.split(",", 1)[1])))
    else:
        img = Image.open(ref)
    return img.convert("RGB")


def _round(vectors):
    # 6 decimals is far below the MLP's resolution and halves the JSON payload; rounding must happen
    # in float64 or tolist() prints the float32 values at full double length.
    return vectors.astype("float64").round(6).tolist()


class MlxBackend:
    """One CLIP model for text, image predicates and images (Apple Silicon)."""
    images = True

    def __init__(self, model_name):
        from mlx_embeddings.utils import load
        self.model, self.processor = load(model_name)

    def _texts(self, texts):
        import mlx.core as mx
        inputs = self.processor(text=texts, return_tensors="np", padding=True, truncation=True)
        out = self.model(**{k: mx.array(v) for k, v in inputs.items()})
        emb = out.text_embeds if hasattr(out, "text_embeds") else out.embeddings
        return [[round(float(x), 6) for x in e] for e in emb.tolist()]

    embed_texts = _texts
    embed_image_texts = _texts

    def embed_images(self, refs):
        import mlx.core as mx
        out = []
        for ref in refs:
            inputs = self.processor(images=_open_image(ref), return_tensors="np")
            res = self.model(pixel_values=mx.array(inputs["pixel_values"]))
            emb = res.image_embeds if hasattr(res, "image_embeds") else res.embeddings
            out.append([round(float(x), 6) for x in emb.tolist()[0]])
        return out


class StBackend:
    """sentence-transformers: a text model for text items, a CLIP model for image + image_text items."""

    def __init__(self, text_model, image_model, device):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self.np = np
        self.text = SentenceTransformer(text_model, device=device)
        self.text_dim = self.text.get_sentence_embedding_dimension()
        self.clip = None
        self.images = False
        self.width = self.text_dim
        if image_model:
            self.clip = SentenceTransformer(image_model, device=device)
            first = self.clip[0]
            if type(first).__name__ != "CLIPModel":
                raise RuntimeError(f"AI_EMBED_IMAGE_MODEL={image_model} is not a CLIP model")
            # sentence-transformers' CLIPModel.tokenize never truncates; a predicate longer than 77
            # tokens would raise inside the text tower. Wrap the tokenizer so every text call
            # truncates to the tower's window.
            tok = first.processor.tokenizer

            class TruncatingTokenizer:
                def __call__(self, texts, **kw):
                    kw.setdefault("truncation", True)
                    kw.setdefault("max_length", CLIP_MAX_TOKENS)
                    return tok(texts, **kw)

                def __getattr__(self, name):
                    return getattr(tok, name)

            first.processor.tokenizer = TruncatingTokenizer()
            self.images = True
            self.clip_dim = len(self.clip.encode(["probe"])[0])
            self.width = max(self.text_dim, self.clip_dim)

    def _pad(self, vectors):
        # Every feature must have one width (the MLP fixes its input size on the first example):
        # zero-pad the narrower model's vectors. Padding leaves cosines unchanged.
        if vectors.shape[1] < self.width:
            vectors = self.np.pad(vectors, ((0, 0), (0, self.width - vectors.shape[1])))
        return vectors

    def embed_texts(self, texts):
        return _round(self._pad(self.text.encode(texts)))

    def embed_image_texts(self, texts):
        if not self.images:
            raise RuntimeError("image_text items need AI_EMBED_IMAGE_MODEL (a CLIP model)")
        return _round(self._pad(self.clip.encode(texts)))

    def embed_images(self, refs):
        if not self.images:
            raise RuntimeError("image items need AI_EMBED_IMAGE_MODEL (a CLIP model)")
        return _round(self._pad(self.clip.encode([_open_image(r) for r in refs])))


def load_backend():
    global backend
    if BACKEND == "mlx":
        backend = MlxBackend(MLX_MODEL)
        backend_desc.update(backend="mlx-embeddings", model=MLX_MODEL, images=True)
        print(f"[embed] mlx-embeddings backend, model={MLX_MODEL} (text+image)", flush=True)
        return
    backend = StBackend(TEXT_MODEL, IMAGE_MODEL, DEVICE)
    backend_desc.update(backend="sentence-transformers", model=TEXT_MODEL, image_model=IMAGE_MODEL or None,
                        device=DEVICE, images=backend.images, dim=backend.width)
    print(f"[embed] sentence-transformers backend: text={TEXT_MODEL}"
          f"{', images=' + IMAGE_MODEL if backend.images else ' (text only)'} device={DEVICE} dim={backend.width}",
          flush=True)


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
        self._send({"status": "ok", **backend_desc})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        req = json.loads(raw or b"{}")
        inp = req.get("input", [])
        if isinstance(inp, str):
            inp = [inp]
        groups = {"text": ([], []), "image_text": ([], []), "image": ([], [])}  # kind -> (indexes, items)
        for i, item in enumerate(inp):
            if isinstance(item, dict) and "image" in item:
                kind, value = "image", item["image"]
            elif isinstance(item, dict) and "image_text" in item:
                kind, value = "image_text", item["image_text"]
            else:
                kind, value = "text", item if isinstance(item, str) else json.dumps(item)
            groups[kind][0].append(i)
            groups[kind][1].append(value)
        data = []
        for kind, (idxs, items) in groups.items():
            if not items:
                continue
            fn = {"text": backend.embed_texts, "image_text": backend.embed_image_texts,
                  "image": backend.embed_images}[kind]
            if kind == "image":
                # one failed image (missing file, undecodable) must not sink the whole batch
                for i, ref in zip(idxs, items):
                    try:
                        data.append({"object": "embedding", "index": i, "embedding": fn([ref])[0]})
                    except Exception as ex:  # noqa: BLE001
                        data.append({"object": "error", "index": i, "error": str(ex)[:200]})
                continue
            try:
                for i, emb in zip(idxs, fn(items)):
                    data.append({"object": "embedding", "index": i, "embedding": emb})
            except Exception as ex:  # noqa: BLE001
                for i in idxs:
                    data.append({"object": "error", "index": i, "error": str(ex)[:200]})
        data.sort(key=lambda d: d["index"])
        self._send({"object": "list", "data": data, "usage": {"total_tokens": len(inp)}})


if __name__ == "__main__":
    load_backend()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    ThreadingHTTPServer.daemon_threads = True
    print(f"[embed] listening on :{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
