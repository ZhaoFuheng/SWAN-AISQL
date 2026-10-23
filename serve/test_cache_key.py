"""Cache-key canonicalization of the recording proxy: a request the engine split for provider prompt caching
must key exactly like the unsplit body the cache recorded. Run: .venv/bin/python -m pytest serve/test_cache_key.py
(or python serve/test_cache_key.py). The server module opens its database on import, so the functions under
test are loaded from the source text instead."""
import hashlib, json, os, re

SRC = open(os.path.join(os.path.dirname(__file__), "ai_cache_server.py")).read()
NS = {"hashlib": hashlib, "json": json}
for name in ("_hash_data_url", "_merge_text_parts", "_carries_cache_fields", "_canon_user", "canon_for", "responses_to_chat", "chat_to_responses", "redact_responses", "redact_images", "key_for"):
    exec(re.search(r"^def %s\(.*?(?=^def |^[A-Z_]+ *=|^class )" % re.escape(name), SRC, re.S | re.M).group(0), NS)
key = lambda b: NS["key_for"](NS["redact_images"](b))
IMG = "data:image/jpeg;base64," + "QUJD" * 50
BP = {"prompt_cache_breakpoint": {"mode": "explicit"}}


def chat(content, **extra):
    return {"model": "m", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": content}], **extra}


def responses(content, **extra):
    return {"model": "m", "instructions": "s", "input": [{"role": "user", "content": content}], **extra}


def test_chat_text_split_keys_like_plain():
    plain = chat("The image fits the description: Colorbar nail lacquer")
    split = chat([{"type": "text", "text": "The image fits the description: ", **BP},
                  {"type": "text", "text": "Colorbar nail lacquer"}], prompt_cache_options={"mode": "explicit"})
    assert key(split) == key(plain)


def test_chat_image_prefix_with_text_cut_keys_like_plain():
    # the factor graph's span runs through the first side's operands: text | image | text, cut before the title
    plain = chat([{"type": "text", "text": "The image "}, {"type": "image_url", "image_url": {"url": IMG}},
                  {"type": "text", "text": " fits the description: Colorbar nail lacquer"}])
    split = chat([{"type": "text", "text": "The image "}, {"type": "image_url", "image_url": {"url": IMG}},
                  {"type": "text", "text": " fits the description: "}, {"type": "text", "text": "\n", **BP},
                  {"type": "text", "text": "Colorbar nail lacquer"}], prompt_cache_options={"mode": "explicit"})
    assert key(split) == key(plain)


def test_chat_image_ending_prefix_keys_like_plain():
    plain = chat([{"type": "text", "text": "The image "}, {"type": "image_url", "image_url": {"url": IMG}},
                  {"type": "text", "text": " depicts white socks"}])
    split = chat([{"type": "text", "text": "The image "}, {"type": "image_url", "image_url": {"url": IMG}},
                  {"type": "text", "text": "\n", **BP}, {"type": "text", "text": " depicts white socks"}])
    assert key(split) == key(plain)


def test_responses_split_keys_like_plain():
    plain = responses("Judge the report below. Is it specific? " + "lorem " * 10)
    split = responses([{"type": "input_text", "text": "Judge the report below. Is it specific? ", **BP},
                       {"type": "input_text", "text": "lorem " * 10}], prompt_cache_options={"mode": "explicit"})
    assert key(split) == key(plain)


def test_responses_image_with_text_cut_keys_like_plain():
    plain = responses([{"type": "input_text", "text": "The image "}, {"type": "input_image", "image_url": IMG},
                       {"type": "input_text", "text": " fits the description: Colorbar"}])
    split = responses([{"type": "input_text", "text": "The image "}, {"type": "input_image", "image_url": IMG},
                       {"type": "input_text", "text": " fits the description: ", **BP},
                       {"type": "input_text", "text": "Colorbar"}])
    assert key(split) == key(plain)


def test_plain_body_keeps_adjacent_text_parts():
    # a request WITHOUT cache fields (the other systems' bodies) keys on its exact part structure: LOTUS sends
    # [text, text, image, text], and its recordings are keyed on that shape
    lotus = chat([{"type": "text", "text": "Context:\n[name]: «x»"}, {"type": "text", "text": "[Images]: \n"},
                  {"type": "image_url", "image_url": {"url": IMG}}, {"type": "text", "text": "\n\nClaim: y"}])
    red = NS["redact_images"](lotus)
    assert [p.get("type") for p in red["messages"][1]["content"]] == ["text", "text", "image_url", "text"]
    assert key(lotus) != NS["key_for"](NS["redact_images"](lotus, merge=True))


def test_canon_key_is_envelope_independent():
    # the same request through chat and through the Responses API shares one canon key (exact keys differ)
    sch = {"type": "object", "properties": {"result": {"type": "boolean"}}, "required": ["result"]}
    c = {"model": "m", "temperature": 0, "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Is it?"}],
         "response_format": {"type": "json_schema", "json_schema": {"name": "ai_filter", "strict": True, "schema": sch}}}
    r = {"model": "m", "store": False, "temperature": 0, "instructions": "S", "input": [{"role": "user", "content": "Is it?"}],
         "prompt_cache_options": {"mode": "explicit"}, "text": {"format": {"type": "json_schema", "name": "ai_filter", "strict": True, "schema": sch}}}
    assert key(c) != key(r)
    assert NS["canon_for"](NS["redact_images"](c)) == NS["canon_for"](NS["redact_images"](r))
    # with an image and a split: chat's carrier block and both split shapes fold to the same canon
    ci = chat([{"type": "text", "text": "The image "}, {"type": "image_url", "image_url": {"url": IMG}},
               {"type": "text", "text": "\n", **BP}, {"type": "text", "text": " depicts socks"}])
    ri = responses([{"type": "input_text", "text": "The image "}, {"type": "input_image", "image_url": IMG, **BP},
                    {"type": "input_text", "text": " depicts socks"}])
    ci["response_format"] = c["response_format"]; ri["text"] = r["text"]
    assert NS["canon_for"](NS["redact_images"](ci)) == NS["canon_for"](NS["redact_images"](ri))
    # a different system prompt or schema name is another request
    c2 = json.loads(json.dumps(c)); c2["messages"][0]["content"] = "T"
    assert NS["canon_for"](NS["redact_images"](c2)) != NS["canon_for"](NS["redact_images"](c))


def test_envelope_conversions_round_trip():
    resp = json.dumps({"model": "m", "output": [{"type": "reasoning", "summary": []}, {"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": "{\"result\": true}"}]}],
                       "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10, "input_tokens_details": {"cached_tokens": 7}}})
    chat_body = json.loads(NS["responses_to_chat"](resp))
    assert chat_body["choices"][0]["message"]["content"] == "{\"result\": true}" and chat_body["usage"]["prompt_tokens"] == 8
    assert chat_body["usage"]["prompt_tokens_details"]["cached_tokens"] == 7
    back = json.loads(NS["chat_to_responses"](json.dumps(chat_body)))
    assert back["output"][0]["content"][0]["text"] == "{\"result\": true}" and back["usage"]["input_tokens"] == 8


def test_different_text_still_differs():
    assert key(chat("a b")) != key(chat("a c"))
    assert key(chat([{"type": "text", "text": "a "}, {"type": "image_url", "image_url": {"url": IMG}}])) != \
        key(chat([{"type": "text", "text": "b "}, {"type": "image_url", "image_url": {"url": IMG}}]))


def test_image_bytes_never_stored():
    red = NS["redact_images"](chat([{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": IMG}}]))
    assert "QUJD" not in json.dumps(red) and "sha256:" in json.dumps(red)


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
