#!/usr/bin/env python3
"""The aisql embeddings server. The code lives in the pip package (python/swan_aisql/embed_server.py, also
`swan-aisql-embed` once installed); this entry point runs that module from the checkout so serve/start_stack.sh
and the benchmark harnesses need no installation. Usage and environment: see the module docstring."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from swan_aisql.embed_server import main  # noqa: E402

if __name__ == "__main__":
    main()
