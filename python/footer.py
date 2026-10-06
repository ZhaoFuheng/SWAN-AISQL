"""Read, and optionally re-stamp, the DuckDB build recorded in an extension binary's metadata footer.

    python footer.py aisql.duckdb_extension                      -> "<platform> <duckdb build> <extension version>"
    python footer.py aisql.duckdb_extension --stamp v2.0.0-alpha43763

DuckDB appends 512 bytes to every extension: eight 32-byte fields stored last-first (magic "4", platform, the
DuckDB build it was compiled for, the extension version, the ABI, three spare), then a 256-byte signature. The
loader compares the build field with its own version directory name: the release tag for a release (an alpha
counts as one), the 10-character source commit for a -dev build. A binary built by the CI pipeline from a
commit carries that commit's full hash, truncated to the field, which matches neither; --stamp writes the
engine's own string in its place. It is only ever used for a binary built from exactly that commit.
"""

import sys

FOOTER = 512
FIELD = 32


def _fields(footer: bytes):
    fields = [footer[i * FIELD : (i + 1) * FIELD] for i in range(8)]
    fields.reverse()  # magic, platform, duckdb build, extension version, abi, ...
    return fields


def read_footer(path: str):
    with open(path, "rb") as f:
        f.seek(-FOOTER, 2)
        fields = [b.rstrip(b"\0").decode() for b in _fields(f.read(FOOTER))]
    if fields[0] != "4":
        sys.exit(f"{path}: not a DuckDB extension (magic {fields[0]!r})")
    return fields[1], fields[2], fields[3]


def stamp(path: str, duckdb_build: str) -> None:
    value = duckdb_build.encode()
    if len(value) > FIELD:
        sys.exit(f"'{duckdb_build}' does not fit the {FIELD}-byte footer field")
    with open(path, "r+b") as f:
        f.seek(-FOOTER, 2)
        start = f.tell()
        footer = bytearray(f.read(FOOTER))
        # field index 2 (duckdb build) sits at reversed slot 8 - 1 - 2 = 5
        off = (7 - 2) * FIELD
        footer[off : off + FIELD] = value.ljust(FIELD, b"\0")
        f.seek(start)
        f.write(footer)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[2] == "--stamp":
        stamp(sys.argv[1], sys.argv[3])
    print(*read_footer(sys.argv[1]))
