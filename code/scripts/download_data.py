#!/usr/bin/env python3
"""Download and authenticate the eight registered Stanford Gset files."""

from __future__ import annotations

import hashlib
import json

from hqml_hierarchical_gctr.experiment import GSET_SOURCES, GSET_URLS, _gset_path


def main() -> None:
    records = []
    for name, expected in GSET_SOURCES.items():
        path = _gset_path(name)
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"digest mismatch for {name}")
        records.append(
            {"name": name, "url": GSET_URLS[name], "sha256": observed, "cache": str(path)}
        )
    print(json.dumps({"status": "authenticated", "files": records}, indent=2))


if __name__ == "__main__":
    main()
