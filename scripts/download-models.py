"""Download and cache the default Memex models.

Stub. Real implementation will:
  - Resolve model IDs from `MemexSettings.models`
  - Download via huggingface-cli with hashes verified
  - Place into `~/.cache/memex/models` (or whatever the config says)
  - Report final disk usage and exit non-zero if anything is missing

See GUIDELINES.md Part III "The model stack and VRAM budget".
"""

from __future__ import annotations

import sys


def main() -> int:
    print("scripts/download-models.py: not yet implemented", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
