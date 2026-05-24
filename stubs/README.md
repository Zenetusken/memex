# Local type stubs

Minimal `.pyi` stubs for third-party libraries that ship no usable type
information (no `py.typed`, or incomplete). Each stub covers **only the
surface `src/memex` actually uses** — they are not complete library
stubs. Wired into pyright via `stubPath = "stubs"` in `pyproject.toml`.

Adding to a stub: type the specific symbol/method Memex calls. Prefer
precise types; fall back to explicit `Any` only at genuinely-dynamic
boundaries (explicit `Any` is fine under strict — it's *inferred*
Unknown that `reportUnknown*` flags).
