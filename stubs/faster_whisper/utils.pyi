# Minimal stub for faster_whisper.utils — covers only `download_model`, used by
# `memex.models.download` to pre-cache the ASR CTranslate2 model (maps a size name → its
# CT2 repo and snapshot_downloads it into the HF cache). faster-whisper ships no `py.typed`.
def download_model(
    size_or_id: str,
    *,
    output_dir: str | None = ...,
    local_files_only: bool = ...,
    cache_dir: str | None = ...,
    **kwargs: object,
) -> str: ...
