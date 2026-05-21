# vLLM daemon under launchd (macOS)

Memex's production target is Linux + NVIDIA CUDA per ADR-0001; macOS is a development tier where the CUDA-only stack doesn't apply. This guide ships a `launchd` user-agent template for running the serve script as a background process on a Mac dev box — useful if you're iterating on the agent / web UI side without the GPU pipeline.

If you actually need GPU inference on macOS, swap the model out of `MEMEX_VLLM_MODEL` for something Metal/MPS-friendly and drop `MEMEX_VLLM_QUANTIZATION` entirely — the default `Qwen/Qwen3-8B-AWQ` with the `awq_marlin` kernel is CUDA-only.

## Quickstart

1. **Copy the template**

   ```sh
   mkdir -p ~/Library/LaunchAgents ~/.local/state/memex
   cp docs/deploy/com.memex.vllm.plist ~/Library/LaunchAgents/
   ```

2. **Edit the paths** in `~/Library/LaunchAgents/com.memex.vllm.plist`. Every `/Users/YOUR_USER/...` placeholder needs the actual absolute path — launchd does not expand shell vars or `~`. Set:
   - `WorkingDirectory` to your Memex clone root
   - The first item of `ProgramArguments` to the absolute path of `scripts/serve-vllm.sh`
   - The `PATH` env to include where `uv` lives (`which uv` to check)
   - `StandardOutPath` / `StandardErrorPath` to wherever you want the logs

3. **Tune `EnvironmentVariables`** in the same plist to override model + binding knobs. The defaults shown match `scripts/serve-vllm.sh`; comment out any key to fall back to the script's built-in default.

4. **Load**

   ```sh
   launchctl load -w ~/Library/LaunchAgents/com.memex.vllm.plist
   ```

   The `-w` flag flips the `Disabled` bit so the agent persists across reboots.

5. **Verify**

   ```sh
   launchctl list | grep memex
   tail -f ~/.local/state/memex/vllm.err.log
   curl http://127.0.0.1:8000/v1/models
   ```

## Operations

| Action | Command |
|---|---|
| Tail logs | `tail -f ~/.local/state/memex/vllm.{out,err}.log` |
| Restart | `launchctl kickstart -k gui/$UID/com.memex.vllm` |
| Stop (this session only) | `launchctl stop com.memex.vllm` |
| Unload (and disable boot start) | `launchctl unload -w ~/Library/LaunchAgents/com.memex.vllm.plist` |
| Reload env changes | `launchctl unload ~/Library/LaunchAgents/com.memex.vllm.plist && launchctl load ~/Library/LaunchAgents/com.memex.vllm.plist` |

## What the plist does

- **`KeepAlive.SuccessfulExit=false`** — respawn on crash, not on clean exit. Equivalent to systemd's `Restart=on-failure`.
- **`ThrottleInterval=5`** — minimum 5 s between respawns. Prevents a chronic-failure thrash.
- **`RunAtLoad=true`** — start as soon as the user logs in.
- **`StandardOutPath` / `StandardErrorPath`** — separate files; launchd has no journald-equivalent, so file-based logs + a cron-driven `logrotate` (or just periodic truncation) is the operating model.
- **No `Type=` analog** — launchd doesn't distinguish `simple` / `forking`; the process exec'd by `ProgramArguments` IS the service. Make sure the script does not double-fork or detach.

The full template lives at [`com.memex.vllm.plist`](com.memex.vllm.plist).

## What's not covered here

- **GPU monitoring** — macOS dev boxes are out of scope for the CUDA pipeline; if you're testing on an Apple Silicon Mac with the MPS backend, see PyTorch's own MPS docs for tuning.
- **The web UI / MCP server** — copy the plist, change the `Label`, the script path, and the env vars, and you've got separate user agents for `memex serve web` / `memex serve mcp`.
- **System-wide daemons** (`/Library/LaunchDaemons/`) — single-user / single-Mac is the assumed setup; LaunchDaemon needs root, persists across logout, and isn't worth the operational weight for a dev box.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `launchctl load` succeeds but `launchctl list` doesn't show it | Plist parse error or wrong `Label` — `plutil -lint ~/Library/LaunchAgents/com.memex.vllm.plist` will report syntax issues. |
| Logs are empty | The process is failing before writing anything. Try `launchctl debug gui/$UID/com.memex.vllm --stdout=/tmp/m.out --stderr=/tmp/m.err && launchctl kickstart -k gui/$UID/com.memex.vllm`. |
| `uv: command not found` | Adjust the `PATH` value inside `EnvironmentVariables`. launchd does *not* inherit your shell's `PATH`. |
| Service keeps respawning | Look at the err log — `MEMEX_VLLM_MODEL` defaulting to a CUDA-only quant on a Mac is the usual culprit. Set it to an MPS-compatible model or undefine `MEMEX_VLLM_QUANTIZATION`. |
| `Operation not permitted` | macOS Gatekeeper / TCC denied the agent disk access. Approve under System Settings → Privacy & Security → Full Disk Access (add Terminal / your shell). |
