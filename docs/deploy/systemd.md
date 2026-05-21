# vLLM daemon under systemd (Linux)

Memex's vLLM orchestrator runs fine in a foreground terminal (`scripts/serve-vllm.sh`) or via the built-in `memex daemon start`, but for production deployments — long-running boxes, automatic restart on crash, log rotation, structured journald audit trail — handing the supervisor role over to systemd is the right call.

## Why systemd, not `memex daemon`

`memex daemon start` is a CLI affordance: it `Popen`'s the serve script with `start_new_session=True`, writes a PID file at `vault/.memex/daemon/vllm.pid`, polls for endpoint reachability, and exits. It's useful interactively. Under systemd you bypass it entirely — systemd becomes the supervisor:

- **Restart on crash** — systemd watches the process; on `vllm serve` exit ≠ 0 (segfault, OOM, driver hiccup) the unit restarts automatically, throttled to 5 restarts per minute.
- **Log rotation** — `StandardOutput=journal` pipes stdout + stderr to journald. `journalctl --vacuum-time=30d` does the rotation.
- **Boot ordering** — `After=network-online.target` keeps the unit out of the early-boot race.
- **Lifecycle integration** — `systemctl --user stop memex-vllm` cleanly SIGTERMs the process group; vLLM does its own graceful shutdown; `TimeoutStopSec=30` escalates to SIGKILL if anything hangs.

`memex daemon status` still works against a systemd-managed vLLM (it just hits the OpenAI-compatible endpoint), but **don't run `memex daemon stop`** — it'll race the systemd restart loop. Use `systemctl --user stop`.

## Quickstart (user unit)

The user-unit path is the recommended setup: no root required, runs under your account, fits the local-first single-user posture.

1. **Install the unit + env file**

   ```sh
   mkdir -p ~/.config/systemd/user ~/.config/memex
   cp docs/deploy/memex-vllm.service ~/.config/systemd/user/
   cp docs/deploy/memex-vllm.env     ~/.config/memex/memex-vllm.env
   ```

2. **Edit the paths** in `~/.config/systemd/user/memex-vllm.service` so `WorkingDirectory=` and `ExecStart=` point at your Memex clone:

   ```ini
   WorkingDirectory=%h/project/Doc_Flo
   ExecStart=%h/project/Doc_Flo/scripts/serve-vllm.sh
   ```

   `%h` expands to your `$HOME`. If you cloned somewhere else, hard-code the absolute path.

3. **Configure the model + binding** by uncommenting + editing lines in `~/.config/memex/memex-vllm.env`. Every line is the documented default from `scripts/serve-vllm.sh`; uncomment only the ones you want to override.

4. **Reload + enable + start**

   ```sh
   systemctl --user daemon-reload
   systemctl --user enable --now memex-vllm.service
   ```

   `--now` starts it immediately; `enable` makes it boot with your user session.

5. **Verify**

   ```sh
   systemctl --user status memex-vllm
   journalctl --user -u memex-vllm -f
   # First boot takes ~40s for vLLM weights + CUDA-graph capture.
   # Once you see "Application startup complete." it's reachable:
   curl http://127.0.0.1:8000/v1/models
   ```

## Operations

| Action | Command |
|---|---|
| Tail logs | `journalctl --user -u memex-vllm -f` |
| Restart | `systemctl --user restart memex-vllm` |
| Stop | `systemctl --user stop memex-vllm` |
| Disable boot start | `systemctl --user disable memex-vllm` |
| Reload env file changes | `systemctl --user daemon-reload && systemctl --user restart memex-vllm` |
| Pin journal retention | `journalctl --vacuum-time=30d` (or set `SystemMaxUse=` in `journald.conf`) |

## What the unit does

- **`Type=simple`** — systemd considers the service running as soon as the ExecStart process forks; vLLM is healthy when the OpenAI-compatible endpoint becomes reachable (~40 s on the reference RTX 4070).
- **`Restart=on-failure`** — only crashes trigger a restart. A clean `systemctl stop` doesn't loop.
- **`StartLimitBurst=5` / `StartLimitIntervalSec=60`** — five restarts per minute is the ceiling; beyond that systemd gives up and the unit enters `failed`. Check `journalctl` to see why.
- **`TimeoutStartSec=300`** — five-minute window for first-time weight downloads or slow disks. Subsequent boots are bound by `MEMEX_VLLM_GPU_FRACTION` + CUDA-graph capture, both <40 s.
- **`KillMode=mixed`** — SIGTERM to the main process for graceful shutdown; SIGKILL to any stragglers if they exceed `TimeoutStopSec=30`.
- **`Environment=PATH=%h/.local/bin:/usr/local/bin:/usr/bin:/bin`** — ensures `uv` (typically at `~/.local/bin/uv`) is findable inside the systemd-stripped environment. Adjust if your `uv` lives elsewhere.

The complete unit lives at [`memex-vllm.service`](memex-vllm.service); the env file at [`memex-vllm.env`](memex-vllm.env). Read both — they're commented heavily.

## System unit (multi-user / shared rig)

If the box is shared and you want vLLM running under a dedicated `memex` system user:

1. Copy the unit to `/etc/systemd/system/memex-vllm.service` (requires root).
2. Add a `User=memex` line under `[Service]`.
3. Replace every `%h` with the absolute path to the memex user's home (`/home/memex` or wherever).
4. `sudo systemctl daemon-reload && sudo systemctl enable --now memex-vllm`.

The trade-offs: needs root for install, but boots before any user logs in. For most single-user laptops/desktops, the user unit is enough.

## The full stack — web + MCP as sibling units

Beyond the vLLM daemon you can run two more services under the same supervisor: the FastAPI/HTMX web UI (`memex serve web`) and the MCP server (`memex serve mcp --transport http`). Both have sibling templates in this directory:

| Unit | Template | Env file | Default bind | Notes |
|---|---|---|---|---|
| `memex-vllm.service` | [`memex-vllm.service`](memex-vllm.service) | [`memex-vllm.env`](memex-vllm.env) | `127.0.0.1:8000` | The OpenAI-compatible inference endpoint |
| `memex-web.service` | [`memex-web.service`](memex-web.service) | [`memex-web.env`](memex-web.env) | `127.0.0.1:7423` | Browser UI (single-user / loopback only) |
| `memex-mcp.service` | [`memex-mcp.service`](memex-mcp.service) | [`memex-mcp.env`](memex-mcp.env) | `127.0.0.1:7424` | MCP HTTP transport; requires a token to bind non-loopback |

Install all three at once:

```sh
# From the repo root
mkdir -p ~/.config/systemd/user ~/.config/memex ~/.local/state/memex
cp docs/deploy/memex-vllm.service docs/deploy/memex-web.service docs/deploy/memex-mcp.service \
   ~/.config/systemd/user/
cp docs/deploy/memex-vllm.env docs/deploy/memex-web.env docs/deploy/memex-mcp.env \
   ~/.config/memex/

# If you want MCP HTTP, generate a token + drop it into the env file:
echo "MEMEX_MCP__AUTH_TOKEN=$(uv run memex mcp generate-token)" \
  >> ~/.config/memex/memex-mcp.env

# Edit each unit so WorkingDirectory + ExecStart point at your clone,
# then:
systemctl --user daemon-reload
systemctl --user enable --now memex-vllm.service memex-web.service memex-mcp.service
```

### Dependency ordering

The web + MCP units both carry `After=memex-vllm.service` + `Wants=memex-vllm.service`. systemd boots vLLM first, then web + MCP in parallel; on `systemctl --user start memex-web`, vLLM is implicitly started too.

The dependency is intentionally *soft*. Web + MCP boot fine even if vLLM is down — the document browser, graph view, and search/get_document/list_documents tools all work without an LLM. Only `/ask` queries (web) and the `ask` tool (MCP) fail when vLLM is unreachable. That's the right trade-off: a single CUDA OOM on the inference side shouldn't take down the entire stack.

`Type=simple` means systemd considers vLLM "started" the instant the process forks, not when the endpoint is actually reachable. In practice the 20-second cold-boot window is short enough that downstream callers see a clean "Connection refused" → retry rather than a hung tool call. If you want hard ordering with reachability checks, a `Type=notify` integration is the proper fix (out of scope for these templates).

### Watching the full stack

```sh
# Tail all three at once
journalctl --user -u memex-vllm -u memex-web -u memex-mcp -f

# Status overview
systemctl --user status memex-vllm memex-web memex-mcp --no-pager
```

## What's not covered here

- **GPU monitoring** — out of scope. `nvidia-smi` works fine; if you want metrics in journald, `nvidia-smi --query-gpu=… --loop=5` in a sidecar unit is one path.
- **Backup of the vault** — the vault is regular files under `~/.memex/vault`; any incremental backup tool (`restic`, `borg`, `rsnapshot`) handles it. A `memex-vault-backup.timer` is a natural follow-up to these unit templates.
- **A `memex upgrade` wrapper** — `git pull && uv sync && systemctl --user restart memex-vllm memex-web memex-mcp` is the manual recipe; bundling it into one CLI command is a follow-up.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Failed to start memex-vllm.service` with `status=203/EXEC` | `ExecStart` path wrong — check `systemctl --user cat memex-vllm` and edit the unit to match your clone. |
| Boots but `curl http://127.0.0.1:8000/v1/models` connection-refused | Look at `journalctl`: probably a CUDA OOM, a download in progress, or a `MEMEX_VLLM_*` typo. `TimeoutStartSec=300` covers ~5 min — extend if you have a slow link. |
| Unit immediately respawns | Check `MEMEX_VLLM_GPU_FRACTION` — if vLLM can't reserve enough KV cache it exits non-zero, which `Restart=on-failure` catches; you'll see the same OOM message every 5 s in the journal. |
| `uv: command not found` in the log | Adjust the `Environment=PATH=…` line to include the directory `which uv` returns. |
| `Permission denied` on the script | `chmod +x scripts/serve-vllm.sh`. |
