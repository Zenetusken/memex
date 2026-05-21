# Vault backup (Linux + systemd, restic-driven)

The vault is regular files under `~/.memex/vault/` — `documents/` is the source of truth (ADR-0003), `.memex/` holds the indexes + manifests + transient state. Any incremental backup tool handles it. This guide ships a turn-key recipe: a `.service` + `.timer` pair driving `scripts/memex-vault-backup.sh`, which wraps **restic** (deduplicating, encrypted, single-binary; supports local + S3 + Backblaze + SSH targets).

After this is installed, your vault is one `restic restore` away from any incident — accidental delete, re-index gone wrong, disk failure, a botched edit you'd like to roll back.

## What gets backed up

By default the script snapshots the entire vault except a few transient directories:

| Path | Backed up? | Why |
|---|---|---|
| `vault/documents/` | ✅ | The canonical Markdown — the source of truth. |
| `vault/.memex/manifests/` | ✅ | Per-doc audit JSON. Small, useful for restore-time consistency checks. |
| `vault/.memex/embeddings.lance/` | ✅ | LanceDB derived state. Regenerable, but including it makes restore hot — no re-embedding needed. |
| `vault/.memex/search.sqlite` | ✅ | FTS5 derived state. Same reasoning. |
| `vault/.memex/graph.ryu` | ✅ | RyuGraph derived state. Same reasoning. |
| `vault/.memex/locks/` | ❌ | fcntl advisory lock files (P1.5). Per-process transient state. |
| `vault/.memex/daemon/` | ❌ | PID + per-session vllm.log. Ignored under systemd; meaningless across restores. |
| `vault/.memex/events.sqlite` | ❌ | Audit bus — already self-prunes at 30 days; regrowing rolling log. |

restic's content-defined chunking deduplicates derived state efficiently across snapshots, so the cost of including the indexes is small (a few MB per nightly snapshot on a typical vault).

## Quickstart

```sh
# 1. Install restic (one-time, distro-dependent)
sudo apt install restic       # Debian / Ubuntu / Pop!_OS
# brew install restic         # macOS
# sudo pacman -S restic       # Arch
# sudo dnf install restic     # Fedora

# 2. Generate an encryption password file (one-time)
#    Pick a strong passphrase and store it somewhere SAFE off this box —
#    if you lose it AND the local repo, snapshots are unrecoverable.
mkdir -p ~/.config/memex
printf 'your-passphrase-of-choice' > ~/.config/memex/restic-password
chmod 600 ~/.config/memex/restic-password

# 3. Install the templates
mkdir -p ~/.config/systemd/user ~/.config/memex
cp docs/deploy/memex-vault-backup.service \
   docs/deploy/memex-vault-backup.timer \
   ~/.config/systemd/user/
cp docs/deploy/memex-vault-backup.env ~/.config/memex/

# 4. Edit ~/.config/systemd/user/memex-vault-backup.service so
#    WorkingDirectory= and ExecStart= match your clone (default
#    assumes %h/project/Doc_Flo).

# 5. Enable + start the timer (also fires once immediately)
systemctl --user daemon-reload
systemctl --user enable --now memex-vault-backup.timer
```

That's it — a daily backup will run at 02:00 local time. The first run initializes the restic repo at `~/.local/state/memex/backups/`.

## Operations

| Action | Command |
|---|---|
| Inspect the schedule | `systemctl --user list-timers memex-vault-backup` |
| Fire a backup right now (outside the schedule) | `systemctl --user start memex-vault-backup.service` |
| Tail the backup log | `journalctl --user -u memex-vault-backup -f` |
| List all snapshots in the repo | `restic snapshots` (after `export RESTIC_REPOSITORY=… RESTIC_PASSWORD_FILE=…`) |
| Repo stats (deduplicated + raw size) | `restic stats --mode raw-data` |
| Verify repo integrity | `restic check` |
| Stop the daily backups | `systemctl --user disable --now memex-vault-backup.timer` |

The `[Service]` block runs at `Nice=10` + `IOSchedulingClass=best-effort` so backups don't compete with the vLLM daemon or interactive work.

## Configuring a remote target

Edit `~/.config/memex/memex-vault-backup.env` and uncomment one of the `RESTIC_REPOSITORY=` blocks (Backblaze, S3, SSH). The env file documents the credentials each backend needs. After editing:

```sh
systemctl --user daemon-reload
systemctl --user restart memex-vault-backup.timer
```

Restic supports many target types out of the box (`restic --help` lists them all): local, SFTP, REST, S3, Swift, Azure, GCS, B2, Rclone (for OneDrive, Dropbox, Google Drive, etc).

The encryption key (`~/.config/memex/restic-password`) is the **only** thing that decrypts your remote repo. Back it up separately — to a password manager, a piece of paper in a safe, anything off this machine. If you lose both the key *and* this machine, the cloud snapshots are unrecoverable.

## Restoring a snapshot

The whole-vault snapshot is hot-restorable: copy it back, restart the services, you're done.

```sh
# 1. Find the snapshot you want
export RESTIC_REPOSITORY=~/.local/state/memex/backups
export RESTIC_PASSWORD_FILE=~/.config/memex/restic-password
restic snapshots

# 2. Restore (to a tmp dir to avoid clobbering anything live)
restic restore latest --target /tmp/memex-restore
# OR by specific snapshot id:
# restic restore abc12345 --target /tmp/memex-restore

# 3. Stop the live services so they don't write during the swap
systemctl --user stop memex-watch memex-mcp memex-web memex-vllm

# 4. Swap in the restored vault
mv ~/.memex/vault ~/.memex/vault.broken
mv "/tmp/memex-restore$HOME/.memex/vault" ~/.memex/

# 5. Restart the stack
systemctl --user start memex-vllm memex-web memex-mcp memex-watch

# 6. (After confirming everything works) remove the broken copy
rm -rf ~/.memex/vault.broken
```

Memex's incremental re-indexing handles any drift between the restored derived state and the canonical Markdown automatically — if `documents/` and the index disagree, `memex doctor` will report it and the next ingest / watcher reaction reconciles.

## macOS users

The bash script at `scripts/memex-vault-backup.sh` is portable. Invoke it from a `launchd` plist (`StartCalendarInterval` for a schedule) or from `cron`. The `.service` + `.timer` templates here are Linux-specific — see [`launchd.md`](launchd.md) for the pattern of authoring a user agent.

## What's not covered here

- **Off-machine key backup.** The restic password is *yours to keep safe*. Memex doesn't escrow it. Use a password manager, a safety-deposit box, anything you trust.
- **Failure notifications.** systemd has `OnFailure=email-on-fail@%i.service` patterns; setting one up needs an MTA configured on the box. Out of scope for the template.
- **Pre/post-backup hooks** (e.g., dumping a manifest catalogue). The script is short — fork it and add your own steps.
- **Backups of the orchestrator's model weights**. Those live in `~/.cache/huggingface/`, are re-downloadable, and are unrelated to the vault.
