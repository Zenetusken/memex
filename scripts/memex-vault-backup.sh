#!/usr/bin/env bash
# scripts/memex-vault-backup.sh — restic-driven incremental backup of
# `$MEMEX_VAULT_PATH`. Idempotent; safe to re-run by hand. Designed
# to be invoked by `memex-vault-backup.service` (fired by the daily
# `.timer`) but works equally well from a shell.
#
# Reads env vars (defaults baked in):
#   MEMEX_VAULT_PATH            (default: $HOME/.memex/vault)
#   RESTIC_REPOSITORY           (default: $HOME/.local/state/memex/backups)
#   RESTIC_PASSWORD_FILE        (REQUIRED — no default; refuses to run)
#   MEMEX_BACKUP_KEEP_DAILY     (default: 7)
#   MEMEX_BACKUP_KEEP_WEEKLY    (default: 4)
#   MEMEX_BACKUP_KEEP_MONTHLY   (default: 6)
#   RESTIC_INIT_IF_MISSING      (default: true — auto-init on first run)
#
# See docs/deploy/backup.md for the full deployment story (cloud
# targets, restore flow, password-file generation).

set -euo pipefail

VAULT="${MEMEX_VAULT_PATH:-$HOME/.memex/vault}"
REPO="${RESTIC_REPOSITORY:-$HOME/.local/state/memex/backups}"
KEEP_DAILY="${MEMEX_BACKUP_KEEP_DAILY:-7}"
KEEP_WEEKLY="${MEMEX_BACKUP_KEEP_WEEKLY:-4}"
KEEP_MONTHLY="${MEMEX_BACKUP_KEEP_MONTHLY:-6}"

if ! command -v restic >/dev/null 2>&1; then
    echo "memex-vault-backup.sh: restic not found." >&2
    echo "  install: apt install restic | brew install restic | pacman -S restic | dnf install restic" >&2
    exit 127
fi

if [ -z "${RESTIC_PASSWORD_FILE:-}" ]; then
    echo "memex-vault-backup.sh: RESTIC_PASSWORD_FILE is required." >&2
    echo "  Create one:" >&2
    echo "    mkdir -p ~/.config/memex" >&2
    echo "    printf 'your-passphrase' > ~/.config/memex/restic-password" >&2
    echo "    chmod 600 ~/.config/memex/restic-password" >&2
    echo "    export RESTIC_PASSWORD_FILE=~/.config/memex/restic-password" >&2
    exit 2
fi

if [ ! -f "$RESTIC_PASSWORD_FILE" ]; then
    echo "memex-vault-backup.sh: password file does not exist: $RESTIC_PASSWORD_FILE" >&2
    exit 2
fi

export RESTIC_REPOSITORY="$REPO"

if [ ! -d "$VAULT" ]; then
    echo "memex-vault-backup.sh: vault does not exist: $VAULT" >&2
    echo "  (Did you set MEMEX_VAULT_PATH? Default is \$HOME/.memex/vault.)" >&2
    exit 3
fi

# Initialize the repo on first run. We disambiguate "repo doesn't
# exist yet" (we should init) from "repo exists but we can't read
# it" (perms, corruption, wrong password) by inspecting restic's
# stderr — fresh-repo errors mention "does not exist" / "config
# file" / "unable to open config" / "Is there a repository". Other
# errors (permission denied, wrong password, network failure) keep
# the existing repo intact and surface the real error.
SNAPSHOT_ERR="$(restic snapshots --no-lock 2>&1 >/dev/null || true)"
if [ -z "$SNAPSHOT_ERR" ]; then
    : # repo readable; proceed to backup
elif echo "$SNAPSHOT_ERR" | grep -qE 'does not exist|unable to open config|config file|Is there a repository'; then
    if [ "${RESTIC_INIT_IF_MISSING:-true}" = "true" ]; then
        echo "→ initializing restic repo at $REPO"
        restic init
    else
        echo "memex-vault-backup.sh: repo not initialized + auto-init disabled." >&2
        echo "  run: restic -r $REPO init" >&2
        exit 4
    fi
else
    # Repo exists but we can't read it — never auto-init over it.
    echo "memex-vault-backup.sh: restic could not read the repo:" >&2
    echo "$SNAPSHOT_ERR" >&2
    echo "  (refusing to auto-init; resolve the error above first)" >&2
    exit 5
fi

echo "→ backup $VAULT → $REPO"
restic backup "$VAULT" \
    --tag memex-vault \
    --exclude "$VAULT/.memex/locks" \
    --exclude "$VAULT/.memex/daemon" \
    --exclude "$VAULT/.memex/events.sqlite"

echo "→ prune (keep-daily=$KEEP_DAILY keep-weekly=$KEEP_WEEKLY keep-monthly=$KEEP_MONTHLY)"
restic forget \
    --tag memex-vault \
    --keep-daily "$KEEP_DAILY" \
    --keep-weekly "$KEEP_WEEKLY" \
    --keep-monthly "$KEEP_MONTHLY" \
    --prune

echo "✓ memex vault backup complete"
