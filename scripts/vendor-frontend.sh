#!/usr/bin/env bash
# Vendor the runtime frontend dependencies that Memex needs to render
# its web UI fully offline. Run once before `memex serve web`; the
# fetched files land under `src/memex/webui/static/` and are committed
# (small static assets, no build step). Re-run to upgrade pins.
#
# Tailwind itself is hand-curated in `src/memex/webui/static/tailwind.css`
# — only the utility classes the templates actually use, so the file
# stays small and we don't carry the JIT engine into runtime.
#
# This script only downloads what we can't reasonably hand-write: the
# HTMX client. Pinned with a SHA-384 integrity hash so a tampered
# mirror is rejected.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATIC="$HERE/../src/memex/webui/static"
mkdir -p "$STATIC"

# ----- HTMX 1.9.10 -----
HTMX_VERSION="1.9.10"
HTMX_URL="https://unpkg.com/htmx.org@${HTMX_VERSION}/dist/htmx.min.js"
HTMX_SHA384="D1Kt99CQMDuVetoL1lrYwg5t+9QdHe7NLX/SoJYkXDFfX37iInKRy5xLSi8nO7UC"
HTMX_OUT="$STATIC/htmx.min.js"

echo "[vendor] fetching htmx ${HTMX_VERSION} → $HTMX_OUT"
curl --fail --silent --show-error --location \
    --output "$HTMX_OUT.tmp" \
    "$HTMX_URL"

# Verify the SHA-384 integrity matches what's pinned in base.html.
ACTUAL_SHA384="$(openssl dgst -sha384 -binary "$HTMX_OUT.tmp" | openssl base64 -A)"
if [[ "$ACTUAL_SHA384" != "$HTMX_SHA384" ]]; then
    echo "[vendor] FAIL: htmx SHA-384 mismatch" >&2
    echo "[vendor]   expected: $HTMX_SHA384" >&2
    echo "[vendor]   actual:   $ACTUAL_SHA384" >&2
    rm -f "$HTMX_OUT.tmp"
    exit 1
fi
mv "$HTMX_OUT.tmp" "$HTMX_OUT"
echo "[vendor] ok: htmx ${HTMX_VERSION} ($(wc -c < "$HTMX_OUT") bytes)"

# ----- Cytoscape 3.30.4 (the /graph/{id} neighbourhood viz) -----
CYTO_VERSION="3.30.4"
CYTO_URL="https://unpkg.com/cytoscape@${CYTO_VERSION}/dist/cytoscape.min.js"
CYTO_SHA384="H3uzGzTfGHUAumB8+s4GEdfFwzAceN9wCCndN8AXubWKFIPuBSWKKtWDx7RhSf/z"
CYTO_OUT="$STATIC/cytoscape.min.js"

echo "[vendor] fetching cytoscape ${CYTO_VERSION} → $CYTO_OUT"
curl --fail --silent --show-error --location \
    --output "$CYTO_OUT.tmp" \
    "$CYTO_URL"

ACTUAL_CYTO_SHA384="$(openssl dgst -sha384 -binary "$CYTO_OUT.tmp" | openssl base64 -A)"
if [[ "$ACTUAL_CYTO_SHA384" != "$CYTO_SHA384" ]]; then
    echo "[vendor] FAIL: cytoscape SHA-384 mismatch" >&2
    echo "[vendor]   expected: $CYTO_SHA384" >&2
    echo "[vendor]   actual:   $ACTUAL_CYTO_SHA384" >&2
    rm -f "$CYTO_OUT.tmp"
    exit 1
fi
mv "$CYTO_OUT.tmp" "$CYTO_OUT"
echo "[vendor] ok: cytoscape ${CYTO_VERSION} ($(wc -c < "$CYTO_OUT") bytes)"

echo "[vendor] done. The webui now renders fully offline."
