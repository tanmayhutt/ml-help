#!/usr/bin/env bash
# Deploy ML Help to the shared server. Touches ONLY ~/tanmay/ml-help and adds ONE import line to the Caddyfile.
# Usage: ./deploy/deploy.sh            (sync + build + up, Caddy step only if the import is missing)
#        ./deploy/deploy.sh --no-caddy (skip the Caddy step entirely)
set -euo pipefail
HOST="ubuntu@15.206.247.203"
REMOTE_DIR="tanmay/ml-help"
PORT=5080
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Syncing source to $HOST:$REMOTE_DIR"
ssh "$HOST" "mkdir -p $REMOTE_DIR"
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude 'backend/.venv' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'data' --exclude 'context.md' --exclude '.env*' --exclude 'node_modules' \
  "$HERE/" "$HOST:$REMOTE_DIR/"

echo "==> Checking port $PORT is free or already ours"
ssh "$HOST" "if ss -tlnp | grep -q '127.0.0.1:$PORT ' && ! docker ps --format '{{.Names}} {{.Ports}}' | grep -q 'ml-help-app-1.*$PORT'; then echo 'Port $PORT is used by something else. Change compose.yaml and the Caddy file.'; exit 1; fi"

echo "==> Building and starting the container (project name ml-help only)"
ssh "$HOST" "cd $REMOTE_DIR && docker compose -p ml-help build --pull && docker compose -p ml-help up -d && docker image prune -f --filter label=com.docker.compose.project=ml-help >/dev/null 2>&1 || true"

echo "==> Waiting for health"
ssh "$HOST" "for i in \$(seq 1 20); do curl -fsS http://127.0.0.1:$PORT/api/health >/dev/null && echo healthy && exit 0; sleep 3; done; echo 'not healthy'; docker logs --tail 50 ml-help-app-1; exit 1"

if [[ "${1:-}" == "--no-caddy" ]]; then echo "==> Skipping Caddy"; exit 0; fi

echo "==> Caddy: add site file and a single import line if missing (backs up Caddyfile first)"
ssh "$HOST" bash -s <<'REMOTE'
set -euo pipefail
SITE=/etc/caddy/ml-help.caddy
MAIN=/etc/caddy/Caddyfile
if sudo -n true 2>/dev/null; then SUDO="sudo -n"; else echo "sudo needs a password on this host; run the Caddy step manually (see deploy/README.md)"; exit 0; fi
$SUDO cp ~/tanmay/ml-help/deploy/ml-help.caddy "$SITE"
if ! grep -q "import /etc/caddy/ml-help.caddy" "$MAIN"; then
  $SUDO cp "$MAIN" "$MAIN.bak-$(date +%Y%m%d-%H%M%S)-before-ml-help"
  printf '\nimport /etc/caddy/ml-help.caddy\n' | $SUDO tee -a "$MAIN" >/dev/null
  echo "import line appended"
fi
$SUDO caddy validate --config "$MAIN" --adapter caddyfile
$SUDO systemctl reload caddy
echo "caddy reloaded"
REMOTE
echo "==> Done: https://ml.tanmaytiwari.me"
