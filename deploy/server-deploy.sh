#!/usr/bin/env bash
# Installed as /usr/local/lib/ml-help/deploy.sh (root-owned). Runs as ubuntu via the restricted SSH key.
set -Eeuo pipefail
cd /home/ubuntu/tanmay/ml-help
exec 9>/home/ubuntu/tanmay/.ml-help-deploy.lock
flock -w 900 9
revision=${1:-}
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo 'Expected a full commit SHA'; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo 'Server worktree has local changes'; exit 1; }
git fetch origin main
[[ "$(git rev-parse origin/main)" == "$revision" ]] || { echo 'Skipping superseded deployment'; exit 0; }
git merge --ff-only "$revision"
previous=$(docker inspect --format '{{.Image}}' ml-help-app-1 2>/dev/null || true)
docker compose -p ml-help build app
if ! docker compose -p ml-help up -d --no-build --wait --wait-timeout 120 app; then
  if [[ -n "$previous" ]]; then
    docker tag "$previous" ml-help-rollback
    ML_HELP_IMAGE=ml-help-rollback docker compose -p ml-help up -d --no-build --wait --wait-timeout 120 app
    echo 'Restored previous image after failed health check'
  fi
  exit 1
fi
curl --fail --silent --show-error http://127.0.0.1:5080/api/health
echo "Deployed $revision"
