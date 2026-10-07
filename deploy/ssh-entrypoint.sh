#!/usr/bin/env bash
# Installed as /usr/local/bin/ml-help-deploy. The deploy key in authorized_keys is pinned to this command.
set -euo pipefail
if [[ "${SSH_ORIGINAL_COMMAND:-}" =~ ^deploy\ ([0-9a-f]{40})$ ]]; then
  exec bash /usr/local/lib/ml-help/deploy.sh "${BASH_REMATCH[1]}"
fi
echo 'Only ML Help deployment commands are permitted' >&2
exit 1
