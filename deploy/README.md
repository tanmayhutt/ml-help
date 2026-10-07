# Deploying ML Help

Target: the shared Ubuntu box, folder `~/tanmay/ml-help`, container bound to `127.0.0.1:5080`, Caddy site `ml.tanmaytiwari.me`.

The server runs several other apps. The deploy script never touches their folders, containers, or Caddy blocks. The only shared file it edits is `/etc/caddy/Caddyfile`, and only by appending one `import` line after taking a timestamped backup.

## First deploy

1. Point the DNS record `ml.tanmaytiwari.me` at the server.
2. Run `./deploy/deploy.sh` from this repository.
3. If sudo prompts for a password on the host, do the Caddy step by hand:

```bash
sudo cp ~/tanmay/ml-help/deploy/ml-help.caddy /etc/caddy/ml-help.caddy
sudo cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.bak-$(date +%Y%m%d-%H%M%S)-before-ml-help
echo 'import /etc/caddy/ml-help.caddy' | sudo tee -a /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl reload caddy
```

## Later deploys

Push to `main`. The **Deploy to Ubuntu** GitHub Actions workflow SSHes in with the `ML_HELP_DEPLOY_KEY` secret (host key pinned by `ML_HELP_KNOWN_HOSTS`). That key is restricted in `authorized_keys` to `/usr/local/bin/ml-help-deploy`, which only accepts `deploy <sha>` and runs the root-owned `/usr/local/lib/ml-help/deploy.sh`. The server checkout fast-forwards to that commit, builds, swaps the container, waits for health, and restores the previous image if health fails. `deploy/ssh-entrypoint.sh` and `deploy/server-deploy.sh` are the sources of those two installed files; changing them requires reinstalling on the host.

`./deploy/deploy.sh --no-caddy` still works for a manual rsync deploy, but it leaves the server checkout dirty for the next push. Prefer pushing.

## Private instance

Create `~/tanmay/ml-help/.env.production` on the server with `ML_ACCESS_TOKEN=<long random string>`, then `docker compose -p ml-help up -d`. The UI asks for the token once and stores it in the browser.

## Cost controls baked in

- One job at a time, queue capped at 6. Each job runs in a throwaway process with a 180 s wall clock and a 2 GB address-space limit.
- Container ceiling: 2 CPUs, 3 GB RAM, 128 processes. sklearn uses at most 2 threads.
- Model race subsamples to 5000 rows, tuning to 5000 rows and 20 trials, t-SNE to 1500 rows.
- Uploads capped at 20 MB, 200k rows, 200 columns, 6 per minute per IP. Datasets and models are deleted after 7 days, at most 40 kept.
- No background loops, no cron, no Redis, no Postgres. Idle cost is one sleeping uvicorn process.

All caps are environment variables; see `backend/app/config.py`.

## Rollback

`docker compose -p ml-help down` stops only this app. Remove the import line from the Caddyfile and reload Caddy to take the site offline.
