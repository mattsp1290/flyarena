# Deployment

The preferred deployment entry point is `./scripts/deploy.sh`. Run it from any
directory with Node 22 (at least 22.22.2), npm, Bash, tar, OpenSSH, and curl on PATH.
If Node or npm is missing from PATH, the script loads nvm from `$NVM_DIR`
(defaulting to `~/.nvm`) and selects an installed Node 22. This also supports
shells that lazy-load nvm. If Node 22 is not installed, run `nvm install 22`
first. To switch an already active Node version, run `nvm use 22`.

Copy `.env.example` to `.env`, restrict it with `chmod 600 .env`, and fill in the
deployment URL, SSH destination, and confirmed dedicated app directory. `.env`
is trusted shell configuration and is ignored by Git. Never put actual hostnames,
IP addresses, or SSH destinations in tracked documentation, scripts, or `VITE_*`
variables. The backend placement requested by the owner is recorded as
`BACKEND_SSH` in `.env`. The default arena and counterfactual workbench have no backend. The optional
synthetic sandbox has a separate service; place it on that designated machine.

## Hosting contract

Consult `~/.agents/` on the infrastructure SSH destination before configuring
`DEPLOY_ROOT`. It must be a dedicated directory ending in `/flyarena`, beneath
`/srv`, `/var/www`, `/opt`, or `/home`, writable by the deployment SSH account.
The static web server must map `/fly/` to `DEPLOY_ROOT/current/` and redirect
`/fly` to `/fly/`. The latter ensures relative browser URLs work correctly.
Configure that route in the authoritative infrastructure configuration; this
script does not guess or overwrite web-server configuration.

The current host uses Apache in Kubernetes. Its deployment has a read-only
hostPath mount of `DEPLOY_ROOT` at `/usr/local/apache2/flyarena`. Mount the entire
app directory, not `current` as a subPath, so release switches remain visible.
The Apache configuration includes:

```apache
LoadModule alias_module modules/mod_alias.so
RedirectMatch 301 ^/fly$ /fly/
Alias /fly/ /usr/local/apache2/flyarena/current/
<Directory "/usr/local/apache2/flyarena">
  Options -Indexes +FollowSymLinks
  AllowOverride None
  Require all granted
</Directory>
```

The initial setup updated the existing Apache deployment manifest on the
infrastructure host as well as the live ConfigMap and Deployment. Original
configuration backups are in `~/.agents/deploy-backups/flyarena-initial/` on that
host. No Kubernetes access or web-server restart is needed for routine deploys.

The SSH host key must already be trusted, and key-based authentication must work
without prompts. SSH aliases and IdentityFile settings in `~/.ssh/config` may be
used. Do not disable host-key verification.

## Build and publish

```bash
./scripts/deploy.sh --build-only
./scripts/deploy.sh
```

Both modes install locked dependencies, run Svelte/TypeScript checks, and build
with the configured `/fly/` base path. The package is `dist/flyarena.tar.gz` and
contains only the built static site. `--build-only` needs no remote access.
Deployment uploads a unique release, extracts it, and atomically switches the
`current` symlink. Older releases remain available. Concurrent deploys should be
avoided because each activation changes the same symlink.

The script then fetches the public HTML and every emitted asset and compares
them with the local build. A failed check exits nonzero; the release remains
active for inspection. A successful upload alone is not verified deployment.

## Rollback

On the infrastructure host, inspect the dedicated app directory's `releases/`
and select a previously verified release. From that app directory, create a
temporary symlink and atomically replace `current`:

```bash
ln -s releases/SELECTED_RELEASE .rollback-current
mv -Tf .rollback-current current
```

Verify the public page and its assets after rollback. Release cleanup is manual;
retain the current release and at least one known-good predecessor.

## Verified deployment

On 2026-09-23, the script completed a live deployment and byte-for-byte public
HTML/asset verification. A Chromium smoke check confirmed the `/fly` redirect,
rendered application, and absence of failed resources or JavaScript errors.
The existing homepage continued to return HTTP 200. Svelte/TypeScript checks
and all 58 unit tests passed. Sensitive connection details remain in `.env`.

## Optional DGX sandbox packaging

The arena and counterfactual workbench remain static and require no backend.
The separate synthetic sandbox can use a local ARM64 GB10 backend.
`./scripts/deploy.sh --package-backend` creates `dist/flyarena-backend.tar.gz`
without reading `.env`, contacting a server, or starting a service. The bundle
contains only the Dockerfile, Python metadata/source, and `scripts/lab.sh`.

Extract a reviewed bundle into a dedicated release directory. Build with
`./scripts/lab.sh --build`; supply a fresh `LAB_TOKEN` of at least 16 characters
and start `./scripts/lab.sh --cuda` (`--cpu` omits GPU access). It binds to
loopback port 8765 by default. Health is `/api/v1/health`; job routes require a
bearer token. Use `LAB_CONTAINER_NAME` and `LAB_PORT` for a temporary test instance.
`LAB_ORIGINS` must list exact approved frontend origins; wildcard CORS is rejected.
Never embed tokens in Vite variables, URLs or static artifacts.

Remote use requires an operator-managed authenticated HTTPS proxy or private
tunnel. The pinned ARM64 NVIDIA image is large, and the 6 GiB container RAM limit
and 2 GiB PyTorch tensor allocator cap do not fully isolate unified GPU memory.
Only one ASGI worker is supported. Verify authenticated completion, cancellation
and export before treating a backend release as operational.

Retain the previous image for rollback. Stop only the intended lab container and
restart that image with the same approved token/origins. Active jobs are ephemeral;
export before restarting. Static release rollback and backend rollback are separate.
This integration does not replace an already-running backend automatically.
