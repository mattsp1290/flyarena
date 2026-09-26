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
`current` symlink. Older releases remain available. Concurrent deploys are
guarded against directly by `--deploy` mode's own lock (see "Deploy lock and
`Release:` marker" below) -- no caller needs to coordinate this separately.

The script then fetches the public HTML and every emitted asset and compares
them with the local build. It then runs a live Chromium smoke check
(`scripts/verify/live-smoke.ts`, via `node_modules/.bin/tsx`) against
`DEPLOY_URL`: it waits for the app to reach ready, expands the Findings panel,
and confirms step 1's sentence and the model ledger both render. This
requires a Chromium browser installed on the machine running `deploy.sh`
(`npx playwright install chromium` -- a one-time setup step, not run by
`deploy.sh` itself). A failed asset check or smoke check triggers the
rollback described below, still holding the lock, and `deploy.sh` exits
nonzero either way; a successful upload alone is not verified deployment.

## Rollback

`deploy.sh --deploy` performs this automatically on a verification or smoke
failure (see "Rollback orchestration" below); the manual procedure remains
available for any other case (e.g. a problem noticed after the fact, outside
a `deploy.sh` run).

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

## Integrated workbench release

On 2026-09-24, main revision `bc10c22` was deployed as
`20260924T141451Z-e994aaab005c`. Public HTML/assets matched the build, and an actual
Chromium visit completed a real-connectome probe, scrubbed replay, opened the
optional sandbox and returned to a ready Arena without HTTP or page errors.
The configured HTTP host requires portable graph hashing; integrity checks remain
mandatory. Packaging now creates its archive outside `dist` to avoid tar observing
its own output-directory mutation. Detailed tests and independent/live Cursor
thermonuclear reviews are recorded in `docs/counterfactual-validation.md`.

Backfilled for the deploy lock/marker guard added below (WP2 of
`.agents/plans/findings-tour`, `flyarena-o3t0`): this was still the current
live release at the time that guard was added, so it is also this document's
anchor `Release:`/`Commit:` marker pair.

Release: 20260924T141451Z-e994aaab005c
Commit: bc10c22

## Deploy lock and `Release:` marker (WP2 of `.agents/plans/findings-tour`)

`./scripts/deploy.sh`'s `--deploy` mode holds an in-script, stale-aware lock
for its whole run (build through byte-for-byte asset verification and the
live smoke check), implemented in `scripts/deploy-lock.sh` (sourced by
`deploy.sh`, and by `scripts/verify/deploy-lock.test.sh` for local testing):

- It atomically `mkdir`s `<DEPLOY_ROOT>/.deploy.lock` on the deploy host (over
  the same SSH mechanism `deploy.sh` already uses) and writes a sibling
  `<DEPLOY_ROOT>/.deploy.lock.owner` file recording `timestamp=`, `hostname=`,
  `pid=`, and `release=` (the timestamp is decided locally by the machine
  running `deploy.sh`, in UTC epoch seconds, so staleness is judged
  consistently regardless of the deploy host's own clock).
- If the lock already exists and its owner's `timestamp` is under 30 minutes
  old, `deploy.sh` aborts immediately with no changes: another deploy is
  believed to be in progress.
- If the owner's `timestamp` is 30 minutes or older, `deploy.sh` aborts and
  reports it as **stale**. The lock is never broken automatically. Once
  you've confirmed no deploy is actually running, clear it manually:

  ```bash
  ssh "$DEPLOY_SSH" "rm -f '<DEPLOY_ROOT>/.deploy.lock.owner'; rmdir '<DEPLOY_ROOT>/.deploy.lock'"
  ```

  (or run the same two commands directly on the host). `.deploy.lock` is kept
  empty by design (the owner data lives in the sibling `.owner` file), so a
  bare `rmdir <DEPLOY_ROOT>/.deploy.lock` always succeeds once the owner file
  is gone.
- The lock is released via a `trap ... EXIT` in `deploy.sh`, so it is
  released on success, on a `die`, and on any unexpected failure.
- Before proceeding, `deploy.sh` also confirms the live `current` symlink's
  target equals the release id in the **last** `Release:` marker line in this
  file (parsed top-to-bottom; only the last one counts). A mismatch means an
  unrecorded deploy happened outside this guard, and `deploy.sh` aborts
  rather than risk clobbering it or rolling back to the wrong release. That
  same read also becomes `previous_release` -- the pre-deploy active release
  this run will restore to if verification or the live smoke check fails.

### `Release:`/`Commit:` marker format

Every recorded deploy entry below should end with a structured marker pair,
each on its own line, with no other text on that line:

```text
Release: <release id, e.g. 20260924T141451Z-e994aaab005c>
Commit: <short sha the release was built from, e.g. bc10c22>
```

`deploy-lock.sh` parses only `Release:` lines (`Commit:` is for a human
reader correlating the release id back to source); it always uses the last
one in the file. Add a fresh `Release:`/`Commit:` pair to every new deploy
entry -- never edit an old one in place, so this file stays an append-only
log the guard can trust.

### Rollback orchestration (verification/smoke failure)

`deploy.sh` itself performs the rollback -- it stays a single process holding
the lock across build, upload, activation, asset verification, and the live
smoke check, so "roll back while still holding the lock" needs no separate
wrapper or a second lock acquisition. On a byte-for-byte verification failure
or a failed live smoke check, `deploy.sh`:

1. re-reads the live `current` target; rolls back **only if** it still equals
   the release this run just published (otherwise another deploy has
   happened -- it leaves `current` untouched and reports);
2. restores the previous release with the documented symlink swap
   (`ln -s releases/<previous> .rollback-current && mv -Tf .rollback-current current`);
3. checks HTTP 200 on the entry URL, and re-runs the live smoke check,
   best-effort (a smoke-check environment issue mid-incident should not mask
   whether the HTTP 200 check -- the authoritative rollback-health signal --
   passed);
4. exits nonzero. `deploy.sh` does not itself edit or push this file or close
   the bean; the operator running it copies the reported outcome (failure
   reason, rollback target, post-rollback check results) into a new dated
   entry here, per the format above, and leaves the bean open.
