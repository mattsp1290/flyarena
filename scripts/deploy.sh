#!/usr/bin/env bash
# Build and publish a static release. Endpoints belong only in the ignored .env.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

die() { printf 'deploy: %s\n' "$*" >&2; exit 1; }
mode=${1:---deploy}
[[ $# -le 1 ]] || die 'Expected at most one option.'
case "$mode" in
  --package-backend)
    mkdir -p dist
    tar -czf dist/flyarena-backend.tar.gz \
      backend/Dockerfile backend/pyproject.toml \
      backend/flyarena_lab/__init__.py backend/flyarena_lab/cli.py \
      backend/flyarena_lab/experiment.py backend/flyarena_lab/model.py \
      backend/flyarena_lab/service.py scripts/lab.sh
    printf 'Packaged dist/flyarena-backend.tar.gz; no remote actions performed.\n'
    exit 0 ;;
  --deploy|--build-only) ;;
  --help) printf 'Usage: ./scripts/deploy.sh [--deploy|--build-only|--package-backend]\nBackend packaging is offline and needs no .env. Configure .env for static build/deploy. Default: build, upload, activate, verify.\n'; exit 0 ;;
  *) die 'Unknown option; use --help.' ;;
esac
[[ -f .env ]] || die 'Copy .env.example to .env and configure deployment.'
# This is a trusted local configuration file, not input from the server.
source .env
base=${DEPLOY_BASE:-/fly/}
[[ "$base" =~ ^/([a-zA-Z0-9_-]+/)+$ ]] || die 'DEPLOY_BASE must be an absolute URL path ending in /.'
# Shell functions that lazy-load nvm are not inherited by this Bash process.
if ! command -v node >/dev/null || ! command -v npm >/dev/null; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  if [[ -s "$NVM_DIR/nvm.sh" ]]; then
    source "$NVM_DIR/nvm.sh" --no-use
    nvm use --silent 22 || die 'Install Node 22 with `nvm install 22`, then retry.'
  fi
fi
for command in node npm; do
  command -v "$command" >/dev/null || die "Missing command: $command. Install Node.js 22 (at least 22.22.2) and npm, or run nvm install 22."
done
command -v tar >/dev/null || die 'Missing command: tar'

if [[ "$mode" == --deploy ]]; then
  for command in ssh scp curl cmp; do
    command -v "$command" >/dev/null || die "Missing command: $command"
  done
  [[ ${DEPLOY_SSH:-} =~ ^[a-zA-Z0-9_-]+@[a-zA-Z0-9.-]+$ ]] || die 'Set DEPLOY_SSH to user@host in .env.'
  [[ ${DEPLOY_ROOT:-} =~ ^/(srv|var/www|opt|home)/[a-zA-Z0-9_/-]+$ ]] || die 'Set DEPLOY_ROOT to a dedicated app directory below /srv, /var/www, /opt, or /home.'
  [[ "$DEPLOY_ROOT" != *//* && "$DEPLOY_ROOT" != */ ]] || die 'DEPLOY_ROOT must not contain // or end in /.'
  [[ "${DEPLOY_ROOT##*/}" == flyarena ]] || die 'Use a dedicated DEPLOY_ROOT ending in /flyarena.'
  [[ ${DEPLOY_URL:-} =~ ^https?://[^/[:space:]]+${base%/}/?$ ]] || die 'DEPLOY_URL must match DEPLOY_BASE.'
  ssh_options=(-o BatchMode=yes -o ConnectTimeout=10)
  # Refuse to replace an existing directory or another application's symlink.
  ssh "${ssh_options[@]}" "$DEPLOY_SSH" "bash -s -- '$DEPLOY_ROOT'" <<'REMOTE'
set -euo pipefail
root=$1
if [[ -e "$root/current" || -L "$root/current" ]]; then
  [[ -L "$root/current" && $(readlink "$root/current") == releases/* ]] || {
    echo 'deploy: current must be a managed releases/ symlink.' >&2; exit 1;
  }
fi
mkdir -p "$root/releases"
test -w "$root/releases"
REMOTE
fi

# Never export deployment settings to Vite; only the public base path is needed.
npm ci --no-audit --no-fund
npm run check
npm run build -- --base "$base"
[[ -f dist/index.html ]] || die 'Build did not produce dist/index.html.'
tar -czf dist/flyarena.tar.gz --exclude=flyarena.tar.gz -C dist .
printf 'Built dist/ and dist/flyarena.tar.gz for %s\n' "$base"
[[ "$mode" == --deploy ]] || exit 0

release="$(date -u +%Y%m%dT%H%M%SZ)-$(node -e 'console.log(require("node:crypto").randomBytes(6).toString("hex"))')"
remote_release="$DEPLOY_ROOT/releases/$release"
ssh "${ssh_options[@]}" "$DEPLOY_SSH" "mkdir '$remote_release'"
scp "${ssh_options[@]}" dist/flyarena.tar.gz "$DEPLOY_SSH:$remote_release/package.tar.gz"
ssh "${ssh_options[@]}" "$DEPLOY_SSH" "bash -s -- '$DEPLOY_ROOT' '$release'" <<'REMOTE'
set -euo pipefail
root=$1
release=$2
tar -xzf "$root/releases/$release/package.tar.gz" -C "$root/releases/$release"
test -s "$root/releases/$release/index.html"
rm -- "$root/releases/$release/package.tar.gz"
chmod -R a+rX "$root/releases/$release"
ln -s "releases/$release" "$root/.current-$release"
mv -Tf "$root/.current-$release" "$root/current"
REMOTE

# Compare public HTML and every emitted asset with the exact local build.
verify_dir=$(mktemp -d)
trap 'rm -rf -- "$verify_dir"' EXIT
curl --fail --silent --show-error --location --connect-timeout 10 --max-time 60 \
  -H 'Cache-Control: no-cache' "${DEPLOY_URL%/}" -o "$verify_dir/response"
cmp -s dist/index.html "$verify_dir/response" || die 'Public entry URL does not serve this release. Inspect the web-server route; previous releases are retained.'
while IFS= read -r -d '' file; do
  relative=${file#dist/}
  url="${DEPLOY_URL%/}/$relative"
  curl --fail --silent --show-error --location --connect-timeout 10 --max-time 60 \
    -H 'Cache-Control: no-cache' "$url" -o "$verify_dir/response"
  cmp -s "$file" "$verify_dir/response" || die "Public content mismatch: $relative. Release is active; inspect the web-server mapping/cache. Previous releases are retained."
done < <(find dist -type f ! -name flyarena.tar.gz -print0)
printf 'Deployment verified. Release: %s\nThe static arena/workbench require no backend. Optional DGX sandbox placement is BACKEND_SSH in .env.\n' "$release"
