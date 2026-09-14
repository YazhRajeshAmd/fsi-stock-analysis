#!/usr/bin/env bash
# TEMPLATE — copy to scripts/build-image.sh in your app repo, then chmod +x.
# Canonical copy: skills/slai-app-creator/assets/templates/build-image.sh.example
#
# Build linux/amd64 locally (no registry). Same engine selection as publish-image-harbor.sh.example
# (Docker if usable, else Podman + Pandora paths).
#
# Repo root = parent of scripts/ (works for standalone app repos). Pair with .env.example for IMAGE_*.
#
# Usage:
#   cp skills/slai-app-creator/assets/templates/build-image.sh.example scripts/build-image.sh && chmod +x scripts/build-image.sh
#   cp -n .env.example .env   # IMAGE_NAME, BUILD_CONTEXT
#   ./scripts/build-image.sh
#
# Optional: IMAGE_TAG, LOCAL_IMAGE, DOCKERFILE, PANDORA_ROOT, PODMAN, RUNC, FORCE_PODMAN=1, SKIP_DOTENV=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if [[ -z "${SKIP_DOTENV:-}" && -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.env"
  set +a
fi

: "${BUILD_CONTEXT:?Set BUILD_CONTEXT (directory containing Dockerfile, relative to repo root) or add to .env}"

if [[ -z "${IMAGE_TAG:-}" ]]; then
  if [[ -d "$ROOT/.git" ]]; then
    IMAGE_TAG="$(git -C "$ROOT" rev-parse HEAD)"
  else
    IMAGE_TAG="local-$(date -u +%Y%m%d%H%M%S)"
  fi
fi
DOCKERFILE="${DOCKERFILE:-$BUILD_CONTEXT/Dockerfile}"

if [[ -n "${LOCAL_IMAGE:-}" ]]; then
  TAGGED="$LOCAL_IMAGE"
else
  : "${IMAGE_NAME:?Set IMAGE_NAME for local tag, or set LOCAL_IMAGE explicitly}"
  TAGGED="${IMAGE_NAME}:${IMAGE_TAG}"
fi

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "Dockerfile not found: $DOCKERFILE" >&2
  exit 1
fi

use_docker() {
  [[ -z "${FORCE_PODMAN:-}" ]] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

if use_docker; then
  echo "Using docker → ${TAGGED}"
  docker build --platform linux/amd64 -f "$DOCKERFILE" -t "$TAGGED" "$BUILD_CONTEXT"
  echo "Built ${TAGGED}"
  exit 0
fi

# --- Podman (Pandora / NFS-safe /tmp storage) ---

resolve_podman() {
  if [[ -n "${PODMAN:-}" && -x "$PODMAN" ]]; then echo "$PODMAN"; return 0; fi
  if command -v podman >/dev/null 2>&1; then command -v podman; return 0; fi
  if [[ -x /tool/pandora/bin/podman ]]; then echo /tool/pandora/bin/podman; return 0; fi
  shopt -s nullglob
  local p64=(/tool/pandora64/.package/podman-*/bin/podman)
  shopt -u nullglob
  if [[ ${#p64[@]} -gt 0 ]]; then echo "${p64[-1]}"; return 0; fi
  return 1
}

first_runc_under() {
  local rdir="$1"
  shopt -s nullglob
  local r=("$rdir"/.package/runc-*/bin/runc)
  shopt -u nullglob
  [[ ${#r[@]} -gt 0 ]] && echo "${r[0]}" && return 0
  return 1
}

resolve_runc() {
  if [[ -n "${RUNC:-}" && -x "$RUNC" ]]; then echo "$RUNC"; return 0; fi
  local root=""
  [[ -n "${PANDORA_ROOT:-}" ]] && root="$PANDORA_ROOT"
  local podman_bin="$1"
  if [[ "$podman_bin" == /tool/pandora/bin/podman ]]; then root=/tool/pandora
  elif [[ "$podman_bin" == /tool/pandora64/.package/podman-* ]]; then root=/tool/pandora64; fi
  if [[ -n "$root" ]]; then
    local found
    found=$(first_runc_under "$root" || true)
    [[ -n "$found" ]] && echo "$found" && return 0
  fi
  command -v runc >/dev/null 2>&1 && command -v runc && return 0
  return 1
}

PODMAN_BIN=$(resolve_podman) || {
  echo "Neither working docker nor podman found. Install Docker or Podman (Pandora: /tool/pandora)." >&2
  exit 1
}

RUNC=$(resolve_runc "$PODMAN_BIN" || true)
BASE=$(mktemp -d /tmp/pmXXXXXX)
ROOTPM="$BASE/rt"
RUNROOT="$BASE/rn"
mkdir -p "$ROOTPM" "$RUNROOT"
export REGISTRY_AUTH_FILE="$BASE/auth.json"
trap 'rm -rf "$BASE" 2>/dev/null || true' EXIT

pm=( "$PODMAN_BIN" --root "$ROOTPM" --runroot "$RUNROOT" )
[[ -n "$RUNC" ]] && pm+=( --runtime "$RUNC" )

echo "Using podman → ${TAGGED}"

"${pm[@]}" build \
  --platform linux/amd64 \
  --storage-opt overlay.ignore_chown_errors=true \
  -f "$DOCKERFILE" \
  -t "$TAGGED" \
  "$BUILD_CONTEXT"

echo "Built ${TAGGED}"
