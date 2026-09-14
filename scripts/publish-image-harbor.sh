#!/usr/bin/env bash
# TEMPLATE — copy to scripts/publish-image-harbor.sh in your app repo, then chmod +x.
# Canonical copy: skills/slai-app-creator/assets/templates/publish-image-harbor.sh.example
#
# Publish linux/amd64 to Harbor. Prefer a verified image-build-lsf job result;
# otherwise build locally with Docker or Podman. Writes .cache/harbor-last-image.env
# (FULL_IMAGE, IMAGE_TAG) after success — use for deployment.yaml image: (see SKILL.md §1).
#
# Repo root = parent of scripts/. Pair with .env.example for HARBOR_* and IMAGE_*.
# If HARBOR_USERNAME/HARBOR_PASSWORD are not already set, this script installs/uses the
# Harbor CLI and reads the short-lived robot credentials it stores locally.
#
# Usage:
#   cp skills/slai-app-creator/assets/templates/publish-image-harbor.sh.example scripts/publish-image-harbor.sh && chmod +x scripts/publish-image-harbor.sh
#   cp -n .env.example .env
#   IMAGE_BUILD_LSF_JOB_DIR=/absolute/path/to/.oci-build/jobs/job.XXXXXXXX \
#     ./scripts/publish-image-harbor.sh  # preferred: load verified OCI result, then push
#   ./scripts/publish-image-harbor.sh    # fallback: local Docker/Podman build, then push
#
# Optional: IMAGE_BUILD_LSF_JOB_DIR (verified LSF result), IMAGE_TAG, DOCKERFILE,
#           PANDORA_ROOT, PODMAN, RUNC, FORCE_PODMAN=1,
#           PODMAN_TEMP_PARENT (private image/auth storage; default /tmp),
#           PODMAN_RUNROOT_PARENT (short runtime state; default /tmp),
#           HARBOR_LAST_IMAGE_FILE (default: $ROOT/.cache/harbor-last-image.env), SKIP_DOTENV=1,
#           HARBOR_CLI_BIN, HARBOR_CLI_BASE_URL, HARBOR_CLI_INSTALL_DIR, HARBOR_CONFIG_DIR.
# DEVELOPER-OWNED ALTERNATE HARBOR (no approval required): when supplying
# HARBOR_USERNAME/HARBOR_PASSWORD directly for a non-default location, also set
# HARBOR_CREDENTIAL_LOCATION=hostname/project to bind those credentials.

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

: "${HARBOR_REGISTRY:?Set HARBOR_REGISTRY (e.g. mkmhub.amd.com) or add to .env}"
: "${HARBOR_PROJECT:?Set HARBOR_PROJECT (e.g. hw-slaiapp-dev) or add to .env}"
: "${IMAGE_NAME:?Set IMAGE_NAME (Harbor repository name) or add to .env}"
if [[ -z "${IMAGE_BUILD_LSF_JOB_DIR:-}" ]]; then
  : "${BUILD_CONTEXT:?Set BUILD_CONTEXT for a local build, or IMAGE_BUILD_LSF_JOB_DIR for the preferred LSF result}"
fi

if [[ ! "${HARBOR_REGISTRY}" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*(:[0-9]{1,5})?$ ]]; then
  echo "HARBOR_REGISTRY must be a hostname with optional port, without a scheme or path." >&2
  exit 1
fi
if [[ ! "${HARBOR_PROJECT}" =~ ^[a-z0-9][a-z0-9._-]*$ ]] ||
   [[ ! "${IMAGE_NAME}" =~ ^[a-z0-9][a-z0-9._-]*$ ]]; then
  echo "HARBOR_PROJECT and IMAGE_NAME must each be one lowercase Harbor path segment." >&2
  exit 1
fi

LSF_ARCHIVE=""
LSF_SOURCE_IMAGE=""
LSF_MANIFEST_DIGEST=""
if [[ -n "${IMAGE_BUILD_LSF_JOB_DIR:-}" ]]; then
  mapfile -t lsf_result < <(python12 - "$IMAGE_BUILD_LSF_JOB_DIR" <<'PY'
import hashlib
import json
import os
import re
import stat
import sys
import tarfile
from pathlib import Path

raw = sys.argv[1]
if not raw or raw.startswith("-") or "\n" in raw or "\r" in raw:
    raise SystemExit("IMAGE_BUILD_LSF_JOB_DIR is malformed")
spelled = Path(raw)
if not spelled.is_absolute():
    raise SystemExit("IMAGE_BUILD_LSF_JOB_DIR must be absolute")

# Reject symlinks in the caller spelling before canonicalization. The verified
# result and evidence remain owner-only local inputs; Harbor credentials are
# acquired only after this validation succeeds.
current = Path("/")
for component in spelled.parts[1:]:
    if component in ("", "."):
        continue
    if component == "..":
        current = current.parent
        continue
    current = current / component
    try:
        value = current.lstat()
    except OSError as exc:
        raise SystemExit(f"IMAGE_BUILD_LSF_JOB_DIR component is unavailable: {current}: {exc}")
    if stat.S_ISLNK(value.st_mode):
        raise SystemExit(f"IMAGE_BUILD_LSF_JOB_DIR must not contain symlinks: {current}")

job = spelled.resolve(strict=True)
output = job / "output"
evidence_path = output / "success.json"
archive = output / "result.oci.tar"
euid = os.geteuid()
for path, kind in ((job, "directory"), (output, "directory"), (evidence_path, "file"), (archive, "file")):
    value = path.lstat()
    if value.st_uid != euid or stat.S_IMODE(value.st_mode) & 0o022:
        raise SystemExit(f"image-build-lsf {path.name} must be EUID-owned and not group/world writable")
    if kind == "directory" and not stat.S_ISDIR(value.st_mode):
        raise SystemExit(f"image-build-lsf {path.name} must be a directory")
    if kind == "file" and not stat.S_ISREG(value.st_mode):
        raise SystemExit(f"image-build-lsf {path.name} must be a regular file")

try:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"image-build-lsf success evidence is invalid: {exc}")
digest_pattern = re.compile(r"^sha256:[0-9a-f]{64}$")
job_id = evidence.get("lsf_job_id")
artifact_digest = evidence.get("artifact_digest")
manifest_digest = evidence.get("manifest_digest")
if (
    evidence.get("schema_version") != 1
    or evidence.get("profile") != "slai-web-app-v1"
    or evidence.get("platform") != "linux/amd64"
    or evidence.get("probe_status") != 200
    or not isinstance(job_id, str)
    or not re.fullmatch(r"[0-9]{1,20}", job_id)
    or not isinstance(artifact_digest, str)
    or not digest_pattern.fullmatch(artifact_digest)
    or not isinstance(manifest_digest, str)
    or not digest_pattern.fullmatch(manifest_digest)
    or not isinstance(evidence.get("artifact_bytes"), int)
    or evidence["artifact_bytes"] < 1
):
    raise SystemExit("image-build-lsf success evidence does not satisfy the SLAI web-app contract")
if archive.stat().st_size != evidence["artifact_bytes"]:
    raise SystemExit("image-build-lsf archive length does not match success evidence")
transport_hasher = hashlib.sha256()
with archive.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        transport_hasher.update(chunk)
observed_transport = "sha256:" + transport_hasher.hexdigest()
if observed_transport != artifact_digest:
    raise SystemExit("image-build-lsf archive digest does not match success evidence")

with tarfile.open(archive, "r") as bundle:
    index_file = bundle.extractfile(bundle.getmember("index.json"))
    if index_file is None:
        raise SystemExit("image-build-lsf OCI index is unreadable")
    index = json.load(index_file)
    manifests = index.get("manifests", []) if isinstance(index, dict) else []
    if not isinstance(manifests, list) or len(manifests) != 1:
        raise SystemExit("image-build-lsf result must contain exactly one OCI manifest")
    descriptor = manifests[0]
    if not isinstance(descriptor, dict) or descriptor.get("digest") != manifest_digest:
        raise SystemExit("image-build-lsf manifest digest does not match success evidence")
    manifest_file = bundle.extractfile(
        bundle.getmember("blobs/sha256/" + manifest_digest.split(":", 1)[1])
    )
    if manifest_file is None:
        raise SystemExit("image-build-lsf OCI manifest is unreadable")
    manifest = manifest_file.read()
if descriptor.get("size") != len(manifest):
    raise SystemExit("image-build-lsf OCI manifest length does not match its descriptor")
if "sha256:" + hashlib.sha256(manifest).hexdigest() != manifest_digest:
    raise SystemExit("image-build-lsf OCI manifest bytes do not match their digest")

print(archive)
print(f"localhost/oci-build-lsf:{job_id}")
print(manifest_digest)
PY
  )
  if [[ ${#lsf_result[@]} -ne 3 ]]; then
    echo "image-build-lsf result validation did not return the expected fields." >&2
    exit 1
  fi
  LSF_ARCHIVE="${lsf_result[0]}"
  LSF_SOURCE_IMAGE="${lsf_result[1]}"
  LSF_MANIFEST_DIGEST="${lsf_result[2]}"
fi

resolve_harbor_cli() {
  if [[ -n "${HARBOR_CLI_BIN:-}" && -x "$HARBOR_CLI_BIN" ]]; then
    echo "$HARBOR_CLI_BIN"
    return 0
  fi
  if [[ -n "${HARBOR_CLI_INSTALL_DIR:-}" && -x "$HARBOR_CLI_INSTALL_DIR/harbor" ]]; then
    echo "$HARBOR_CLI_INSTALL_DIR/harbor"
    return 0
  fi
  if [[ -n "${HARBOR_CLI_INSTALL_DIR:-}" && -x "$HARBOR_CLI_INSTALL_DIR/harbor.exe" ]]; then
    echo "$HARBOR_CLI_INSTALL_DIR/harbor.exe"
    return 0
  fi
  if command -v harbor >/dev/null 2>&1; then
    command -v harbor
    return 0
  fi
  if [[ -x "$HOME/.local/bin/harbor" ]]; then
    echo "$HOME/.local/bin/harbor"
    return 0
  fi
  if [[ -n "${USERPROFILE:-}" && -x "$USERPROFILE/.local/bin/harbor.exe" ]]; then
    echo "$USERPROFILE/.local/bin/harbor.exe"
    return 0
  fi
  return 1
}

install_harbor_cli() {
  local base="${HARBOR_CLI_BASE_URL:-https://atlartifactory.amd.com:8443/artifactory/SW-SLAI-PROD-LOCAL/harbor-cli}"
  local installer status
  echo "Harbor CLI not found; installing from ${base}/install.sh" >&2
  umask 077
  installer="$(mktemp /tmp/harbor-cli-install.XXXXXX)" || return 1
  if ! curl -fsSL -o "$installer" "${base}/install.sh"; then
    rm -f -- "$installer"
    return 1
  fi
  sh "$installer"
  status=$?
  rm -f -- "$installer"
  return "$status"
}

load_harbor_cli_credentials() {
  local harbor_bin="$1"
  local cred_file="${HARBOR_CONFIG_DIR:-$HOME/.config/harbor}/credentials"
  "$harbor_bin" auth login "$HARBOR_PROJECT"
  if [[ ! -f "$cred_file" ]]; then
    echo "Harbor credentials file was not created: $cred_file" >&2
    return 1
  fi
  local parsed
  parsed="$(python12 - "$cred_file" "$HARBOR_PROJECT" "$HARBOR_REGISTRY" <<'PY'
import configparser
import sys
from pathlib import Path

path = Path(sys.argv[1])
project = sys.argv[2]
expected_registry = sys.argv[3]
section = f"project.{project}"
cfg = configparser.RawConfigParser()
cfg.read(path)
if section not in cfg:
    raise SystemExit(f"missing [{section}] in {path}")
robot_name = cfg[section].get("robot_name", "").strip()
robot_secret = cfg[section].get("robot_secret", "").strip()
registry = cfg[section].get("harbor_registry", "").strip()
if not robot_name or not robot_secret:
    raise SystemExit(f"missing robot_name or robot_secret in [{section}]")
if not registry:
    raise SystemExit(f"missing harbor_registry in [{section}]; refusing to forward unbound credentials")
if registry != expected_registry:
    raise SystemExit("credential registry does not match HARBOR_REGISTRY; refusing login")
print(robot_name)
print(robot_secret)
PY
)"
  HARBOR_USERNAME="$(printf '%s\n' "$parsed" | sed -n '1p')"
  HARBOR_PASSWORD="$(printf '%s\n' "$parsed" | sed -n '2p')"
  export HARBOR_USERNAME HARBOR_PASSWORD
}

if [[ -z "${HARBOR_USERNAME:-}" || -z "${HARBOR_PASSWORD:-}" ]]; then
  HARBOR_BIN="$(resolve_harbor_cli || true)"
  if [[ -z "$HARBOR_BIN" ]]; then
    install_harbor_cli
    HARBOR_BIN="$(resolve_harbor_cli)" || {
      echo "Harbor CLI install completed but harbor is still not on PATH or at ~/.local/bin/harbor." >&2
      exit 1
    }
  fi
  load_harbor_cli_credentials "$HARBOR_BIN"
else
  SELECTED_LOCATION="${HARBOR_REGISTRY}/${HARBOR_PROJECT}"
  BOUND_LOCATION="${HARBOR_CREDENTIAL_LOCATION:-mkmhub.amd.com/hw-slaiapp-dev}"
  if [[ "$SELECTED_LOCATION" != "$BOUND_LOCATION" ]]; then
    echo "HARBOR_USERNAME/HARBOR_PASSWORD are not bound to the selected registry/project; refusing login." >&2
    echo "Set HARBOR_CREDENTIAL_LOCATION to the selected hostname/project associated with those credentials." >&2
    exit 1
  fi
fi

if [[ -z "${IMAGE_TAG:-}" ]]; then
  if [[ -d "$ROOT/.git" ]]; then
    IMAGE_TAG="$(git -C "$ROOT" rev-parse HEAD)"
  else
    IMAGE_TAG="local-$(date -u +%Y%m%d%H%M%S)"
  fi
fi
if [[ -z "$LSF_ARCHIVE" ]]; then
  DOCKERFILE="${DOCKERFILE:-$BUILD_CONTEXT/Dockerfile}"
fi
FULL_IMAGE="${HARBOR_REGISTRY}/${HARBOR_PROJECT}/${IMAGE_NAME}:${IMAGE_TAG}"

write_harbor_last_image_env() {
  local f="${HARBOR_LAST_IMAGE_FILE:-$ROOT/.cache/harbor-last-image.env}"
  mkdir -p "$(dirname "$f")"
  umask 077
  {
    echo "# Generated by publish-image-harbor.sh after a successful push."
    echo "# Use FULL_IMAGE for deployment.yaml spec.template.spec.containers[].image"
    echo "FULL_IMAGE=$FULL_IMAGE"
    echo "IMAGE_TAG=$IMAGE_TAG"
  } > "$f"
  echo "Recorded $f"
}

if [[ -z "$LSF_ARCHIVE" && ! -f "$DOCKERFILE" ]]; then
  echo "Dockerfile not found: $DOCKERFILE" >&2
  exit 1
fi

use_docker() {
  [[ -z "${FORCE_PODMAN:-}" ]] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

if [[ -z "$LSF_ARCHIVE" ]] && use_docker; then
  echo "Using docker → ${FULL_IMAGE}"
  echo "$HARBOR_PASSWORD" | docker login "$HARBOR_REGISTRY" --username "$HARBOR_USERNAME" --password-stdin
  docker build --platform linux/amd64 -f "$DOCKERFILE" -t "$FULL_IMAGE" "$BUILD_CONTEXT"
  docker push "$FULL_IMAGE"
  write_harbor_last_image_env
  echo "Pushed ${FULL_IMAGE}"
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
  echo "Podman is required to load an image-build-lsf OCI archive; otherwise install working Docker or Podman for a local fallback build." >&2
  exit 1
}

RUNC=$(resolve_runc "$PODMAN_BIN" || true)

canonical_secure_parent() {
  local label="$1"
  local raw="$2"
  python12 - "$label" "$raw" <<'PY'
import os
import stat
import sys

label, raw = sys.argv[1:]
euid = os.geteuid()

def fail(message: str) -> None:
    raise SystemExit(f"{label}: {message}")

if not raw:
    fail("path must not be empty")
if raw.startswith("-"):
    fail("leading-dash paths are not allowed")
if "\n" in raw or "\r" in raw:
    fail("line breaks are not allowed in paths")

# Walk the caller-supplied spelling with lstat before realpath so a symlink in
# any component is rejected instead of silently followed.
current = os.path.sep if os.path.isabs(raw) else os.getcwd()
for component in raw.split(os.path.sep):
    if component in ("", "."):
        continue
    if component == "..":
        current = os.path.dirname(current)
        continue
    current = os.path.join(current, component)
    try:
        component_stat = os.lstat(current)
    except OSError as exc:
        fail(f"must be an existing directory ({exc.strerror}: {current})")
    if stat.S_ISLNK(component_stat.st_mode):
        fail(f"must not contain symlink components ({current})")

candidate = os.path.abspath(raw)
real = os.path.realpath(candidate)
if real != candidate:
    fail("must resolve without symlink or alias substitution")
if not os.path.isdir(real):
    fail("must be an existing directory")

def mode_bits(value: os.stat_result) -> int:
    return stat.S_IMODE(value.st_mode)

def root_owned_sticky(value: os.stat_result) -> bool:
    return value.st_uid == 0 and bool(value.st_mode & stat.S_ISVTX)

final_stat = os.stat(real)
final_private = final_stat.st_uid == euid and not (mode_bits(final_stat) & 0o022)
if not (final_private or root_owned_sticky(final_stat)):
    fail(
        "final directory must be EUID-owned without group/world write, "
        "or a root-owned sticky directory"
    )

ancestor = os.path.dirname(real)
while True:
    ancestor_stat = os.stat(ancestor)
    if ancestor_stat.st_uid not in (0, euid):
        fail(f"ancestor must be root/EUID-owned ({ancestor})")
    if mode_bits(ancestor_stat) & 0o022 and not root_owned_sticky(ancestor_stat):
        fail(f"ancestor must not be group/world writable ({ancestor})")
    parent = os.path.dirname(ancestor)
    if parent == ancestor:
        break
    ancestor = parent

print(real)
PY
}

PODMAN_TEMP_PARENT="$(canonical_secure_parent PODMAN_TEMP_PARENT "${PODMAN_TEMP_PARENT:-/tmp}")"
RUNROOT_PARENT_INPUT="${PODMAN_RUNROOT_PARENT:-/tmp}"
PODMAN_RUNROOT_PARENT="$(canonical_secure_parent PODMAN_RUNROOT_PARENT "$RUNROOT_PARENT_INPUT")"

# Podman rejects runroot paths longer than 50 characters. Keep large image
# storage and REGISTRY_AUTH_FILE under PODMAN_TEMP_PARENT, but allocate the
# small runtime directory from an independently validated short parent.
if (( ${#PODMAN_RUNROOT_PARENT} + 9 > 50 )); then
  echo "PODMAN_RUNROOT_PARENT is too long: runroot would exceed Podman's 50-character limit." >&2
  echo "Choose a short existing secure directory; shared ETX hosts should use /tmp." >&2
  exit 1
fi

cleanup_direct_child() {
  local child="$1"
  local parent="$2"
  local prefix="$3"
  [[ -n "$child" && -n "$parent" ]] || return 0
  [[ "$(dirname -- "$child")" == "$parent" ]] || return 0
  case "$(basename -- "$child")" in
    "$prefix"??????) rm -rf -- "$child" 2>/dev/null || true ;;
  esac
}

BASE=""
RUNROOT=""
cleanup_podman_storage() {
  cleanup_direct_child "${RUNROOT:-}" "$PODMAN_RUNROOT_PARENT" pr
  cleanup_direct_child "${BASE:-}" "$PODMAN_TEMP_PARENT" pm
}
trap cleanup_podman_storage EXIT

umask 077
BASE=$(mktemp -d -- "$PODMAN_TEMP_PARENT/pmXXXXXX")
RUNROOT=$(mktemp -d -- "$PODMAN_RUNROOT_PARENT/prXXXXXX")
ROOTPM="$BASE/rt"
mkdir -p -- "$ROOTPM"
export REGISTRY_AUTH_FILE="$BASE/auth.json"

pm=( "$PODMAN_BIN" --root "$ROOTPM" --runroot "$RUNROOT" )
[[ -n "$RUNC" ]] && pm+=( --runtime "$RUNC" )

echo "Using podman → ${FULL_IMAGE}"
echo "$HARBOR_PASSWORD" | "${pm[@]}" login "$HARBOR_REGISTRY" \
  --username "$HARBOR_USERNAME" --password-stdin

if [[ -n "$LSF_ARCHIVE" ]]; then
  echo "Loading verified image-build-lsf result ${LSF_MANIFEST_DIGEST}"
  "${pm[@]}" load --input "$LSF_ARCHIVE" >/dev/null
  "${pm[@]}" image exists "$LSF_SOURCE_IMAGE" || {
    echo "image-build-lsf archive did not restore expected image ${LSF_SOURCE_IMAGE}." >&2
    exit 1
  }
  "${pm[@]}" tag "$LSF_SOURCE_IMAGE" "$FULL_IMAGE"
else
  "${pm[@]}" build \
    --platform linux/amd64 \
    --storage-opt overlay.ignore_chown_errors=true \
    -f "$DOCKERFILE" \
    -t "$FULL_IMAGE" \
    "$BUILD_CONTEXT"
fi

"${pm[@]}" push "$FULL_IMAGE"
write_harbor_last_image_env
echo "Pushed ${FULL_IMAGE}"
