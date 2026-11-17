#!/usr/bin/env bash
# See BUILDING.md for this command and its options.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec "$root/scripts/workbench.sh" clean "$@"
