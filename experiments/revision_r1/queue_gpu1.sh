#!/usr/bin/env bash
# GPU 1 queue: R1-3 mirrored runs whose guard probe matched the submitted run.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
"$HERE/run_job.sh" 1 r13_osflip_inception_v3 train.py --backbone inception_v3 --flip_os_images
