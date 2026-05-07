#!/bin/bash
# Acquire and unpack the EuRoC V1_01_easy sequence (Vicon Room 1, easy).
#
# Acquisition paths, tried in order:
#   1. Per-sequence zip already at data/raw/V1_01_easy.zip   (~150 MB; preferred)
#   2. Vicon Room 1 *bundle* zip from the ETH Research Collection at
#      data/raw/<anything containing "Vicon" or "vicon_room1">.zip  (~5.7 GB)
#      The bundle nests V1_01_easy.zip inside; we extract just that.
#   3. Legacy ASL HTTP mirror (auto-download, may be flaky).
#
# Manual-download sources (use either):
#   a) Legacy ASL page (per-sequence, smallest):
#        https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
#      Save the V1_01_easy zip to:  data/raw/V1_01_easy.zip
#
#   b) ETH Research Collection (Vicon Room 1 bundle, ~5.7 GB):
#        https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f
#      Save the "Vicon Room 1 Datasets" zip to: data/raw/  (any filename matching *Vicon*Room*1*.zip)
#
# Idempotent: if mav0/ is already extracted, exits immediately.

set -euo pipefail

SEQUENCE="V1_01_easy"
LEGACY_URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/${SEQUENCE}/${SEQUENCE}.zip"
RC_URL="https://www.research-collection.ethz.ch/entities/researchdata/bcaf173e-5dac-484b-bc37-faf97a594f1f"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/raw"
SEQ_ZIP="${RAW_DIR}/${SEQUENCE}.zip"
OUT_DIR="${RAW_DIR}/${SEQUENCE}"

mkdir -p "${RAW_DIR}"

# 0. Already extracted? Accept either layout:
#    a) data/raw/mav0/...                    (mav0 dropped directly into raw/)
#    b) data/raw/V1_01_easy/mav0/...         (zip-extracted layout)
if [ -f "${RAW_DIR}/mav0/imu0/data.csv" ]; then
    echo "[fetch_euroc] mav0/ found directly at ${RAW_DIR} -- nothing to do"
    exit 0
fi
if [ -f "${OUT_DIR}/mav0/imu0/data.csv" ]; then
    echo "[fetch_euroc] ${SEQUENCE} already extracted at ${OUT_DIR} -- nothing to do"
    exit 0
fi

# 1. Per-sequence zip already on disk?
if [ -f "${SEQ_ZIP}" ]; then
    echo "[fetch_euroc] using per-sequence zip ${SEQ_ZIP}"

# 2. Bundle zip on disk? (Research Collection)
else
    BUNDLE_ZIP="$(find "${RAW_DIR}" -maxdepth 1 -type f -iname '*vicon*room*1*.zip' -print -quit 2>/dev/null || true)"
    if [ -n "${BUNDLE_ZIP:-}" ]; then
        echo "[fetch_euroc] found Research-Collection bundle ${BUNDLE_ZIP}"
        echo "[fetch_euroc] extracting only ${SEQUENCE}.zip from the bundle"
        # The bundle layout typically nests one zip per sequence at the top level
        # (sometimes under a directory). Try to extract just our target.
        TMP_DIR="$(mktemp -d -p "${RAW_DIR}" bundle_extract.XXXXXX)"
        trap 'rm -rf "${TMP_DIR}"' EXIT
        unzip -q -j "${BUNDLE_ZIP}" "*${SEQUENCE}.zip" -d "${TMP_DIR}" 2>/dev/null || {
            echo "[fetch_euroc] could not find ${SEQUENCE}.zip inside ${BUNDLE_ZIP}" >&2
            echo "[fetch_euroc] try: unzip -l '${BUNDLE_ZIP}' | grep ${SEQUENCE}" >&2
            exit 1
        }
        INNER_ZIP="$(find "${TMP_DIR}" -maxdepth 1 -name "*${SEQUENCE}.zip" -print -quit)"
        if [ -z "${INNER_ZIP}" ]; then
            echo "[fetch_euroc] inner ${SEQUENCE}.zip not found after bundle extract" >&2
            exit 1
        fi
        mv "${INNER_ZIP}" "${SEQ_ZIP}"
        rm -rf "${TMP_DIR}"
        trap - EXIT
        echo "[fetch_euroc] saved per-sequence zip to ${SEQ_ZIP}"

    # 3. Try the legacy auto-download.
    else
        echo "[fetch_euroc] no zip on disk; attempting legacy mirror"
        echo "[fetch_euroc]   ${LEGACY_URL}"
        DL_OK=0
        if command -v curl >/dev/null 2>&1; then
            if curl -L --fail --connect-timeout 15 --retry 2 -o "${SEQ_ZIP}" "${LEGACY_URL}"; then
                DL_OK=1
            fi
        elif command -v wget >/dev/null 2>&1; then
            if wget --timeout=15 --tries=2 -O "${SEQ_ZIP}" "${LEGACY_URL}"; then
                DL_OK=1
            fi
        fi
        if [ "${DL_OK}" -ne 1 ]; then
            rm -f "${SEQ_ZIP}"
            cat <<EOF >&2

[fetch_euroc] automatic download failed.

Manually download V1_01_easy from one of:

  (a) Legacy ASL page (per-sequence, ~150 MB) -- preferred:
      https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
      Save the resulting zip to: ${SEQ_ZIP}

  (b) ETH Research Collection (Vicon Room 1 bundle, ~5.7 GB):
      ${RC_URL}
      Save the "Vicon Room 1 Datasets" zip into: ${RAW_DIR}/
      (filename can stay as-downloaded; this script will detect *vicon*room*1*.zip)

Then re-run: bash data/fetch_euroc.sh
EOF
            exit 1
        fi
    fi
fi

# Extract the per-sequence zip.
echo "[fetch_euroc] extracting ${SEQ_ZIP} to ${OUT_DIR}"
mkdir -p "${OUT_DIR}"
unzip -q -o "${SEQ_ZIP}" -d "${OUT_DIR}"

# Some EuRoC zips wrap their contents in an extra V1_01_easy/ directory; flatten if so.
if [ ! -f "${OUT_DIR}/mav0/imu0/data.csv" ] && [ -f "${OUT_DIR}/${SEQUENCE}/mav0/imu0/data.csv" ]; then
    echo "[fetch_euroc] flattening nested ${SEQUENCE}/ directory"
    shopt -s dotglob
    mv "${OUT_DIR}/${SEQUENCE}"/* "${OUT_DIR}/"
    shopt -u dotglob
    rmdir "${OUT_DIR}/${SEQUENCE}"
fi

if [ ! -f "${OUT_DIR}/mav0/imu0/data.csv" ]; then
    echo "[fetch_euroc] extraction looks wrong: ${OUT_DIR}/mav0/imu0/data.csv missing" >&2
    echo "[fetch_euroc] inspect with: ls -R ${OUT_DIR}" >&2
    exit 1
fi

echo "[fetch_euroc] done."
echo "  IMU csv : ${OUT_DIR}/mav0/imu0/data.csv"
echo "  GT  csv : ${OUT_DIR}/mav0/state_groundtruth_estimate0/data.csv"
echo "Next: python data/preprocess.py"
