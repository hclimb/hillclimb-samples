#!/usr/bin/env bash
# Offline pipeline: optionally build local DBs, then run dbpedia_gen.py
#
# Usage:
#   ./scripts/data/gen_multihop_offline.sh --hf-repo user/multihop-qa
#   ./scripts/data/gen_multihop_offline.sh --build --hf-repo user/multihop-qa
#   ./scripts/data/gen_multihop_offline.sh --num 500 --hops 4 --paths-per-node 3 --hf-repo user/multihop-qa

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate venv
source "$ROOT/.venv/bin/activate"

# Load .env
if [[ -f "$ROOT/.env" ]]; then
    set -a
    source "$ROOT/.env"
    set +a
fi
DATA_DIR="${DATA_DIR:-$ROOT/data}"

BUILD=0
NUM=1000000
HOPS=3
PATHS_PER_NODE=1
PARQUET_ROWS=10000
HF_REPO="ragrawal36/multihop_qa"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)          BUILD=1; shift ;;
        --num)            NUM="$2"; shift 2 ;;
        --hops)           HOPS="$2"; shift 2 ;;
        --paths-per-node) PATHS_PER_NODE="$2"; shift 2 ;;
        --parquet-rows)   PARQUET_ROWS="$2"; shift 2 ;;
        --hf-repo)        HF_REPO="$2"; shift 2 ;;
        --data-dir)       DATA_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

export DBPEDIA_DB="${DBPEDIA_DB:-$DATA_DIR/dbpedia.db}"
export WIKI_DB="${WIKI_DB:-/dev/shm/wiki.db}"

if [[ $BUILD -eq 1 ]]; then
    echo "=== Building DBpedia DB ==="
    python "$ROOT/datagen/multihop_qa/build_dbpedia_db.py" --data-dir "$DATA_DIR"

    echo ""
    echo "=== Building Wikipedia DB ==="
    python "$ROOT/datagen/multihop_qa/build_wiki_db.py" --data-dir "$DATA_DIR"
else
    if [[ ! -f "$DBPEDIA_DB" ]]; then
        echo "=== dbpedia.db not found at $DBPEDIA_DB, downloading from Google Drive ==="
        gdown "1_emft5757BWPVVMHXTJnULGsPPgY_1p3" -O "$DBPEDIA_DB"
    fi

    if [[ ! -f "$WIKI_DB" ]]; then
        echo "=== wiki.db not found at $WIKI_DB, downloading from GCS ==="
        if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
            echo "ERROR: Not logged into gcloud. Run 'gcloud auth login' first."
            exit 1
        fi
        BOTO_CONFIG=/tmp/gsutil_config/.boto
        mkdir -p /tmp/gsutil_config
        cat > "$BOTO_CONFIG" << 'EOF'
[GSUtil]
check_hashes = if_fast_else_skip
parallel_thread_count = 10
parallel_process_count = 8
sliced_object_download_threshold = 150M
sliced_object_download_max_components = 8
EOF
        export BOTO_CONFIG
        gsutil cp gs://memory-layers-outputs-suhas/wiki.db "$WIKI_DB"
    fi
fi

echo ""
echo "=== Generating multihop QA ==="
echo "  DBPEDIA_DB=$DBPEDIA_DB"
echo "  WIKI_DB=$WIKI_DB"
echo "  VLLM_BASE_URL=${VLLM_BASE_URL:-http://localhost:8001/v1}"
echo "  num=$NUM | hops=$HOPS | paths_per_node=$PATHS_PER_NODE | parquet_rows=$PARQUET_ROWS"
echo ""

[[ -z "$HF_REPO" ]] && { echo "ERROR: --hf-repo is required"; exit 1; }

cd "$ROOT/datagen/multihop_qa"
python dbpedia_gen.py \
    --hf-repo        "$HF_REPO" \
    --num            "$NUM" \
    --hops           "$HOPS" \
    --paths-per-node "$PATHS_PER_NODE" \
    --parquet-rows   "$PARQUET_ROWS"
