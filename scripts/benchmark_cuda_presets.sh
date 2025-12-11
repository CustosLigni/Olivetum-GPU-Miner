#!/usr/bin/env bash
#
# Quick CUDA benchmark matrix for ethminer.
# Usage: ./scripts/benchmark_cuda_presets.sh [path_to_ethminer_binary]
# Env overrides:
#   BLOCKS="64 128 256" GRIDS="2048 4096 8192" STREAMS="1 2" RUNTIME=25
# Each run is limited by timeout RUNTIME (seconds) and parses "Mean X Mh".

set -uo pipefail

BIN="${1:-./build/ethminer/ethminer}"
BLOCKS=${BLOCKS:-"64 128 256"}
GRIDS=${GRIDS:-"2048 4096 8192"}
STREAMS=${STREAMS:-"1 2 3"}
RUNTIME=${RUNTIME:-25}

if [[ ! -x "$BIN" ]]; then
    echo "Binary not executable: $BIN"
    exit 1
fi

echo "Benchmarking CUDA presets with $BIN"
echo "Blocks: $BLOCKS"
echo "Grids : $GRIDS"
echo "Streams: $STREAMS"
echo "Timeout per run: ${RUNTIME}s"
echo ""
echo "block;grid;streams;hashrate_MH_s;status"

for b in $BLOCKS; do
    for g in $GRIDS; do
        for s in $STREAMS; do
            cmd=( timeout "$RUNTIME" "$BIN"
                -U
                -M 0
                --cu-block-size "$b"
                --cu-grid-size "$g"
                --cu-streams "$s"
            )

            out=""
            if ! out="$("${cmd[@]}" 2>&1)"; then
                echo "$b;$g;$s;n/a;fail"
                continue
            fi

            rate=$(echo "$out" | awk '
                /Simulation results/ {
                    for (i = 1; i <= NF; ++i) {
                        if ($i == "Mean") {
                            if ((i+1) <= NF) {
                                gsub(/Mh/, "", $(i+1));
                                print $(i+1);
                                exit;
                            }
                        }
                    }
                }
            ')

            if [[ -z "$rate" ]]; then
                rate="n/a"
                status="no_parse"
            else
                status="ok"
            fi

            echo "$b;$g;$s;$rate;$status"
        done
    done
done
