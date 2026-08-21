#!/bin/bash
# One-off runner: backfills 08/09/10-08-2026 then redoes today (11-08-2026),
# retrying each date with --resume (skips crop_keywords already dated that
# day in last_known_prices.json) whenever a run dies early — the site itself
# has been flaky today (transient form-reload failures cascading into an
# abort after enough of them in one browser session), independent of the
# scraper code, so a bare single run per date isn't reliable enough to trust
# unattended. Not meant to be kept around — delete after this backfill.
cd "C:\Users\Praneeth\Desktop\Agent\pipecat" || exit 1
PY="./venv/Scripts/python.exe"
MAX_RETRIES=6

for d in 2026-08-08 2026-08-09 2026-08-10 2026-08-11; do
    echo "===== BACKFILL START: $d ====="
    attempt=1
    while [ $attempt -le $MAX_RETRIES ]; do
        echo "--- $d attempt $attempt/$MAX_RETRIES ---"
        if [ $attempt -eq 1 ]; then
            "$PY" -m bot_processors.pricing.agmarknet_scraper --date="$d"
        else
            "$PY" -m bot_processors.pricing.agmarknet_scraper --date="$d" --resume
        fi
        status=$?
        if [ $status -eq 0 ]; then
            echo "--- $d succeeded on attempt $attempt ---"
            break
        fi
        echo "--- $d attempt $attempt failed (exit $status), retrying ---"
        attempt=$((attempt + 1))
    done
    echo "===== BACKFILL DONE: $d ====="
done
echo "===== ALL BACKFILLS COMPLETE ====="
