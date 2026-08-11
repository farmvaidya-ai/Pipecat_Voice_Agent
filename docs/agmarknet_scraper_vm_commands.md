# Agmarknet Scraper VM — Command Reference

Quick-reference cheat sheet for operating the scraper on the VM
(`sethu-admin1`, Azure, `104.211.91.182`) day to day. For the full
deployment story (what was built and why), see
`agmarknet_scraper_vm_deployment.md` / `.pdf` in this same folder.

## Connecting

```bash
ssh -i "sethu-admin1_key.pem" azureuser@104.211.91.182
```

## Checking scraper status

**Is it running right now?**
```bash
sudo systemctl status agmarknet-scraper
```
Look for `Active: active (running)`.

**Watch it live** (updates in real time; Ctrl+C to stop watching — this
does *not* stop the scraper itself):
```bash
sudo journalctl -u agmarknet-scraper -f
```

**Just the progress count** (how many commodities done in the current
cycle):
```bash
sudo journalctl -u agmarknet-scraper --no-pager | grep "commodities done" | tail -10
```

**Check for errors/failures:**
```bash
sudo journalctl -u agmarknet-scraper --no-pager | grep -iE "error|failed|traceback" | tail -20
```

**Confirm a cycle actually saved to the database** (look for the
"upserted" line — this only appears once a full cycle finishes):
```bash
sudo journalctl -u agmarknet-scraper --no-pager | grep "upserted" | tail -5
```

## Controlling the service

**Restart it manually:**
```bash
sudo systemctl restart agmarknet-scraper
```

**Stop it** (won't restart until you start it again, even after a reboot):
```bash
sudo systemctl stop agmarknet-scraper
```

**Start it again:**
```bash
sudo systemctl start agmarknet-scraper
```

**Disable it from auto-starting on boot** (rarely needed — it's meant to
stay enabled):
```bash
sudo systemctl disable agmarknet-scraper
```

## Manually backfilling a specific past date

Runs once and exits — not the always-on daemon. Useful if you notice a
specific day is missing and don't want to wait for the daemon's own
automatic 3-day gap-check:
```bash
cd ~/projects/pipecat
source venv/bin/activate
python -m bot_processors.pricing.agmarknet_scraper --date=2026-08-09 --resume
```
`--resume` skips any commodity that already has data for that date — safe
to re-run if it fails partway.

## Checking resource usage

**Memory/CPU used by the service:**
```bash
sudo systemctl status agmarknet-scraper --no-pager | grep -E "Memory|CPU"
```

**Overall VM memory/disk:**
```bash
free -h
df -h /
```

## Updating the code

The VM pulls from GitHub via a read-only deploy key — it can fetch updates
but never push:
```bash
cd ~/projects/pipecat
git pull
sudo systemctl restart agmarknet-scraper
```

## Checking the actual scraped data

The VM itself doesn't store the data — every row gets written straight to
Postgres on the dev Windows PC over the Tailscale tunnel. To see the real
data (not just progress logs), use pgAdmin or a query **on the Windows
side**, not the VM:
```sql
-- one row per day, sorted, with a count
SELECT arrival_date, count(*) FROM fact_market_prices
GROUP BY arrival_date ORDER BY arrival_date;

-- every row, pre-sorted (the fact_market_prices_ordered view)
SELECT * FROM fact_market_prices_ordered;
```

## Tailscale (networking)

**Check the tunnel is up:**
```bash
sudo tailscale status
```

**Check connectivity to the Windows PC specifically:**
```bash
sudo tailscale ping -c 2 100.109.45.19
```
