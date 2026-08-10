"""Prints the dim_contacts table to the terminal — who has called, how many
times, first/last seen. Run anytime with:

    python scripts/view_callers.py
"""

import asyncio
import sys
from pathlib import Path

# Run as a plain script (not -m), so the project root — where the
# bot_processors package lives — isn't on sys.path by default the way it
# is for python -m scripts.view_callers; add it explicitly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from bot_processors.core.db_pool import close_pool, init_pool, get_pool


async def main() -> None:
    await init_pool()
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT phone_number, name, call_count, first_seen, last_seen, "
            "confirmed_state, confirmed_district, confirmed_mandal, confirmed_village, confirmed_pincode "
            "FROM dim_contacts ORDER BY last_seen DESC"
        )
    await close_pool()

    if not rows:
        print("No callers recorded yet.")
        return

    header = (
        f"{'Name':<15} {'Phone Number':<15} {'First Seen':<20} {'Last Seen':<20} {'Calls':>5}  "
        f"{'State':<15} {'District':<15} {'Mandal':<15} {'Village':<15} {'Pincode':<8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{(row['name'] or '-'):<15} {row['phone_number']:<15} "
            f"{str(row['first_seen'])[:19]:<20} {str(row['last_seen'])[:19]:<20} {row['call_count']:>5}  "
            f"{(row['confirmed_state'] or '-'):<15} {(row['confirmed_district'] or '-'):<15} "
            f"{(row['confirmed_mandal'] or '-'):<15} {(row['confirmed_village'] or '-'):<15} "
            f"{(row['confirmed_pincode'] or '-'):<8}"
        )
    print(f"\n{len(rows)} caller(s) total.")


if __name__ == "__main__":
    asyncio.run(main())
