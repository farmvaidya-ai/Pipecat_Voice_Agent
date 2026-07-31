"""Prints the dim_contacts table to the terminal — who has called, how many
times, first/last seen. Run anytime with:

    python view_callers.py
"""

import asyncio

from dotenv import load_dotenv

load_dotenv(override=True)

from bot_processors.db_pool import close_pool, init_pool, get_pool


async def main() -> None:
    await init_pool()
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT phone_number, call_count, first_seen, last_seen "
            "FROM dim_contacts ORDER BY last_seen DESC"
        )
    await close_pool()

    if not rows:
        print("No callers recorded yet.")
        return

    print(f"{'Phone Number':<15} {'Calls':>5}  {'First Seen':<25} {'Last Seen':<25}")
    print("-" * 75)
    for row in rows:
        print(f"{row['phone_number']:<15} {row['call_count']:>5}  {str(row['first_seen']):<25} {str(row['last_seen']):<25}")
    print(f"\n{len(rows)} caller(s) total.")


if __name__ == "__main__":
    asyncio.run(main())
