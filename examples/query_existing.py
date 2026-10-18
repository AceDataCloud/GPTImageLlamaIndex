"""Query a previously submitted task once, without generating."""

import argparse
import asyncio
import json

from llama_index.tools.gpt_image_acedatacloud import GPTImageToolSpec


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id")
    args = parser.parse_args()
    spec = GPTImageToolSpec()
    try:
        print(json.dumps(await spec.agpt_image_get_task(args.task_id), indent=2))
    finally:
        await spec.aclose()


if __name__ == "__main__":
    asyncio.run(main())
