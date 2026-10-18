"""Actual native tools with synthetic API responses only; no real generation."""

import asyncio
import json

import respx

from llama_index.tools.gpt_image_acedatacloud import GPTImageToolSpec

CLASS = GPTImageToolSpec
GEN = "gpt_image_generate_image"
EDIT = "gpt_image_edit_image"
QUERY = "gpt_image_get_task"
ROUTE = "/openai/images/generations"
EDIT_ROUTE = "/openai/images/edits"
TASK = "/openai/tasks"
KEY = "url"
EXT = "png"
KWARGS = {
    "prompt": "Camera moves through a sunrise garden",
    "model": "gpt-image-1",
    "size": "1024x1024",
    "quality": "low",
    "count": 1,
}
EDIT_KWARGS = {
    "prompt": "Camera moves through a sunrise garden",
    "model": "gpt-image-1",
    "size": "1024x1024",
    "quality": "low",
    "count": 1,
    "image_urls": ["https://example.org/first.png"],
}
EXPECTED = {
    "async": True,
    "model": "gpt-image-1",
    "prompt": "Camera moves through a sunrise garden",
    "n": 1,
    "size": "1024x1024",
    "quality": "low",
    "response_format": "url",
}
REQUIRE_FINISHED = True


async def main():
    with respx.mock(assert_all_mocked=True) as router:
        router.post("https://api.acedata.cloud" + ROUTE).respond(200, json={"task_id": "synthetic-task"})
        router.post("https://api.acedata.cloud" + TASK).respond(
            200,
            json={
                "id": "synthetic-task",
                "finished_at": 123,
                "response": {
                    "success": True,
                    "data": [{KEY: "https://example.invalid/synthetic." + EXT, "state": "succeeded"}],
                },
            },
        )
        spec = CLASS(api_token="offline-dummy")
        try:
            generate, _, query = spec.to_tool_list()
            submission = generate.call(**KWARGS).raw_output
            recovered = CLASS(api_token="offline-dummy")
            try:
                result = (await recovered.to_tool_list()[2].acall(task_id=submission["task_id"])).raw_output
            finally:
                await recovered.aclose()
            assert result["status"] == "succeeded" and len(router.calls) == 2
            print(
                json.dumps(
                    {
                        "mode": "synthetic-offline",
                        "tools": [t.metadata.name for t in spec.to_tool_list()],
                        "submission": submission,
                        "recovered_result": result,
                    },
                    indent=2,
                )
            )
        finally:
            await spec.aclose()


if __name__ == "__main__":
    asyncio.run(main())
