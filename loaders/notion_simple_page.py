from __future__ import annotations
from typing import List

from notion_client import Client
from langchain.schema import Document
from langchain.document_loaders.base import BaseLoader


class NotionSimplePageLoader(BaseLoader):
    def __init__(self, integration_token: str, page_id: str, timeout: int = 30):
        self.client = Client(auth=integration_token)
        self.page_id = page_id
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    # Public API required by LangChain                                   #
    # ------------------------------------------------------------------ #
    def load(self) -> List[Document]:
        text = self._extract_page_text(self.page_id)
        return [Document(page_content=text, metadata={"page_id": self.page_id})]

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _extract_page_text(self, root_id: str) -> str:
        """Depth‑first walk over every block, collecting plain text."""
        stack = [root_id]
        lines: List[str] = []

        while stack:
            block_id = stack.pop()
            resp = self.client.blocks.children.list(block_id, page_size=100)

            for blk in resp["results"]:
                txt = self._block_to_text(blk)
                if txt:
                    lines.append(txt)

                if blk.get("has_children"):
                    stack.append(blk["id"])

            if resp.get("has_more"):
                stack.append(resp["next_cursor"])

        return "\n".join(lines)

    @staticmethod
    def _block_to_text(blk) -> str | None:
        """Return concatenated plain_text for any block that has rich_text."""
        t = blk["type"]
        data = blk[t]
        rich = data.get("rich_text") or data.get("text")
        if not rich:
            return None
        return "".join(span["plain_text"] for span in rich)