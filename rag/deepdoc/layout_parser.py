from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import pymupdf


class BlockType(str, Enum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    FOOTER = "footer"
    HEADER = "header"
    PAGE_BREAK = "page_break"
    NOISE = "noise"
    LIST_ITEM = "list_item"
    QUOTE = "quote"
    CODE = "code"
    MATH = "math"
    UNKNOWN = "unknown"


# ----------------------------------------------------------------------
# Data models
# ----------------------------------------------------------------------


@dataclass
class BBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid_y(self) -> float:
        return (self.y0 + self.y1) / 2

    def overlaps_vertically(self, other: BBox, threshold: float = 10.0) -> bool:
        return abs(self.centroid_y - other.centroid_y) < threshold


@dataclass
class Block:
    block_type: BlockType
    bbox: BBox
    text: str
    page: int
    order: int
    children: list["Block"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.block_type.value,
            "bbox": {
                "x0": round(self.bbox.x0, 2),
                "y0": round(self.bbox.y0, 2),
                "x1": round(self.bbox.x1, 2),
                "y1": round(self.bbox.y1, 2),
            },
            "text": self.text,
            "page": self.page,
            "order": self.order,
            "metadata": self.metadata,
        }


@dataclass
class PageLayout:
    page_num: int
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)
    rotation: int = 0

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "rotation": self.rotation,
            "blocks": [b.to_dict() for b in self.blocks],
        }


# ----------------------------------------------------------------------
# Detection heuristics
# ----------------------------------------------------------------------


# Font size thresholds (points) in PyMuPDF default scale
TITLE_SIZE_MIN = 18.0
HEADING_SIZE_MIN = 13.5
SMALL_SIZE_MAX = 9.0

# Common header/footer keywords
FOOTER_KEYWORDS = re.compile(
    r"^(page\s+\d+|©|\||-{5,}|\*{3,}|confidential|draft)",
    re.IGNORECASE,
)
HEADER_KEYWORDS = re.compile(
    r"^(chapter\s+\d|section\s+\d|\.{2,})",
    re.IGNORECASE,
)

# Table delimiters
TABLE_HINTS = re.compile(r"\|.*\|.*\|")


def _is_title_candidate(spans: list[dict], bbox: BBox) -> bool:
    if not spans:
        return False
    avg_size = sum(s["size"] for s in spans) / len(spans)
    return avg_size >= TITLE_SIZE_MIN


def _is_heading_candidate(spans: list[dict], bbox: BBox) -> bool:
    if not spans:
        return False
    avg_size = sum(s["size"] for s in spans) / len(spans)
    return HEADING_SIZE_MIN <= avg_size < TITLE_SIZE_MIN


def _is_noise(text: str, bbox: BBox) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    # Very short lines that appear at page edges (likely artifacts)
    if len(stripped) < 3:
        return True
    # Single characters repeated
    if len(stripped) == 1 and stripped in "|-_~^*":
        return True
    # Footers/headers
    if FOOTER_KEYWORDS.match(stripped) or HEADER_KEYWORDS.match(stripped):
        return True
    return False


def _classify_text_block(
    text: str,
    bbox: BBox,
    spans: list[dict],
    page_height: float,
    page_width: float,
) -> BlockType:
    if _is_noise(text, bbox):
        return BlockType.NOISE

    # Footer / header by position
    if bbox.y1 / page_height > 0.92:
        return BlockType.FOOTER
    if bbox.y0 / page_height < 0.08:
        return BlockType.HEADER

    # Title by font size
    if _is_title_candidate(spans, bbox):
        return BlockType.TITLE

    # Heading by font size
    if _is_heading_candidate(spans, bbox):
        return BlockType.HEADING

    # Table hint: text contains multiple pipe delimiters on same line
    if TABLE_HINTS.search(text):
        return BlockType.TABLE

    # Block-level list items
    stripped = text.strip()
    if re.match(r"^[\-\*\•]\s+", stripped) or re.match(r"^\d+[.)]\s+", stripped):
        return BlockType.LIST_ITEM

    # Quoted text
    if stripped.startswith('"') or stripped.startswith('"') or stripped.startswith('"'):
        return BlockType.QUOTE

    # Code blocks (indented, contains braces/brackets)
    if bbox.x0 > page_width * 0.05 and ("{" in text or "}" in text or "    " in text):
        return BlockType.CODE

    return BlockType.PARAGRAPH


# ----------------------------------------------------------------------
# Main layout parser
# ----------------------------------------------------------------------


def parse_pdf_layout(file_path: Path) -> list[PageLayout]:
    """
    Parse a PDF with layout awareness, returning structured pages and blocks.
    """
    layouts: list[PageLayout] = []

    with pymupdf.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            # Render at 72 DPI for block coordinates; use page.rect directly.
            page_rect = page.rect
            blocks_raw = page.get_text("dict")["blocks"]

            page_layout = PageLayout(
                page_num=page_num,
                width=page_rect.width,
                height=page_rect.height,
                rotation=page.rotation,
            )

            order = 0
            for raw in blocks_raw:
                # Skip empty blocks
                if raw.get("type") == 0 and not raw.get("lines"):
                    continue

                if raw["type"] == 0:
                    # Text block
                    bbox_data = raw["bbox"]
                    bbox = BBox(
                        x0=bbox_data[0],
                        y0=bbox_data[1],
                        x1=bbox_data[2],
                        y1=bbox_data[3],
                    )

                    lines = raw.get("lines", [])
                    block_text_parts: list[str] = []
                    all_spans: list[dict] = []

                    for line in lines:
                        for span in line.get("spans", []):
                            all_spans.append({
                                "size": span.get("size", 0),
                                "font": span.get("font", ""),
                                "flags": span.get("flags", 0),
                                "color": span.get("color", 0),
                                "text": span.get("text", ""),
                            })
                        line_text = " ".join(span["text"] for span in line.get("spans", []))
                        block_text_parts.append(line_text)

                    block_text = "\n".join(block_text_parts).strip()
                    if not block_text:
                        continue

                    block_type = _classify_text_block(
                        text=block_text,
                        bbox=bbox,
                        spans=all_spans,
                        page_height=page_rect.height,
                        page_width=page_rect.width,
                    )

                    block = Block(
                        block_type=block_type,
                        bbox=bbox,
                        text=block_text,
                        page=page_num,
                        order=order,
                        metadata={
                            "fonts": list({s["font"] for s in all_spans}),
                            "avg_font_size": round(sum(s["size"] for s in all_spans) / len(all_spans), 2)
                            if all_spans
                            else 0.0,
                        },
                    )
                    page_layout.blocks.append(block)

                elif raw["type"] == 1:
                    # Image block
                    bbox_data = raw["bbox"]
                    bbox = BBox(
                        x0=bbox_data[0],
                        y0=bbox_data[1],
                        x1=bbox_data[2],
                        y1=bbox_data[3],
                    )
                    block = Block(
                        block_type=BlockType.FIGURE_CAPTION,
                        bbox=bbox,
                        text="[image]",
                        page=page_num,
                        order=order,
                        metadata={
                            "image_extent": raw.get("ext", "unknown"),
                            "width": raw.get("width", 0),
                            "height": raw.get("height", 0),
                        },
                    )
                    page_layout.blocks.append(block)

                order += 1

            # Sort blocks by reading order (top-to-bottom, left-to-right)
            page_layout.blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
            for idx, block in enumerate(page_layout.blocks):
                block.order = idx

            layouts.append(page_layout)

    return layouts


def layout_to_readable_text(layouts: list[PageLayout]) -> str:
    """
    Convert parsed layout back to clean reading-order text.
    Preserves section separators.
    """
    parts: list[str] = []

    for page in layouts:
        for block in page.blocks:
            if block.block_type == BlockType.NOISE:
                continue
            if block.block_type == BlockType.FOOTER:
                continue
            if block.block_type == BlockType.HEADER:
                continue
            if block.block_type == BlockType.PAGE_BREAK:
                parts.append("\n--- PAGE BREAK ---\n")
                continue

            prefix = ""
            if block.block_type == BlockType.TITLE:
                prefix = "\n## "
            elif block.block_type == BlockType.HEADING:
                prefix = "\n### "
            elif block.block_type == BlockType.TABLE:
                prefix = "\n[TABLE]\n"

            parts.append(f"{prefix}{block.text}\n")

    return "\n".join(parts).strip()
