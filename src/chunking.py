"""Syntax-aware Python and section-aware Markdown/text segmentation."""

import ast
import bisect
import re

from src.models import Chunk


def _windows(
    text: str, start: int, end: int, limit: int,
) -> list[tuple[int, int]]:
    """Split a region at line boundaries with bounded overlap."""
    spans: list[tuple[int, int]] = []
    while start < end:
        stop = min(start + limit, end)
        if stop < end:
            newline = text.rfind("\n", start + limit // 2, stop)
            if newline >= 0:
                stop = newline + 1
        spans.append((start, stop))
        if stop == end:
            break
        overlap = min(200, limit // 5)
        start = max(start + 1, stop - overlap)
    return spans


def chunk_file(
    text: str, file_path: str, limit: int, python: bool,
) -> list[Chunk]:
    """Preserve exact character offsets while splitting code or text."""
    boundaries = {0, len(text)}
    symbols: list[tuple[int, int, str]] = []
    if python:
        lines = [0] + [m.end() for m in re.finditer("\n", text)]
        try:
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    line = min([node.lineno] + [
                        dec.lineno for dec in node.decorator_list
                    ])
                    start = lines[line - 1]
                    end_line = node.end_lineno or node.lineno
                    end = (lines[end_line] if end_line < len(lines)
                           else len(text))
                    boundaries.add(start)
                    symbols.append((start, end, node.name))
        except (SyntaxError, ValueError, RecursionError):
            pass
    else:
        for match in re.finditer(r"(?m)^#{1,6}\s+|\n[ \t]*\n", text):
            boundaries.add(match.start())
    ordered = sorted(boundaries)
    # Pack small logical sections together; split only oversized sections.
    regions: list[tuple[int, int]] = []
    start = 0
    for boundary in ordered[1:]:
        if boundary - start > limit:
            previous = ordered[bisect.bisect_left(ordered, boundary) - 1]
            if previous > start:
                regions.append((start, previous))
                start = previous
            if boundary - start > limit:
                regions.extend(_windows(text, start, boundary, limit))
                start = boundary
    if start < len(text):
        regions.append((start, len(text)))
    return [Chunk(
        file_path=file_path, first_character_index=start,
        last_character_index=end, text=text[start:end],
        symbols=" ".join(name for a, b, name in symbols
                         if a < end and b > start),
    ) for start, end in regions if text[start:end].strip()]
