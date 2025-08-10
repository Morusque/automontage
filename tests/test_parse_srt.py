import ast
import re
from pathlib import Path

# Extract `parse_srt` from the original script without executing top-level code
source_path = Path(__file__).resolve().parent.parent / "01 extractTextFromAudio01.py"
source = source_path.read_text(encoding="utf-8")
module = ast.parse(source)
parse_srt_node = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "parse_srt")
parse_srt_code = ast.get_source_segment(source, parse_srt_node)
namespace = {}
exec(parse_srt_code, {"re": re}, namespace)
parse_srt = namespace["parse_srt"]


def test_parse_srt_basic():
    srt = "1\n00:00:00,000 --> 00:00:01,000\nHello world!\n"
    expected = [{"start": 0.0, "end": 1.0, "text": "Hello world!"}]
    assert parse_srt(srt) == expected


def test_parse_srt_ignores_invalid_blocks():
    srt = (
        "1\n00:00:00,000 --> 00:00:01,000\nValid segment.\n\n"
        "2\nInvalid timestamp line\nBroken segment text\n"
    )
    expected = [{"start": 0.0, "end": 1.0, "text": "Valid segment."}]
    assert parse_srt(srt) == expected
