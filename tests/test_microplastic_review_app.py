import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILT_APP = ROOT / "review_packages" / "microplastic_qnn_review_v1" / "index.html"
GENERATOR = ROOT / "scripts" / "build_microplastic_qnn_review_app.py"


def _assert_non_destructive_empty_state(source: str) -> None:
    assert 'id="emptyState"' in source
    assert "viewer.innerHTML" not in source
    assert "emptyState.hidden=false" in source
    assert "emptyState.hidden=true" in source


def test_review_generator_keeps_viewer_nodes_when_a_filter_is_empty() -> None:
    _assert_non_destructive_empty_state(GENERATOR.read_text(encoding="utf-8"))


def test_built_review_app_keeps_all_review_items_and_storage_key() -> None:
    source = BUILT_APP.read_text(encoding="utf-8")
    _assert_non_destructive_empty_state(source)
    assert "microplastic-qnn-review-4575ab680e71" in source

    match = re.search(r"const items=(\[.*?\]);\s*const storageKey", source, re.DOTALL)
    assert match is not None
    items = json.loads(match.group(1))
    assert sum(item["item_type"] == "candidate" for item in items) == 706
    assert sum(item["item_type"] == "audit" for item in items) == 120
