import pandas as pd

from density_audit.eval_dino_shift import included_rows
from density_audit.random_seed import resolve_seed
from scripts.sanitize_checkpoint import sanitize


def test_manifest_include_parser_handles_strings() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["a", "b", "c", "d", "e"],
            "include": ["true", "false", "1", "0", "yes"],
        }
    )
    assert included_rows(frame)["dataset"].tolist() == ["a", "c", "e"]


def test_explicit_seed_is_preserved() -> None:
    assert resolve_seed(12345) == 12345


def test_generated_seed_is_in_supported_range() -> None:
    seed = resolve_seed(None)
    assert 0 <= seed < 2**31


def test_checkpoint_sanitizer_preserves_model_token_keys() -> None:
    payload = {"model": {"backbone.cls_token": "tensor-placeholder"}, "api_key": "private"}
    sanitized = sanitize(payload)
    assert sanitized["model"]["backbone.cls_token"] == "tensor-placeholder"
    assert sanitized["api_key"] == "<redacted>"
