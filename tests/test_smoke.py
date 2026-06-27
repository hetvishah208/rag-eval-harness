"""Lightweight import + structure smoke tests (no model calls)."""
import json

import yaml


def test_configs_load():
    for p in ["config/default.yaml", "config/experiments.yaml"]:
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg, f"{p} is empty"


def test_default_config_keys():
    with open("config/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ["corpus", "chunking", "embedding", "vector_store", "retrieval", "generation", "judge"]:
        assert k in cfg, f"missing key: {k}"


def test_gold_dataset_wellformed():
    with open("data/eval/gold_qa.json", encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) >= 1
    for r in rows:
        assert "question" in r and "ground_truth" in r


def test_parse_score():
    from src.eval.run_eval import parse_score
    assert parse_score("0.8") == 0.8
    assert parse_score("0.8 - the answer is supported") == 0.8
    assert parse_score("1.0") == 1.0
    assert parse_score("nonsense with no number") is None
    # out-of-range leading token should not pass through as a valid score
    assert parse_score("on a scale of 1 to 10, this is 8") in (None, 1.0)


def test_modules_importable():
    import src.utils                # noqa
    import src.ingest.chunk_docs    # noqa
    import src.ingest.build_index   # noqa
    import src.retrieve.retriever   # noqa
    import src.generate.answer      # noqa
    import src.eval.run_eval        # noqa