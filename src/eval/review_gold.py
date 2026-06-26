"""
Interactive review tool for gold draft Q&A pairs.

Shows each draft one at a time (question + answer + source chunk) and lets you:
  k = keep as-is
  r = reject (drop it)
  e = edit the question and/or answer, then keep
  q = quit and save progress so far
  b = go back to previous item

Writes verified keepers to data/eval/gold_qa.json (the format run_eval.py expects).
The source_chunk field is stripped from the final output (it's only for your review).

Usage:
    python -m src.eval.review_gold
    python -m src.eval.review_gold --in data/eval/gold_draft.jsonl --out data/eval/gold_qa.json
"""
import argparse
import json
import os


def load_drafts(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_keepers(out_path):
    """Resume support: if gold_qa.json already has rows, keep them."""
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def save_keepers(out_path, keepers):
    # strip review-only fields before writing
    clean = []
    for k in keepers:
        clean.append({
            "question": k["question"],
            "ground_truth": k["ground_truth"],
            "source": k.get("source", ""),
            "category": k.get("category", ""),
            "verified": True,
        })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)


def fmt_answer(gt):
    """ground_truth may be a string or a list."""
    if isinstance(gt, list):
        return ", ".join(str(x) for x in gt)
    return str(gt)


def multiline_input(prompt, default):
    print(prompt)
    print(f"  (current: {default})")
    print("  Enter new value, or press Enter to keep current:")
    val = input("  > ").strip()
    return val if val else default


def review(drafts, out_path):
    keepers = load_existing_keepers(out_path)
    start_count = len(keepers)
    if start_count:
        print(f"[review] resuming — {start_count} keepers already in {out_path}\n")

    rejected = 0
    reject_reasons = []
    i = 0
    history = []  # stack of (index, action) for 'back'

    total = len(drafts)
    while i < total:
        d = drafts[i]
        print("=" * 72)
        print(f"  ITEM {i + 1}/{total}   |   kept so far: {len(keepers)}   rejected: {rejected}")
        print("=" * 72)
        print(f"\n  SOURCE: {d.get('source', '?')}")
        print(f"\n  Q: {d['question']}")
        print(f"\n  A: {fmt_answer(d['ground_truth'])}")
        print(f"\n  --- source chunk (what the Q was drafted from) ---")
        chunk = d.get("source_chunk", "(no chunk stored)")
        print("  " + chunk.replace("\n", "\n  ")[:700])
        print("\n" + "-" * 72)
        print("  [k]eep   [r]eject   [e]dit   [b]ack   [q]uit & save")
        choice = input("  > ").strip().lower()

        if choice == "k":
            keepers.append(d)
            history.append(("keep", i))
            i += 1
        elif choice == "r":
            reason = input("  reject reason (optional, for your README stats): ").strip()
            if reason:
                reject_reasons.append(reason)
            rejected += 1
            history.append(("reject", i))
            i += 1
        elif choice == "e":
            new_q = multiline_input("\n  EDIT QUESTION:", d["question"])
            new_a = multiline_input("\n  EDIT ANSWER:", fmt_answer(d["ground_truth"]))
            edited = dict(d)
            edited["question"] = new_q
            edited["ground_truth"] = new_a
            keepers.append(edited)
            history.append(("keep", i))
            print("  [saved edited version]")
            i += 1
        elif choice == "b":
            if history:
                action, prev_i = history.pop()
                if action == "keep" and keepers:
                    keepers.pop()
                elif action == "reject":
                    rejected -= 1
                i = prev_i
                print("  [went back]")
            else:
                print("  [already at first item]")
        elif choice == "q":
            print("\n[review] quitting early, saving progress...")
            break
        else:
            print("  (unrecognized — use k/r/e/b/q)")

    save_keepers(out_path, keepers)
    print("\n" + "=" * 72)
    print(f"[review] DONE. {len(keepers)} keepers written to {out_path}")
    print(f"[review] rejected this session: {rejected}")
    if reject_reasons:
        print("[review] reject reasons (for your GOLD_DATASET.md stats):")
        for r in reject_reasons:
            print(f"    - {r}")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/eval/gold_draft.jsonl")
    ap.add_argument("--out", default="data/eval/gold_qa.json")
    args = ap.parse_args()
    drafts = load_drafts(args.inp)
    print(f"[review] loaded {len(drafts)} drafts from {args.inp}\n")
    review(drafts, args.out)