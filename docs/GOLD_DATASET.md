# How I built and checked the gold dataset

## Where the questions came from

I generated the question set instead of hand-writing it from scratch, then checked
every single one by hand. The process:

1. Pulled substantive chunks (over ~200 characters) from the indexed corpus.
2. For each chunk, had a local model (Mistral 7B via Ollama) draft a question and a
   reference answer based only on that chunk.
3. Reviewed every draft myself and kept the good ones.

I over-generated on purpose — drafted 70, kept the ones that held up — so the final
set is the stuff that survived review, not whatever the model happened to spit out.

## What I rejected and why

I dropped a draft if any of these were true:

- The answer wasn't actually supported by the chunk it came from.
- The question only made sense if you were already looking at a specific code
  snippet ("what model is used in this snippet?") — useless for a retrieval system
  that has no idea which snippet you mean.
- It was answerable from general knowledge without needing the docs at all.
- It was a near-duplicate of one I'd already kept.
- The answer was just wrong, or the question was too vague to have one right answer.

A lot of the rejects were the "in this snippet / in this chunk" type, plus a few
where example-dataset text had leaked into the docs and the model wrote a question
about *that* instead of about Transformers.

## Final numbers

- Drafted: 70
- Failed to parse / dropped automatically: 9
- Reviewed by hand: 61
- Kept after review: 28 (plus 3 I'd seeded earlier by hand = 31 total)
- Roughly a 46% keep rate on the reviewed batch

The point of writing this down is that the keep rate itself says something — a good
chunk of LLM-drafted questions aren't good eval questions, and the value is in the
filtering, not the generating.

---

# Checking whether I can trust the judge

The eval uses Mistral 7B as the judge, which is a lot smaller than what people
usually use (GPT-4-class models). Smaller judge = noisier scores. So instead of just
trusting it, I spot-checked it against my own judgment.

## How I did it

I took 13 questions from a completed eval run, ran each one back through the RAG
pipeline so I could read the actual generated answer and the retrieved sources, and
scored two metrics myself by hand — faithfulness and answer relevancy — on the same
0-to-1 idea the judge uses. Then I compared my scores to the judge's. I counted a
pair as "agreement" when my score and the judge's landed on the same side (both high,
both middling, or both low).

## What I found

I agreed with the judge on **12 out of 26** individual judgments — about **46%**.
Faithfulness lined up on 5 of 13, answer relevancy on 7 of 13.

The bigger takeaway is *how* it disagreed: the judge made mistakes in **both
directions**, which is worse than being consistently strict or consistently lenient.

**Too lenient on hallucinations.** This was the most important failure. On several
questions the model invented technical detail that wasn't in the docs — a fake
constructor signature for 4-bit loading, a made-up processor example, an entirely
fabricated "Gemma vision encoder tower" that doesn't exist. I scored those near 0
because they're wrong and not grounded in the retrieved context. The judge gave them
0.9. A 7B judge clearly struggles to notice when generated code or API names are
plausible-looking but fake.

**Too harsh on correct answers.** The flip side. On the BEiT question the answer was
completely right — correct model name, correct expansion of the acronym, grounded in
the source — and the judge gave it 0.5. Same pattern on a couple of others where the
answer was fine but got marked down.

**It agreed with me on the easy cases.** Where the answer was clearly good and
clearly grounded (the quantization-purpose question, the attention-kernel params, the
gradient-checkpointing one), the judge and I matched. So it's okay at the obvious
ends and unreliable in between.

## What this means for the results

I'd read the config-comparison results as directional, not precise. The judge is
noisy enough per-question that I wouldn't stake anything on a small gap between two
configs. The averages across 31 questions smooth a lot of that out, and the relative
ordering of the configs still seems reasonable, but the absolute scores should be
taken with a grain of salt.

If this were going into anything real, the single highest-value upgrade would be
swapping the 7B judge for a stronger model — the hallucination-detection gap is the
kind of thing that directly undermines a faithfulness metric. That's the honest
limitation of doing this entirely on free, local, CPU-only tooling, and it's a
tradeoff I made on purpose to keep the whole project zero-cost.