# What I found running the configs

I ran four retrieval setups over the same 31-question gold set and scored each one
on faithfulness, answer relevancy, context precision, and context recall. Here's
what actually came out, and what I think it means.

| Config | Embedding | Chunk | top-k | Rerank | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|--------|-----------|-------|-------|--------|-------------|-------------|---------------|------------|
| A | MiniLM | 256 | 3 | no | 0.794 | 0.745 | 0.742 | 0.726 |
| B | MiniLM | 512 | 5 | no | 0.839 | 0.813 | 0.703 | 0.750 |
| C | BGE-small | 256 | 3 | no | 0.810 | 0.765 | 0.729 | 0.739 |
| D | BGE-small | 256 | 3 | yes | **0.860** | **0.835** | 0.694 | **0.769** |

## The short version

Adding a reranker (config D) gave the best answers — highest faithfulness, highest
answer relevancy, and the best context recall of the bunch. If I had to ship one
setup, it's this one. The catch is that it also had the *lowest* context precision,
which makes sense once you look at how it works, and I get into that below.

## Going config by config

**A vs C — does the embedding model matter?**
A and C are the same setup (256-char chunks, top-3, no rerank) with only the
embedding model swapped: MiniLM in A, BGE-small in C. BGE came out ahead on
faithfulness (0.810 vs 0.794) and answer relevancy (0.765 vs 0.745). Not a huge
jump, but it's consistent — the better embedding model retrieves slightly better
context, and the answers improve because of it. Worth the swap since both models
are about the same size and speed.

**A vs B — bigger chunks and more of them.**
B uses 512-char chunks and pulls back 5 instead of 3. Context recall went up
(0.750 vs 0.726), which is exactly what you'd expect — more text and more chunks
means a better chance the piece that actually answers the question is in there.
But precision dropped (0.703 vs 0.742). Also expected: when you grab more context,
some of it is only loosely related, so the signal-to-noise ratio gets worse. This
is the classic precision/recall tradeoff and it showed up cleanly in the numbers.

**C vs D — what reranking buys you.**
D is config C with a cross-encoder reranker bolted on: it pulls a wider set of
candidates, then re-scores them and keeps the best few. The answer-side metrics
jumped — faithfulness went from 0.810 to 0.860, answer relevancy from 0.765 to
0.835. So the reranker is clearly handing the generator better material to work
with.

The odd one out is context precision, which actually got *worse* (0.694, the
lowest of all four configs). I think what's happening is that the reranker starts
from a bigger candidate pool than plain retrieval does, so even after re-ranking,
a few borderline chunks survive into the final set. The generator handles them
fine — that's why faithfulness still went up — but the precision metric, which
judges the retrieved context directly, penalizes those extra chunks.

## What I'd actually ship

**Config D (BGE-small + reranking).** It produces the most faithful, most relevant
answers, and that's what a user actually experiences. The lower context precision
is a real tradeoff, not something I'd hide, but in practice the generator is good
at ignoring the bit of extra noise, so it doesn't hurt the final answer. If I were
running this somewhere that cared a lot about retrieval cost, I'd revisit it, since
reranking adds a model call to every query.

## Things I'd be honest about in an interview

- **The judge is a 7B local model (Mistral), not GPT-4.** It's noisier. I validated
  it by hand on a sample (see GOLD_DATASET.md) but the absolute scores should be
  read as "roughly in this range," not precise to the third decimal. The *relative*
  comparison between configs is the part I trust.
- **A few judge calls returned unparseable output** and got recorded as null, then
  excluded from the averages. It was a small number out of 124 question-evals, but
  it's there and I didn't want to paper over it.
- **The config D run time was way off** (the reranker model downloaded mid-run and
  my machine likely slept partway through), so I don't report wall-clock time as a
  metric — only the quality scores, which are all complete.
- **31 questions is a modest set.** Enough to see real differences between configs,
  not enough to make strong claims about small gaps. I'd want more questions before
  treating, say, a 0.01 precision difference as meaningful.