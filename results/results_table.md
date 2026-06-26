| config | name | chunk_size | embedding | top_k | reranker | faithfulness | answer_relevancy | context_precision | context_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | MiniLM / 256 / top-3 / no rerank | 256 | all-MiniLM-L6-v2 | 3 | None | 0.794 | 0.745 | 0.742 | 0.726 |
| B | MiniLM / 512 / top-5 / no rerank | 512 | all-MiniLM-L6-v2 | 5 | None | 0.839 | 0.813 | 0.703 | 0.750 |
| C | BGE-small / 256 / top-3 / no rerank | 256 | BAAI/bge-small-en-v1.5 | 3 | None | 0.810 | 0.765 | 0.729 | 0.739 |
| D | BGE-small / 256 / top-3 / with rerank | 256 | BAAI/bge-small-en-v1.5 | 3 | cross-encoder/ms-marco-MiniLM-L-6-v2 | 0.860 | 0.835 | 0.694 | 0.769 |
