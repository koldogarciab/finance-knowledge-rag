# Formal evaluation

## Evaluation controls

- The 30-question retriever benchmark was frozen before final parameter selection.
- Hybrid weights were selected using a separate 10-question development set.
- No corpus, chunk, embedding or formal case was modified after benchmark freeze.
- Generation evaluation used the local `qwen3.5:4b` model through Ollama.
- End-to-end evaluation used global retrieval, `top_k=5` and no metadata filters.

## Retriever results

### Global retrieval

| Metric | Dense baseline | Hybrid | Change |
|---|---:|---:|---:|
| MRR | 0.7117 | **0.8107** | +0.0990 |
| Hit@1 | 66.67% | **76.67%** | +10.00 pp |
| Hit@3 | 70.00% | **80.00%** | +10.00 pp |
| Hit@5 | 73.33% | **90.00%** | +16.67 pp |
| Hit@10 | 90.00% | 90.00% | 0.00 pp |
| Mean rank | 18.30 | **3.97** | -14.33 |

### Retrieval with metadata filters

| Metric | Dense baseline | Hybrid |
|---|---:|---:|
| MRR | 0.8486 | **0.9031** |
| Hit@1 | 76.67% | **86.67%** |
| Hit@5 | 93.33% | **96.67%** |
| Hit@10 | 100.00% | 100.00% |

### File-type observations

- PDF retrieval improved substantially and reached 1.0000 MRR.
- DOCX and JSON remained at 1.0000 MRR.
- CSV remained the most difficult format because repeated months, departments and similar numeric rows create lexical competition.
- Markdown reached 100% Hit@5 despite a lower MRR than the dense baseline.

## Generation results

### Supported questions

| Metric | Result |
|---|---:|
| Answered | **30 / 30** |
| Unexpected abstentions | **0** |
| Valid citations | **100%** |
| Expected chunk retrieved | **100%** |
| Expected chunk sent to model | **100%** |
| Expected chunk cited | **100%** |
| Mean semantic similarity | **0.8617** |
| Mean reference-token recall | **0.8311** |
| Mean numeric-fact recall | **0.9627** |
| Mean grounded-claim rate | **0.9544** |

Generation modes:

| Mode | Count |
|---|---:|
| Ollama | 28 |
| Ollama repair | 1 |
| Context extractive fallback | 1 |

Automated quality flags:

| Flag | Count |
|---|---:|
| Pass | 23 |
| Review | 7 |
| Fail | 0 |

### Unsupported questions

| Metric | Result |
|---|---:|
| Correct abstentions | **10 / 10** |
| False answers | **0** |
| Abstentions before generation | **10** |
| Ollama calls avoided | **10** |

### Latency

| Metric | Seconds |
|---|---:|
| Mean wall time | 19.403 |
| Median wall time | 21.455 |
| P95 wall time | 35.105 |
| Mean Ollama provider time | 21.891 |

## Manual review

Six of the seven automated-review answers were accepted as substantively correct.

One answer, `formal_json_net_working_capital`, was classified as partial:

- formula and limit were correct;
- citations and grounding were correct;
- the operational and liquidity explanation was incomplete;
- no unsupported information was introduced.

Final manual outcome:

| Verdict | Count |
|---|---:|
| Accepted | **29** |
| Conservative partial | **1** |
| Materially incorrect | **0** |
| Hallucinated | **0** |

## Interpretation

The benchmark demonstrates that the final pipeline can:

- retrieve evidence across five formats;
- preserve financial numbers and policy thresholds;
- provide validated citations;
- reject unsupported requests;
- recover safely from invalid local-model output.

The benchmark is intentionally transparent but limited in size. It supports the project conclusions without claiming production-level generalisation.
