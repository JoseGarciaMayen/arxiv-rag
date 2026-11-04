# RAG evaluation literature and metric choices

This short reading list informs the benchmark protocol. Automated judge scores
are useful diagnostics, not substitutes for a small human-audited set.

## Evaluation frameworks and benchmarks

1. **RAGAS — Es et al., EACL 2024 Demo**. [Paper](https://arxiv.org/abs/2309.15217),
   [ACL Anthology](https://aclanthology.org/2024.eacl-demo.16/). Separates
   retrieval/context relevance from generation faithfulness and answer
   relevance; proposes reference-free LLM-assisted metrics. The paper also
   validates metric preferences against human pairwise judgments (and notes
   that context relevance is harder to judge consistently).
2. **ARES — Saad-Falcon et al., NAACL 2024**.
   [Paper](https://arxiv.org/abs/2311.09476),
   [ACL Anthology](https://aclanthology.org/2024.naacl-long.20/). Evaluates
   context relevance, answer faithfulness, and answer relevance using learned
   judges, then calibrates estimates with a small human-labeled set using
   prediction-powered inference. This motivates retaining human checks and
   uncertainty estimates rather than trusting a judge score alone.
3. **RGB — Chen et al., AAAI 2024**.
   [Paper](https://arxiv.org/abs/2309.01431),
   [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29728). Stress-tests
   generator behavior in four controlled settings: noise robustness, negative
   rejection when evidence is absent, multi-document information integration,
   and counterfactual robustness to misleading evidence.
4. **RAGTruth — Niu et al., ACL 2024**.
   [Paper](https://arxiv.org/abs/2401.00396),
   [ACL Anthology](https://aclanthology.org/2024.acl-long.585/). Provides
   human annotations for unsupported/hallucinated spans in generated RAG
   responses, supporting claim/span-level faithfulness evaluation instead of
   relying only on an overall answer score.

## What this means for arxiv-rag

### Retrieval-only (current benchmark)

- Keep **Recall@k**: are answer-bearing passages found?
- Keep **MRR**: how high is the first answer-bearing passage?
- Keep **nDCG@k**: how well are multiple graded or binary relevant passages
  ordered? If labels remain binary, call this binary nDCG.
- Add **Precision@k** and **HitRate@k** only after pooling candidates and
  annotating relevant *and non-relevant* passages. Current positive-only labels
  cannot support a trustworthy precision estimate.
- Report by query/document as well as macro average. Use bootstrap confidence
  intervals when the set is larger; 50 questions are too few for a strong
  general performance claim.

### End-to-end RAG (next evaluation stage)

For a stratified subset with manually reviewed reference answers/evidence,
record the exact generated answer, retrieved passages, model/provider and
parameters. Evaluate:

- **Answer correctness/completeness** against the reviewed reference answer
  (human rubric first; optionally an LLM judge calibrated against human ratings).
- **Faithfulness/groundedness**: supported claims divided by answer claims,
  with unsupported and contradicted claims distinguished on a human-audited
  sample.
- **Answer relevance**: does it directly answer what was asked?
- **Context precision and recall**: is retrieved evidence useful and complete?
- **Abstention quality**: on questions with no answer in the corpus, measure
  correct abstention and false-answer rate. This needs explicit unanswerable
  questions.
- **Operational metrics**: p50/p95 latency, token/API cost, and failure rate.

Run controlled stress slices (irrelevant/noisy context, missing evidence,
multi-chunk questions, and conflicting/false passages) before claiming robust
production behavior. Separate retrieval experiments from generator experiments
so a score change has an interpretable cause. Freeze a final holdout set that is
not used to tune retrieval parameters or prompts.

## Interpretation cautions

- RAGAS/ARES scores depend on the judge model, prompt, and domain; log those
  versions and manually audit examples, especially disagreements.
- A metric score is not an intrinsic property of the model: it depends on the
  corpus, query distribution, annotation policy, and operating point (`k`).
- Our present question set has answer-bearing IDs but does not yet label every
  retrieved candidate, so Recall/MRR/nDCG are pilot measures and precision,
  context noise, and answer quality remain unmeasured.
