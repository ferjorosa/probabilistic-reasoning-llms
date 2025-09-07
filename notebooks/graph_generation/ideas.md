Probar diferentes tipos de estructuras:

(Podemos controlarlo desde el treewidht?, al final tree-widh correlates with exact inference hardness)

* Naive Bayes
* TAN
* Tree
* Forest
* DAG
    * We should probably test different kind of DAGs here + random (?)

--------------------

Segund Chat-gpt: https://chatgpt.com/share/68bd8975-7f98-800d-a58c-8c7873bfcd12

### Key complexity controls (parameters to sweep)

n (num vars): e.g., {3, 5, 10, 20, 50, 100}.

Max indegree / treewidth: generate graphs with treewidth targets (w = 1,2,3,4,5, …). Treewidth correlates with exact-inference hardness.

Variable arity: binary vs ternary vs quaternary. Multi-valued variables increase state-space.

CPT skewness: sample CPT rows from Dirichlet(α) with α small (0.1 — skewed) vs α=1 (uniform) vs α large (flat).

Determinism fraction: proportion of CPT rows with deterministic 0/1 entries (e.g., 0%, 10%, 50%).

Evidence fraction / type: no evidence, 1 observed, small set (10%), large set (50%), and rare evidence (observations with low prior probability).

Query types: P(X), P(X|E), joint probabilities P(X, Y | E), MAP(X|E), most probable explanation (MPE).

### Query selection

For eahc BN instance, sample a fixed set of queries

Make sure queries are balanced across easy/hard regions (e.g., variables in different topological positions — leaves, roots, hubs).

### Evaluation metrics

For each (LLM answer p̂, ground truth p):

Absolute Error (AE): |p̂ − p|.

Squared Error (SE): (p̂ − p)².

Root Mean Squared Error (RMSE): sqrt(mean(SE)).

Mean Absolute Error (MAE): mean(AE).

Maximum Absolute Error across queries.

Fraction within tolerance: e.g., % queries with AE < 0.01, <0.05.

KL divergence (for distributions / joint queries): KL(P || Q) if you have full distributions; be careful if p̂ = 0.

Brier score for binary-event predictions (Brier = (p̂ − y)² averaged over labeled outcomes) — if you simulate outcomes.

Calibration curve & expected calibration error (ECE) — if LLM also gives confidence estimates or you transform probabilities into bins and compare empirical frequencies.

For MAP/MPE tasks: accuracy of the returned argmax (is the LLM’s predicted most probable assignment actually the MAP?) and top-k accuracy.

Also log:

Parsing failures and non-numeric output rates.

LLM runtime / tokens (optional).

Which prompt style produced best results.

### Hypotheses to test (examples)

> To me they are not that interesting. I think the most important point is to understand

Error grows with treewidth more than with raw node count.

Deterministic CPTs (logical constraints) are especially hard for LLMs.

LLMs do worse on queries requiring summing over many latent configurations (many hidden variables to marginalize).

Few-shot examples and chain-of-thought may help for small nets but not for large treewidth.

Skewed CPTs (close to 0/1 or rare events) produce larger relative errors.