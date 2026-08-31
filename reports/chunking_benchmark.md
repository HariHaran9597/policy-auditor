## Chunking / reranking benchmark (full-evidence hit rate@1)

A question is a hit only when every annotated evidence span has at least one overlapping retrieved chunk. The 15 answerable questions form the denominator; designed refusals are excluded.

| Configuration | Chunks | Reranker | Full-evidence hits | Hit rate | Δ vs previous | Target coverage |
|---|---:|:---:|---:|---:|---:|---:|
| Fixed 512 chars | 53 | no | 11/15 | 73.3% | — | 76.5% |
| Semantic structure-aware | 32 | no | 12/15 | 80.0% | +6.7 pp | 82.3% |
| Semantic + 2-line overlap | 32 | no | 14/15 | 93.3% | +13.3 pp | 94.1% |
| Semantic + overlap + BGE reranker | 32 | yes | 14/15 | 93.3% | +0.0 pp | 94.1% |

### Per-question results

| Question | fixed-512 | semantic | semantic-overlap | semantic-overlap-reranker |
|---|---:|---:|---:|---:|
| How many working days of annual leave do employees get, and how many days can be carried over into the next year? | miss | miss | hit | hit |
| How long is paternity leave and within how many months of the child's birth must it be used? | hit | hit | hit | hit |
| What is the per-night hotel ceiling for a Grade L3 employee travelling within India, and what flight class are they entitled to? | hit | hit | hit | hit |
| What is the notice period when a Grade L3 employee resigns? | hit | hit | hit | hit |
| Within how many days must the Internal Committee complete a POSH inquiry, and how quickly are complaints acknowledged? | hit | hit | hit | hit |
| What is the minimum password length and the password rotation period? | hit | hit | hit | hit |
| After how many failed login attempts is an account locked, and for how long? | hit | hit | hit | hit |
| What are the acknowledgement and resolution targets for a Sev2 security incident? | hit | hit | hit | hit |
| How quickly must a suspected personal-data breach be reported and to whom? | hit | hit | hit | hit |
| How many days per week must an employee living within 50 km of an office work from that office? | hit | hit | hit | hit |
| As of March 2026, what is the monthly remote-work internet allowance for a Grade L1 fully-remote employee? | hit | hit | hit | hit |
| Can Restricted data be accessed from a personal device? | miss | miss | miss | miss |
| How many hours per week of side employment cause the approval to be denied? | hit | hit | hit | hit |
| What performance rating is required to be eligible for promotion, and what else must be true? | miss | hit | hit | hit |
| Can unused sick leave be carried into the next year? | miss | miss | hit | hit |
| What is the tuition reimbursement for an employee's child's college education? | — | — | — | — |
