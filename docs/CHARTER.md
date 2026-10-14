Charter: TabPFN LLM Router

Mission

Build and evaluate a TabPFN-powered decision router for selecting an appropriate LLM for an incoming question.

The router should learn from empirical question/model performance and make the cheapest reliable model selection possible.

Core Principle

Do not build another LLM.

Build a small decision system around TabPFN.

QUESTION
↓
FEATURES
↓
TABPFN
↓
ROUTING DECISION
↓
LLM

Phase 1 — Repository and Architecture

Establish the project structure.

Separate:

- data collection
- feature generation
- model training
- routing policy
- LLM evaluation
- benchmarking
- inference

Keep the TabPFN implementation replaceable so alternative classifiers can be benchmarked later.

Phase 2 — Question Corpus

Create a diverse question corpus.

Do not optimize the dataset for one domain.

Include easy, moderate, and difficult examples across:

- knowledge
- reasoning
- mathematics
- programming
- debugging
- ML/data science
- writing
- analysis
- multi-step tasks
- ambiguous tasks

Avoid artificially making every question difficult.

The dataset should represent realistic user questions.

Phase 3 — Candidate Models

Select three representative LLM tiers:

CHEAP
STANDARD
FRONTIER

The exact models should be configurable rather than hard-coded into the router.

For every question, record available evaluation information for each model.

At minimum:

- success/failure
- quality score
- latency
- token usage
- estimated cost

Phase 4 — Ground Truth

Determine the minimum model tier capable of successfully solving each question.

Preferred target:

CHEAP succeeds → CHEAP
STANDARD succeeds → STANDARD
FRONTIER required → FRONTIER

Do not blindly equate human-perceived difficulty with model difficulty.

Human ratings may be retained as auxiliary labels/features.

Phase 5 — Feature Engineering

Construct a compact tabular representation of each question.

Start with inexpensive deterministic features.

Then investigate semantic/embedding-derived features.

Do not build an unnecessarily huge feature space.

The router must be cheap enough that feature extraction does not erase the savings from routing.

Phase 6 — TabPFN

Train TabPFN as the primary routing classifier.

Evaluate:

fit(X, y)
predict(X)
predict_proba(X)

Record probability estimates.

Do not immediately optimize thresholds.

First establish whether TabPFN can learn the basic routing problem.

Phase 7 — Decision Policy

Implement a layer between classifier output and final model selection.

The classifier predicts capability/tier.

The policy determines the actual action.

Support at least:

- maximum-probability routing
- confidence threshold
- conservative routing

Eventually support cost-aware routing.

Example:

P(CHEAP) = .82
P(STANDARD) = .15
P(FRONTIER) = .03

→ CHEAP

A low-confidence example should be permitted to escalate.

Phase 8 — Baselines

Implement simple alternatives:

- always CHEAP
- always STANDARD
- always FRONTIER
- random
- rule-based difficulty router

The TabPFN router must beat useful baselines before additional complexity is introduced.

Phase 9 — Evaluation

Measure:

- routing accuracy
- model-selection accuracy
- answer quality
- task success
- cost
- latency
- escalation rate
- quality/cost frontier

The most important comparison is:

TABPFN ROUTER
vs.
ALWAYS FRONTIER

Report both quality retained and cost saved.

Phase 10 — Error Analysis

Inspect routing failures manually.

Categorize failures such as:

- cheap model unexpectedly succeeds
- cheap model unexpectedly fails
- deceptive/simple-looking question
- long but easy question
- short but difficult question
- domain-specific difficulty
- reasoning failure
- coding failure
- ambiguity
- insufficient context

Use these findings to improve the dataset before adding model complexity.

Phase 11 — Final Benchmark

Produce a reproducible benchmark showing:

Quality Cost
CHEAP ... ...
STANDARD ... ...
FRONTIER ... ...
RULE ROUTER ... ...
TABPFN ROUTER ... ...

Also report the distribution of routing decisions.

Guardrails

Do not:

- assume difficulty labels are ground truth
- optimize against the test set
- leak model evaluation information into inference features
- use expensive LLM calls to generate features if they cost more than the routing savings
- declare success from routing accuracy alone
- confuse classifier confidence with actual answer quality

Maintain a strict train/validation/test separation.

Exit Criteria

The project is ready for a second iteration if:

1. The end-to-end router works.
2. The evaluation dataset contains meaningful variation in difficulty.
3. TabPFN beats simple routing baselines.
4. The router demonstrates measurable cost savings.
5. Quality degradation relative to frontier-only inference is quantified.
6. Failure modes are understood.

If those conditions are not met, improve the dataset and target definition before increasing model complexity.