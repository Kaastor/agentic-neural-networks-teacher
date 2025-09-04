# AI Portfolio Projects

A cohesive, production-first portfolio that covers RAG, time-series, recommenders, vision, safety, and platform work — now with reinforcement learning (RL) integrated where it creates real product value, plus a dedicated RL project for depth.

---

## At-a-glance Build Order (compounding value)

1. **Lakehouse Mini (12)** → 2) **DemandForecast++ (3)** → 3) **ExperimentHub (10)**
2. **TrustyRAG (1)** → 5) **ModelGateway (9)** → 6) **DocRedactor (5)**
3. **SupportCopilot (8)** → 8) **ColdStartRec (4)** → 9) **LogSleuth (2)**
4. **EdgeVision QA (6)** → 11) **PriceOptRL (7)** → 12) **GenImageSafe (11)** → 13) **PolicyLab RL Platform (NEW)**

> **Why this order:** Data foundations first, then forecasting + eval infra; core LLM apps; gateway/ops; safety/compliance; then exploration systems and edge CV; finish with explicit RL depth.

---

## Portfolio Checklist (every app ships with)

* **Problem doc**: KPIs, baselines, kill criteria, success thresholds.
* **Signed artifacts**: model/prompt/index with SBOM + SLSA attestation.
* **Eval report**: slice coverage + bootstrap CIs; judge hygiene for LLMs.
* **Runbook**: shadow → canary → rollback + oncall & SLOs.
* **Cost dashboard**: per-request cost & utilization; budget alerts.
* **Governance**: model/data cards; audit trails; safety checklist.
* **Diagrams**: C4 + sequence + data lineage.

---

# Projects

## 1) TrustyRAG — Span‑Grounded Enterprise Search (with RL‑tuned retrieval blend)

**What:** Ask questions over a private corpus using **hybrid BM25+ANN+rERANK** with span‑level citations and structured JSON answers.
**RL angle:** Contextual bandit to learn **reranker weighting** and query‑rewrite policy from offline click/venue metrics; OPE (IPS/DR) to gate online.

**KPIs:** Support‑coverage\@k ≥ 0.85; faithfulness error ≤ 3%; p95 ≤ 1.2s; \$/answer.
**Core tech:** FastAPI; PyTorch/transformers; FAISS/ScaNN + BM25; signed/versioned prompts & retrievers; JSON schemas.
**Evals:** Coverage/faithfulness with judge hygiene (pairwise + anchors + CIs + human‑agreement sampling) in CI; slice by doc type.
**Ops/Safety:** Canary + rollback; evals‑in‑prod hooks; PII redaction at ingest; immutable IDs; audit trail.
**Artifacts:** Model/prompt/index registry w/ signatures, eval pack dashboard, cost board, runbook.

---

## 2) LogSleuth — LLM + IR for SRE Triage (with policy learning)

**What:** Natural‑language queries over logs/metrics; safe **function calls** for diagnostics.
**RL angle:** Bandit policy to prioritize diagnostic **tool sequences** that minimize MTTD, trained offline from incident postmortems; conservative OPE before canary.

**KPIs:** MTTD ↓ 30%; false‑story rate ≤ 1%; p95 streaming ≤ 1.0s.
**Core tech:** Retrieval over time‑windowed Parquet; sandboxed command tools; structured JSON actions.
**Evals:** Synthetic incidents + real postmortems; regression tests; seed stability.
**Ops/Safety:** SSRF/tool hardening; least‑privilege IAM; shadow → canary; per‑slice costs.
**Artifacts:** Trace views, sequence diagrams, signed release notes.

---

## 3) DemandForecast++ — Classical → GBM → DL Time‑Series (with control policy hand‑off)

**What:** Forecast daily item demand with baselines → GBMs → optional seq2seq.
**RL link:** Provide **policy features** to downstream control (e.g., reorder points, promos). Pair with PolicyLab or PriceOptRL to evaluate inventory/pricing policies under forecast uncertainty.

**KPIs:** MAPE / pinball loss; calibration within CI bands; infra cost / forecast.
**Core tech:** Leakage‑free CV; feature store; contracts & schemas; idempotent/backfillable pipelines.
**Evals:** Stratified eval sets; per‑slice error; bootstrap CIs; drift monitors.
**Ops/Safety:** Data lineage; retention/cost controls; SBOM for jobs; signed models.
**Artifacts:** Data contracts that fail CI, cost/utilization board, model/data cards.

---

## 4) ColdStartRec — Safe Exploration Recommender (RL‑first)

**What:** Popularity/ALS baselines → two‑tower retrieval + re‑rank; **bandits/OPE** for exploration.
**RL:** Epsilon‑greedy/Thompson with guardrails; slate re‑ranking via policy gradients offline with strict IPS/DR and variance control.

**KPIs:** CTR/Conversion lift; calibration\@k; regret; fairness slices.
**Core tech:** Implicit MF; ANN retrieval; off‑policy evaluation (IPS/DR/SNIPS); policy constraints.
**Evals:** Offline → OPE → tiny canary; regression tests.
**Ops/Safety:** Kill criteria; consent/opt‑out surfaces; audit trails.
**Artifacts:** Simulator notebook, policy cards, rollout runbook.

---

## 5) DocRedactor — PII Discovery & Redaction as a Service

**What:** Ingest docs, detect PII, redact, and return signatures & audit proofs.
**RL:** N/A (keep deterministic); optional active learning loop (non‑RL) for boundary cases.

**KPIs:** Recall ≥ 0.97 @ precision ≥ 0.95; p95 ≤ 800ms; \$/page.
**Core tech:** Rules + NER; licensing/IP hygiene; GDPR data‑subject flows.
**Evals:** Stratified PII testbed; robustness & drift alerts.
**Ops/Safety:** SBOM + SLSA; cosign for artifacts; immutable audit logs.
**Artifacts:** DPA template, governance/model cards, compliance checklist.

---

## 6) EdgeVision QA — Low‑Latency Defect Detection (with bandit gating)

**What:** On‑device classifier (mixed‑precision) with cloud assist for hard cases.
**RL/Bandits:** **Deferred inference** gating via contextual bandits to decide when to escalate to cloud; reward balances FP/FN vs energy and latency.

**KPIs:** Miss rate ≤ 1% at FP ≤ 2%; on‑device p95 ≤ 40ms; energy budget.
**Core tech:** Lightweight CNN/ViT; PEFT/quantization; batching & admission control.
**Evals:** Stability/conditioning checks; per‑slice; regression seeds.
**Ops/Safety:** OTA with signature verify; rollback; backup/restore; cost per metric.
**Artifacts:** Hardware profiling, flamegraphs, deployment signatures.

---

## 7) PriceOptRL — Safe RL for Dynamic Pricing (core RL app)

**What:** Simulator + PPO/A2C with constraints; staged to live with risk caps.
**KPIs:** Profit lift vs control; constraint violations ≤ 0.5%; regret.
**Core tech:** MDP/POMDP formalism; reward design; **offline datasets + OPE**; seeds/reproducibility; uncertainty‑aware value estimates.
**Evals:** Ablations; counterfactual tests; fairness slices.
**Ops/Safety:** Canary for policies; kill switches; governance notes.
**Artifacts:** Simulator hygiene doc, policy eval reports, policy cards.

---

## 8) SupportCopilot — SOP‑Grounded Agent for CS Teams (policy‑guided)

**What:** Agent drafts replies grounded in verified SOPs/KB with span citations; tools for ticket lookup and refunds.
**RL:** Preference model + constrained policy optimization to **select tools and reply strategies** that minimize handle time while keeping hallucination ≤ 1%; offline OPE, then small canary.

**KPIs:** Time‑to‑resolve ↓; hallucination ≤ 1%; CSAT ↑; \$/ticket.
**Core tech:** Hybrid RAG; tool use with structured JSON; prompt/index versioning & signing.
**Evals:** Faithfulness/support coverage; human‑agreement sampling; judge CI.
**Ops/Safety:** Tool sandbox; transparency UI; shadow → canary.
**Artifacts:** HITL reviewer UX; rollback plan; policy cards.

---

## 9) ModelGateway — Secure, Observable Inference Mesh (policy‑aware serving)

**What:** FastAPI/gRPC gateway: batching, streaming, quotas, retries, circuit breakers.
**RL fit:** Hosts **policy services** (bandits/RL) next to LLM/CV models; admission control integrates policy constraints and cost budgets.

**KPIs:** p90/p99 budgets; error‑budget burn; cost/request.
**Core tech:** Token accounting; per‑tenant quotas; traces/metrics/logs; evals‑in‑prod hooks.
**Evals:** Load/regression tests; chaos drills; seed stability for canaries.
**Ops/Safety:** SBOM, SLSA, cosign verify at deploy; backups/restores for models/vectors/prompts.
**Artifacts:** C4 + sequence diagrams, runbooks, dashboards.

---

## 10) ExperimentHub — Metrics Trees & CI Evals (with RL OPE gates)

**What:** Define **metrics trees**, manage eval sets, and gate CI on model/data changes.
**RL support:** First‑class **OPE suites** (IPS/DR/SNIPS, confidence bounds), simulator adapters, and regression budgets for policies.

**KPIs:** % repos with CI evals; mean time to detect regressions.
**Core tech:** Dataset versioning; slice metrics; behavior‑driven tests for LLMs + policy tests for RL.
**Evals:** Bootstrap CIs; judge calibration jobs; off‑policy diagnostics.
**Ops/Safety:** Signed eval packs; immutable IDs; audit trails.
**Artifacts:** Screenshots/reports; reusable GitHub Actions templates.

---

## 11) GenImageSafe — Diffusion with Safety & IP Guardrails

**What:** Content generation with IP/license checks, consent capture, safety filters.
**RL:** Optional user‑feedback bandit for **prompt safety rewrites** (opt‑in), not for image generation itself.

**KPIs:** Unsafe output rate ≤ 0.5%; p95 ≤ 1.2s; dispute rate.
**Core tech:** Diffusion pipeline; license classifier; watermark detection.
**Evals:** Red‑team prompts; robustness tests; bias slices.
**Ops/Safety:** Model/data cards; signed artifacts; auditability.
**Artifacts:** Governance pack, safety checklist, red‑team report.

---

## 12) Lakehouse Mini — Contracts, Catalog & Cost (RL‑ready data)

**What:** A small **lakehouse** backing the portfolio with contracts, lineage, and **backfillable** pipelines.
**RL boost:** Curates offline logs (actions/rewards/propensities) for **bandits/RL**; generates counterfactual simulators and cohort slices.

**KPIs:** Contract violations in CI = 0; backfill success ≥ 99%; \$/TB/month.
**Core tech:** Parquet/Delta; schema registry; retries/idempotency; data quality checks; entitlement & catalog.
**Evals:** Drift monitors; range/dist/null checks; retention policy tests.
**Ops/Safety:** SBOM awareness; IAM; quotas; containerization + basic K8s.
**Artifacts:** Data maps, lineage graphs, cost allocation report.

---

## 13) **PolicyLab RL Platform** — Offline→Online RL, Safely (NEW)

**What:** A reusable **RL platform** with simulators (pricing, inventory control, troubleshooting sequences), offline policy learning, OPE, and safe online rollout tooling.
**Why:** Centralize RL infra so other apps can plug policies in without duplicating work.

**KPIs:** Policy lift vs control (by domain); OPE–online gap ≤ X%; safety constraint violations ≤ 0.5%; time‑to‑canary.
**Core tech:** PPO/A2C/DQN baselines; conservative & constrained RL; reward models from preferences; IPS/DR/DR‑Doubly Robust; uncertainty‑aware evaluation; seeding & reproducibility.
**Evals:** Simulator realism checks; ablations; counterfactual tests; fairness slices.
**Ops/Safety:** Canary scaffolds; kill‑switches; policy versioning/signing; audit trails.
**Artifacts:** Simulator hygiene doc; policy eval dashboards; rollout runbook; reusable SDK.

---

# Cross‑Cutting RL Patterns (used across apps)

* **Offline‑first**: Learn from logs; use IPS/DR/SNIPS for risk‑aware gating.
* **Constraints & transparency**: Formalize guardrails (budget, safety, fairness) as hard constraints or penalties; expose policy cards.
* **Reward shaping & preference modeling**: Where explicit rewards are sparse, learn reward models from pairwise preferences and align with KPIs.
* **Small, safe canaries**: 1–5% traffic with kill criteria; promote only on pre‑declared thresholds.
* **Repro & governance**: Seeds, deterministic data snapshots, signed artifacts, immutable IDs.

---

## Appendix — Suggested Repo Templates (per project)

```
repo/
  apps/<app-name>/
    api/ (FastAPI/gRPC)
    policies/ (bandits/RL)
    prompts/ (signed + versioned)
    evals/ (CI suites + OPE)
    infra/ (IaC, k8s manifests)
    dashboards/ (SLO + cost)
    runbook.md
  data/
    contracts/
    pipelines/
    datasets/ (versioned snapshots)
  platform/
    gateway/ (ModelGateway)
    experimenthub/ (metrics trees + CI)
    policylab/ (RL SDK + simulators)
  .github/workflows/ (CI gates)
```

## First Issues to Tackle (starter set)

* Add **metrics tree** & wire to CI (ExperimentHub).
* Sign artifacts with **cosign** in CI; attach SBOM/SLSA.
* Implement **OPE harness** (IPS/DR) with calibrated CIs.
* Stand up **ModelGateway** with quotas, rate limits, tracing.
* Create **Lakehouse** contracts and backfill jobs.
* Ship one polished **demo UI** per app.
