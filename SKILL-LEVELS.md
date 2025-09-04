# Level 1 — CORE FOUNDATIONS (ship a reliable v1)

### 0) Operating Principles

* Outcomes→KPIs, baselines before fancy; write it down; version everything; **evals/obs/safety/cost** from day one. **\[Must]**
* Artifact integrity: sign & verify models/prompts/indices/datasets; immutable IDs; audit trails. **\[Must]**

### 1) Mathematical & ML Foundations

* Linear algebra, calculus-for-optimization, stability/conditioning; autodiff checks; mixed-precision guardrails. **\[Must]**
* Probability & statistics: calibration, CIs vs significance, bootstrap, uncertainty-aware thresholds. **\[Must]**
* Learning basics: bias–variance, regularization, early stopping; ablations. **\[Must]**

### 2) Software Engineering Mastery

* Python 3.x excellence (typing, generators, asyncio, vectorization), SQL proficiency; clean architecture, packaging. **\[Must]**
* Testing (unit/integration/property), lint/format/type gates, CI; profiling & flamegraphs. **\[Must]**
* Tooling hygiene incl. SBOM awareness. **\[Must]**

### 3) Data Engineering & DataOps

* Ingestion+storage (Parquet, lakehouse), contracts & schemas, idempotent/backfillable pipelines; lineage, retries. **\[Must]**
* Data quality: null/range/dist checks, drift monitors, PII discovery/redaction; contract tests that fail CI. **\[Must]**
* Cost & retention basics; entitlement/catalog awareness. **\[Must]**

### 4) Classical ML Bread-and-Butter

* Solid baselines: linear/logistic, trees/GBMs, SVMs, clustering; leakage-free CV; imbalance tactics; calibration. **\[Must]**

### 5) Deep Learning (applied core)

* NLP/Transformers essentials: tokenization, embeddings, IR/ranking, summarization/QA/NER; prompt-length budgeting. **\[Must]**
* Generative models: diffusion/VAEs/GANs fundamentals + safety/IP pitfalls. **\[Must]**
* Awareness-only: CNN/ViT basics; ASR/TTS; graph learning—prefer simpler baselines first. **\[Must]** (awareness depth kept minimal here)

### 6) Training, Scaling & Efficiency

* Throughput/utilization basics: AMP/bfloat16, clipping, schedulers, checkpointing, dataloader performance. **\[Must]**
* PEFT/quantization/distillation fundamentals; memory–latency trade-offs. **\[Must]**
* Cost thinking: \$ per achieved metric; GPU utilization dashboard staples. **\[Must]**

### 7) LLMs, RAG & Agents (Applied GenAI)

* Prompting craft; structured JSON outputs; function calling; **hybrid BM25+ANN + rerank**; dedupe/normalize embeddings. **\[Must]**
* **Span-grounded citations**; evals that check support-coverage & faithfulness\@k; **judge hygiene** in CI (pairwise + anchors + CIs + human agreement sampling). **\[Must]**
* **Version & sign** prompts/retrievers/indices; joint model+index+prompt rollback. **\[Must]**
* Basic guardrails & secure tool sandboxing. **\[Must]**

### 8) Inference, Serving & Reliability

* FastAPI/gRPC; streaming; timeouts/retries/circuit breakers; rate limiting; shadow→canary→rollback. **\[Must]**
* Observability: traces/metrics/logs, **evals-in-prod**, per-slice metrics, token/req cost. **\[Must]**
* Capacity: batching, admission control, P90/P99 budgeting; degrade modes. **\[Must]**
* Backups & restores for models/vectors/prompts/features; verify signatures at deploy. **\[Must]**

### 9) Experimentation, Evaluation & Testing

* Metrics trees; stratified eval sets; slice metrics; fairness/robustness basics; dataset versioning; CI gates. **\[Must]**
* LLM behavior-driven tests; seed stability; regression tests. **\[Must]**

### 10) Security, Privacy, Compliance & Safety

* Threat model with ML-specific risks; SSRF/tool hardening; least-privilege IAM; secrets mgmt. **\[Must]**
* **SBOM + SLSA baseline**; **artifact signing & verification (Sigstore/cosign)**. **\[Must]**
* PII discovery/redaction; licensing/IP hygiene (CC-BY-SA, non-commercial, field-of-use). **\[Must]**
* Governance basics: model/data cards, audit trails, red-team prompts; GDPR/CCPA basics; EU AI Act tiers awareness. **\[Must]**

### 11) Product, UX & Human Factors

* Problem framing, KPIs & kill criteria; uncertainty messaging; consent & transparency surfaces; HITL reviewer UX basics. **\[Must]**

### 12) Cloud, Hardware & Edge

* Cloud fluency: IAM, networking, storage classes, quotas; cost allocation; containerization + basic K8s. **\[Must]**

### 13) Collaboration & Leadership

* Docs-as-code: design docs, runbooks, model/data cards; C4/sequence diagrams; cross-functional fluency. **\[Must]**

### 14) Reinforcement Learning (Core pillar)

* MDP/POMDP formalism; DQN & PPO/A2C basics; reward design anti-patterns; simulator hygiene; seeds/reproducibility. **\[Must]**
* Offline datasets & **basic OPE**; safe rollout/canary for policies. **\[Must]**

**Core artifacts & routines (lightweight, no projects):** reproducible run kit; signed registries for model/prompt/index; eval pack with slice coverage & judge calibration; runbook with canary/rollback; cost dashboard spec; safety checklist.

---

# Level 2 — SYSTEMS & SCALE (senior depth, own production)

### 0) Operating Principles → programmatic

* Make **“evals-as-code + cost/latency SLOs”** a platform habit; scale documentation & ablation rigor org-wide. **\[Strong]**

### 1) Foundations at scale

* Learning theory for capacity/regularization; uncertainty propagation across pipelines; causal cautions; time-series stationarity & **policy-aware metrics**. **\[Strong]**
* Numerics upgrades: Kahan, log-domain ops, NaN sentinels; finite-difference autodiff tests at suite level. **\[Strong]**

### 2) Software Engineering & Systems

* Concurrency: async vs threads vs procs; queues/backpressure; idempotent RPC/gRPC patterns; graceful shutdown. **\[Strong]**
* Performance engineering: memory/cache locality; NUMA awareness; before/after profiles with hard SLAs. **\[Strong]**

### 3) Data Engineering & DataOps

* Streaming (Kafka/Flink), backfills at scale; lineage/cost SLAs; programmatic data quality & drift playbooks. **\[Strong]**
* Labeling: weak/active supervision, programmatic labeling; agreement metrics; **synthetic-data governance**. **\[Strong]**
* Feature stores (offline/online TTL & skew detection) with point-in-time correctness. **\[Strong]**

### 4) Classical ML at scale

* Hyperparameter search (Optuna), leakage diagnostics automation, feature importance caveats for governance; calibrated ensembles. **\[Strong]**

### 5) Deep Learning depth

* Time-series DL (temporal models, probabilistic forecasts); vision detection/segmentation hygiene; long-context trade-offs. **\[Strong]**
* Generative control (LoRA/ControlNet), safety filters at training & serving. **\[Strong]**

### 6) Training/Scaling

* DDP/FSDP/ZeRO; sharded optimizers; failure recovery; elastic training on spot; schedulers (SLURM/K8s). **\[Strong]**
* Compression at scale (QLoRA, pruning/sparsity, MoE routing basics) with accuracy deltas and rollback. **\[Strong]**

### 7) LLMs, RAG & Agents

* Evolving eval packs; **index/prompt registries** with rollback; cost/latency SLOs for **judges & rerankers**; slice-drift monitors. **\[Strong]**
* Agents: multi-step planning, approval gates, scoped tools, sandboxing, failure policies (idempotent retries, DLQs). **\[Strong]**

### 8) Inference & SRE

* ONNX/AOT, tokenizer throughput, KV paging; batching/streaming policy tuning; autoscaling models. **\[Strong]**
* Multi-region readiness; DR/restore drills; **vendor abstraction & quota/price alarms** with automatic cut-over. **\[Strong]**

### 9) Experimentation & Causality

* A/B beyond basics: CUPED, sequential tests, non-inferiority, novelty effects; ramp policies & long-tail monitoring. **\[Strong]**

### 10) Security, Privacy, Compliance

* Robustness: poisoning/backdoor checks, membership inference risk; watermark/fingerprint handling. **\[Strong]**
* Privacy depth: DP basics (noise budgets), federated patterns; content provenance policy & surfacing. **\[Strong]**
* Policy: DPIAs, data residency & cross-border transfers; vendor SLAs/quotas with degrade modes. **\[Strong]**

### 11) Product & Human Factors

* Explainability (global/local), selective abstention; fairness mitigation playbooks; judge/feedback disagreement monitoring. **\[Strong]**

### 12) Cloud, Hardware & Edge

* GPU/TPU fundamentals, kernels/CUDA basics; throughput vs latency; TVM/ONNX Runtime; vLLM/TensorRT-LLM serving. **\[Strong]**
* Edge/mobile awareness → acceptance tests & minimal on-device eval packs. **\[Strong]**

### 13) Collaboration & Leadership

* Technical leadership: roadmaps, ADRs, design reviews; on-call & incident playbooks; cross-team alignment on SLOs. **\[Strong]**

### 14) Reinforcement Learning depth

* **GAE**, exploration (ε-greedy, UCB/Thompson, entropy/curiosity awareness), distributional RL. **\[Strong]**
* **Bandits & uplift**, reward modeling via pairwise prefs; **offline RL (CQL/IQL/TD3+BC)**; OPE at scale (IPS/DR/FQE) with CIs; **constraints/CMDPs**. **\[Strong]**
* Productionization: vectorized envs, policy registries, admission control for online learning. **\[Strong]**

---

# Level 3 — MASTERY & GOVERNANCE (staff/principal scope)

### Strategy & Standards

* Org-wide standards for **model+prompt+retriever+index** registries with signing, provenance, and **one-command rollback** across fleets. **\[Strong]**
* Portfolio bets & roadmap; cost–reliability flywheel; multi-vendor posture & hard cost ceilings. **\[Strong]**
* Safety/governance programs: red-teaming, incident response, approvals/audits; content provenance & watermark policy. **\[Strong]**

### Foundations & Numerics

* Risk-aware numerics (stability under shift), uncertainty budgeting across decision systems; calibration governance. **\[Strong]**

### Platforms & Reliability

* Multi-region design, disaster recovery objectives, error budgets; company-wide observability incl. **evals-in-prod** & judge decision logs + human audits. **\[Strong]**
* DR drills as a program (models/vectors/prompts/features); key rotation; supply-chain hardening to SLSA targets. **\[Strong]**

### LLMs, RAG & Agents at fleet level

* Model/index fleet strategy; safety tuning programs; judge/reranker monitoring with rollback playbooks; **token/cost/latency SLOs** as platform contracts. **\[Strong]**

### Experimentation as an org capability

* Metrics trees across teams; slice regression alerts wired to accountability; experimentation governance (pre-registration, non-inferiority standards). **\[Strong]**

### Security, Privacy, Regulation

* Enterprise threat modeling including tool chains; **SBOM + SLSA targets** enforced; artifact attestations everywhere. **\[Strong]**
* Policy leadership: GDPR/CCPA operations, **EU AI Act** risk-tier implementations, DPIAs, residency; vendor reviews. **\[Strong]**

### Product, UX & Human Systems

* Org guidance on uncertainty/fairness UX, reviewer incentives, escalation pathways; cross-domain fairness governance. **\[Strong]**

### Cloud/Hardware/Edge Strategy

* Accelerator portfolio strategy; capacity planning; NUMA/PCIe/NVLink-aware deployments; edge privacy & thermal budgets as policy. **\[Strong]**

### Collaboration & Leadership

* Mentorship programs; reusable patterns; blameless postmortems with action owners; documentation culture. **\[Strong]**

### Reinforcement Learning at scale

* Leads **multi-agent RL**, hierarchical/options, PBT; risk-sensitive RL (CVaR), robust RL under shift; **RLHF/RLAIF pipelines**. **\[Strong]**
* Org-wide RL governance: **policy safety program** (risk budgets, kill-switches), reward policy reviews, environment/experience registries with signing & rollback; **online-learning SLOs** & FinOps for sim/online. **\[Strong]**

---

## Tech Stack Progression (what to be fluent with by level)

* **Level 1:** Python/SQL/Bash; scikit-learn/XGBoost/LightGBM; PyTorch; HF Transformers; FAISS/pgvector + BM25; FastAPI/gRPC; Docker; basic K8s; MLflow/W\&B; Great Expectations/Soda; Prometheus/Grafana/OTel; Sigstore/cosign; Airflow/Prefect; Kafka; DataHub; Evidently/Giskard/Ragas/promptfoo.
* **Level 2:** Lightning/Accelerate; DeepSpeed/FSDP; Optuna; vLLM/TensorRT-LLM; KServe/Seldon/BentoML; Elastic/OpenSearch kNN; Milvus/Weaviate; Redis caches; Helm; TVM/ONNX Runtime; Flink/Spark/Delta/Iceberg; Feast; privacy/robustness toolkits; RLlib/Stable-Baselines3/CleanRL; d3rlpy/FQE.
* **Level 3:** Ray Train/Serve; Megatron-style parallelism; policy/index/judge registries at org scale; confidential computing patterns; governance stacks; accelerator strategy (TPU/GPU mix); DR/backup automation.

---

## Readiness & Anti-Patterns (carry across levels)

* **Readiness checklists:** use the provided Data/RAG, Training, Serving, Security, Human Factors, RL checklists as “go/no-go” gates at each level. Aim to tick every box before declaring a capability done.
* **Pitfalls to avoid at all levels:** unversioned or unsigned artifacts (esp. prompts/indices/judges), leakage via time/joins, demos without evals/citations/safety, unbounded queues, no DR drills, unmonitored judges/rerankers, RL without OPE/safety gates.

---

### How to use this roadmap

* Treat **Level 1** as your universal base—don’t skip.
* Layer **Level 2** to reliably own production systems.
* Use **Level 3** to operate at org scope (strategy, governance, and fleet-level excellence).

If you want, I can turn this into a checklist you can track against (same structure, box-tickable).
