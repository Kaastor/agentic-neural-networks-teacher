# SKILLS

### Legend

* **Must** = core for almost every applied role; you should demonstrably do this.
* **Strong** = frequently needed; expected for senior/lead, or to own production.
* **Awareness** = know concepts/trade-offs; ramp fast when a project demands it.

## Tracks & Leveling Matrix (calibration helper)

Three overlapping **tracks** share foundations but diverge in emphasis:

* **A) Applied ML (Product):** modeling, experimentation, user/KPI impact.
* **B) ML Platform & Infra:** reliability, scale, cost, tooling, governance.
* **C) GenAI / RAG:** LLM lifecycle, retrieval, prompt/tool use, guards/evals.
* **D) Reinforcement Learning (RL):** sequential decision-making, online/offline RL, safe deployment. *(treated as a primary algorithmic pillar)*

| Skill Area                         | L3 (Mid)                                                                                                                                                                                                                                                                                | L4 (Senior)                                                                                                                                                                                                                                                    | L5 (Staff)                                                                                                                                                                                                                                  | L6 (Principal)                                                                                                                                                                                                             |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A: Applied ML**                  | Must in data quality, **contract tests** & leakage-free CV, GBM & basic DL, **evals-as-code**                                                                                                                                                                                           | + Strong in experiment design, per-slice metrics, product KPIs                                                                                                                                                                                                 | + Leads cross-team metrics tree, launch guardrails, fairness/long-tail playbooks                                                                                                                                                            | + Org-wide standards, drives strategy/portfolio bets                                                                                                                                                                       |
| **B: Platform/Infra**              | Must in CI/CD, containers, FastAPI, basic K8s, observability, **artifact signing & attestations**                                                                                                                                                                                       | + Strong in canary/rollback, cost dashboards, feature/feature-store reliability                                                                                                                                                                                | + Owns multi-region, disaster recovery, SLO/error budgets, **SBOM + SLSA targets**, supply chain                                                                                                                                            | + Company-wide platform vision, cost & reliability flywheel                                                                                                                                                                |
| **C: GenAI/RAG**                   | Must in prompting, JSON schema outputs, **hybrid search + rerank**, span-grounded citations, **LLM-judge hygiene in CI (pairwise + anchors + CIs + human agreement sampling)**, **prompt/retriever/index versioning with signed immutable IDs**, **basic guardrails + tool sandboxing** | + Strong in eval packs evolution, guardrails tuning, index versioning & rollback                                                                                                                                                                               | + Leads SFT/PEFT, token/cost/latency SLOs, prompt & retriever registries, judge/reranker monitoring                                                                                                                                         | + Safety/governance programs, model & index fleet strategy, vendor posture                                                                                                                                                 |
| **D: Reinforcement Learning (RL)** | **Must** in MDP/POMDP formalism, value-based (DQN) & policy-gradient (PPO/Actor–Critic) basics, simulator hygiene, reward design anti-patterns, **offline datasets & basic OPE**, safe rollout/canary for policies, reproducibility & seeds                                             | **Strong** in GAE, entropy/exploration (ε-greedy, UCB/Thompson, entropy/curiosity), distributional RL, **bandits & uplift**, reward modeling/preference learning (pairwise), **offline RL (CQL/IQL/TD3+BC)**, OPE at scale (IPS/DR/FQE), **constraints/CMDPs** | Leads simulator-service integration, **multi-agent RL**, PBT, hierarchical/options, risk-sensitive RL (CVaR), robust RL under shift, **RLHF/RLAIF pipelines**, credit assignment, environment/experience registries with signing & rollback | Org-wide RL strategy & governance, **policy safety program** (risk budgets, kill-switches), reward policy reviews, cross-domain RL platforms (recsys/ads/robotics/ops), **online-learning SLOs** and FinOps for sim/online |

Use this to map **Must / Strong / Awareness → level expectations** and to keep interviews aligned.

## 0) Operating Principles (meta **Musts**)

* **Outcomes > models.** KPIs, user value, and incremental shipping.
* **Data first.** Quality, lineage, governance, drift, cost.
* **Ruthless simplicity.** Baselines, ablations, cheapest thing that works.
* **Reproducibility.** One-command runs (envs, seeds, data snapshots).
* **Observability.** Metrics/logs/traces/evals before scaling.
* **Safety & compliance by default.** PII, licensing, security posture, abuse cases.
* **FinOps mindset.** Model the cost/quality curve; watch P90/P99 and \$/req or \$/token.
* **Write it down.** Design docs, ADRs, runbooks, postmortems, model/data cards.
* **Version everything.** Data, code, models, **prompts**, **retrievers**, **indices**, **evals**.
* **Provenance & integrity.** **Attest & sign every release artifact** (models/prompts/indices/datasets); immutable IDs (`model@sha`, `prompt@sha`); verify at deploy; audit trails.
  **Proof:** crisp design docs & experiment charters, ablation tables, runbooks, **cost dashboards**, postmortems, **registries with signed artifacts and provenance**.

## 1) Mathematical & ML Foundations

**Principles:** numerics & stability matter as much as algorithms; diagnose conditioning & uncertainty; prefer calibrated decisions.

* **Linear algebra & calculus (with optimization) — Must** Gradients/Jacobians/Hessians, conditioning, optimizer & schedule choices, normalization/initialization. **Proof:** implement custom loss + gradient; catch FP16/bfloat16 issues; clip & scale correctly; **autodiff correctness tests** (finite differences). **Pitfalls:** silent NaNs, unbounded losses, ignoring conditioning. **Numerics upgrades:** Kahan summation, log-domain ops, mixed-precision failure modes & guardrails, **loss-NaN sentinels**.
* **Probability & statistics — Must** Bayesian & frequentist intuition; CIs vs significance; calibration; bootstrap; uncertainty propagation; causal cautions. **Proof:** calibrated probabilities, uncertainty-aware thresholds & dashboards.
* **Learning theory & generalization — Strong** Bias–variance, capacity, regularization, early stopping; data vs model vs training ablations.
* **Information & signal processing — Strong** Filtering, spectral methods, compression/entropy, sampling theory.
* **Time-series & causality — Strong** Stationarity, leakage traps, proper CV, counterfactuals, **policy-aware metrics**.
  **Stack:** NumPy, SciPy, JAX, PyTorch/TensorFlow, statsmodels, pymc, scikit-learn.

## 2) Software Engineering Mastery

**Principles:** clean boundaries, typed/auto-checked code, deterministic builds, measured performance.

* **Language excellence (Python + perf language; SQL) — Must** Idiomatic Python 3.x, type hints, generators, asyncio; vectorization; FFI (C++/Rust). **SQL:** window functions, CTEs, cost-aware queries.
* **Code quality & scale — Must** Clean architecture, isolation, packaging; memory/cache locality; profiling. **Proof:** flamegraphs, speedups with before/after profiles.
* **Concurrency & systems — Strong** Async vs threads vs procs; queues/backpressure; RPC/gRPC; idempotency; graceful shutdown.
* **Tooling & hygiene — Must** Tests (unit/integration/property), lint/format, typing gates, pre-commit, CI, versioning, **SBOM awareness**.
  **Stack:** Python, C++/Rust, SQL; Poetry/pip-tools, pytest/hypothesis, ruff/black, mypy, py-spy/perf.

## 3) Data Engineering & DataOps

**Principles:** data contracts + idempotent, observable pipelines; **point-in-time correctness**; cost & privacy baked in.

* **Ingestion & storage — Must** Batch/stream; schemas/contracts; Parquet/Avro; lakehouse; partitioning/Z-ordering; compaction.
* **Pipelines — Must** Idempotency, retries, backfills, lineage; late/duplicated data handling; SLAs/SLOs; cost control.
* **Quality & governance — Must** Checks (nulls, ranges, distributions), drift monitors; PII discovery/redaction; catalogs; entitlements; retention; **contract tests that fail CI**.
* **Labeling & supervision — Strong** Active/weak supervision, HITL, programmatic labeling; agreement metrics; **synthetic-data governance** (provenance, labeling, drift/BSA risks).
* **Feature engineering at scale — Strong** Offline/online stores, TTLs, point-in-time correctness; skew detection; feature costs.
  **Stack:** Spark/Databricks, Flink, Kafka/Kinesis, Airflow/Prefect/Dagster, dbt, Delta/Iceberg, Parquet, Feast, Great Expectations/Soda, DataHub, LakeFS/DVC.

## 4) Classical ML Bread-and-Butter

**Principles:** strong baselines beat fancy models; prioritize leakage-free CV and calibration.

* **Algorithms — Must:** linear/logistic, trees/GBMs, SVMs, clustering, naïve Bayes; know when GBM beats Transformer for tabular.
* **Modeling habits — Must:** robust CV, leakage prevention, class imbalance tactics, feature importance caveats, calibration.
  **Stack:** scikit-learn, XGBoost/LightGBM/CatBoost, Optuna/Hyperopt.

## 5) Deep Learning Architectures

**Principles:** transfer & data-centric tricks > model bloat; measure long-context & tokenizer costs.

* **Vision — Awareness:** CNNs, ViTs; detection/segmentation; OCR; augmentations; small-data transfer.
* **NLP — Must:** Tokenization, Transformers, embeddings, IR/ranking, summarization/QA/NER; long-context trade-offs; instruction tuning basics; **tokenizer throughput & prompt-length budgeting**.
* **Speech & audio — Awareness:** ASR/TTS/diarization; streaming latency.
* **Time-series with DL — Strong:** Temporal models, anomalies, probabilistic forecasts.
* **Graph learning — Awareness:** GNN primitives; **prefer simpler baselines** first.
* **Generative models — Strong:** Diffusion/VAEs/GANs; control (LoRA/ControlNet); safety filters; IP/licensing risks.
* **Reinforcement & bandits — **Upgraded to Core via Section 14**:** see the dedicated RL section for full stack/ops/evals.
  **Stack:** PyTorch (core), TF/Keras, HF Transformers/Diffusers, PyG/DGL, Lightning/Accelerate.

## 6) Training, Scaling & Efficiency

**Principles:** maximize **utilization** and **throughput**; pay only for quality that moves KPIs.

* **Performance training — Must:** AMP/bfloat16, scaling/clipping, schedulers, regularization, checkpointing, dataloader throughput.
* **PEFT & compression — Must:** LoRA/QLoRA/PEFT; distillation; pruning/sparsity; MoE routing basics; memory–latency trade-offs.
* **Distributed training — Strong:** DDP/FSDP/ZeRO; tensor/pipeline parallel; sharded optimizers; failure recovery.
* **Schedulers & clusters — Strong:** SLURM/K8s; preemption-safe jobs; elastic training; spot-aware strategies.
* **Cost awareness — Must:** GPU utilization & memory bandwidth; pricing models; **\$ per achieved metric** dashboards.
  **Stack:** PyTorch DDP/FSDP, DeepSpeed, Megatron-LM, Ray Train/Tune, NCCL, PyTorch Profiler, W\&B/MLflow/Comet.

## 7) LLMs, RAG & Agents (Applied GenAI)

**Principles:** retrieval before fine-tune; **hybrid search + rerank**; **span-grounded citations**; **version + sign prompts and indices**.

* **LLM lifecycle — Strong:** pretrain vs SFT vs RLHF/RLAIF/DPO; safety tuning; evals per phase; data licenses & contamination hygiene; release gates.
* **Prompting craft — Must:** roles, few-shot/CoT, tool use, routing, self-correction; structured outputs (JSON/Schema), function calling, deterministic parsing; **prompt registry + hashing/signing**.
* **RAG — Must:** chunking (semantic/overlap), **hybrid BM25+ANN**, reranking, embeddings hygiene (dedupe, normalization), caching (semantic/page), **retriever & index versioning with signed immutable IDs**, **joint model+index+prompt rollback**.
* **RAG metrics — Must:** **support-coverage** (answers with grounded spans), **faithfulness\@k** with span grounding, **answerable-vs-unanswerable detection** + abstention rate; fail CI on regression.
* **Evals & hallucination mitigation — Must:** golden sets, adversarial prompts, citation-aware checks, trace-based evals, **LLM-as-judge calibration** (**pairwise > absolute; anchors; human agreement sampling; CIs on win-rates**).
* **Judges & rerankers as first-class models — Must:** track versions, **monitor slice drift**, rollback plans, cost/latency SLOs.
* **Agents & orchestration — Strong:** multi-step planning, **approval gates**, **tool scopes**, memory/state, failure policies (idempotent retries, dead-letter queues), sandboxing tools.
* **Efficient inference — Must:** KV cache mgmt, paged attention, quantization (AWQ/GPTQ), speculative decoding, batching/streaming; **cost/latency SLOs**.
  **Stack:** Embeddings/ANN: FAISS, HNSWlib, ScaNN, pgvector, Milvus/Weaviate/Pinecone, Elastic/OpenSearch kNN. RAG/Agents: LangChain, LlamaIndex, Haystack, LangGraph. Serving: vLLM, TensorRT-LLM, TGI, Triton, FastAPI/gRPC, Redis caches. Eval/Guardrails: Ragas, Giskard, DeepEval, promptfoo, NeMo/Guardrails.ai, OpenTelemetry.

## 8) Inference, Serving & Reliability

**Principles:** **SLO-driven** engineering; safe rollouts; everything observable; **sign & verify artifacts**; **version prompts/retrievers** alongside models.

* **APIs & services — Must:** streaming, admission control, rate limiting, retries/timeouts/circuit breakers; multi-region patterns; schema contracts.
* **Optimization — Strong:** ONNX/AOT, fusion, pinned memory/NUMA, KV paging, tensor layouts; CPU offload; tokenizer throughput.
* **Rollouts — Must:** shadow/canary/blue-green; feature flags; **model+prompt+index registry integration**; one-command rollback.
* **Observability — Must:** traces/metrics/logs; golden signals; token/req cost; **evals in prod**; data slices & drift; **judge decision logs** with sampled human audits.
* **Capacity & cost — Must:** autoscaling, throughput models, batching policies, queueing theory 101; P90/P99 budgets; SLOs & error budgets.
* **DR & continuity — Must:** backup/restore for **models, vectors, prompts, features**; periodic **restore drills**; key rotation; degrade modes.
* **Vendor dependency playbooks — Must:** quotas/price spikes alarms; **multi-vendor abstraction**, automatic cut-over within hard cost ceilings.
  **Stack:** Docker, Kubernetes, Helm, KServe/Seldon/BentoML, FastAPI/gRPC, ONNX/TVM, Triton, vLLM/TensorRT-LLM, Prometheus/Grafana, OpenTelemetry, Loki, **Sigstore/cosign**.

## 9) Experimentation, Evaluation & Testing

**Principles:** “**evals as code**” in CI; align offline↔online; measure slices & stability, not just averages.

* **Experiment design — Strong:** metrics trees (north-star vs proxies), traffic allocation, power/duration, novelty effects; offline↔online alignment.
* **Model eval — Must:** stratified sets, per-slice metrics, calibration, **fairness & robustness**; versioned datasets; CI gates.
* **A/B & beyond — Strong:** CUPED, sequential tests, guardrails, long-tail monitoring; non-inferiority tests; ramp policies.
* **ML testing — Must:** data tests, slice tests, regression tests, behavior-driven tests for LLMs (prompt–response contracts), seed stability.
* **LLM eval hygiene — **Must (explicit)**:** judge calibration, position-bias mitigation, **human agreement checks**, **confidence intervals on win-rates**; **log judge rationales** for audit.
  **Stack:** MLflow/W\&B/Comet, EvidentlyAI, Giskard, CheckList, Great Expectations/Soda, `scipy.stats`, metrics stores.

## 10) Security, Privacy, Compliance & Safety

**Principles:** **threat model every tool path**; least privilege; audit trails; plan for abuse and data residency.

* **Security for ML systems — Must:** STRIDE + ML-specific risks; supply chain (deps, **SBOM**, **SLSA targets**), **artifact signing & verification (Sigstore/cosign)**, model exfiltration, prompt injection, SSRF via tools, sandboxing, signing & attestations.
* **Privacy — Must:** PII discovery/redaction, data minimization, retention; DP basics (noise budgets); federated patterns; **private inference awareness** (TEEs; PIR/FHE/MPC **Awareness**).
* **Robustness — Strong:** poisoning defenses, backdoor checks, membership inference risk; watermark/fingerprints; model fingerprinting.
* **Governance — Must:** model/dataset cards, approvals & audit trails, role-based entitlements; human review policies; red-teaming; incident response.
* **Policy & regulation — Must:** GDPR/CCPA basics, sector regs, **EU AI Act** risk tiers & obligations; **data residency & cross-border transfers** (DPIAs); **vendor SLAs/quotas/degrade modes**.
* **Licensing/IP hygiene — Must (with pitfalls):** CC-BY-SA share-alike contamination; non-commercial datasets in commercial settings; field-of-use limits; scraping T\&Cs; **proof via license inventory and CI checks**.
* **Content provenance & watermark handling — Strong:** detect/record provenance signals; trust policy; fallbacks when absent; surface to users when relevant.
  **Stack:** Rebuff/Guardrails, Presidio, Opacus/TF-Privacy, TEEs/HSMs/Confidential VMs, Vault/Secrets Manager, signed registries/attestations.

## 11) Product, UX & Human Factors

**Principles:** align with user jobs-to-be-done; communicate uncertainty; reduce cognitive load; earn trust.

* **Problem framing — Must:** constraints → objective; success metrics; risks; buy vs build; stakeholder map; de-scope paths.
* **Human-in-the-loop — Must:** reviewer UX, escalation, feedback capture; queue design; incentive alignment.
* **Explainability & trust — Strong:** global/local explainers, uncertainty surfacing, selective abstention, **citations in RAG**.
* **Change management — Strong:** incident playbooks, on-call, postmortems; versioned releases.
* **UX writing & style — Must:** response tone guidelines, error/uncertainty messaging, **opt-outs/consent** surfaces, transparency defaults.
* **Fairness in production — Must:** **alerts on slice regressions**, mitigation playbooks, review loops with domain owners.
* **Judge/feedback observability — Must:** monitor disagreement rates, slice drift; periodic human audits; rollback plan.
  **Stack:** Product analytics, SHAP/Captum, Streamlit/Gradio/Next.js, design docs/RFCs.

## 12) Cloud, Hardware & Edge

**Principles:** least privilege, right-size everything, design for latency/thermal envelopes at the edge.

* **Cloud fluency — Must:** IAM least-privilege, VPCs/subnets, peering, security groups, storage classes & lifecycle, quotas; cost allocation/chargeback.
* **Accelerators — Strong:** GPU/TPU fundamentals, kernels, CUDA basics, Triton-lang; throughput vs latency; PCIe/NVLink.
* **Edge & mobile — Awareness → Strong if applicable:** QAT, distillation, thermals, offline modes, privacy; **acceptance test (“passes on a 5-year-old Android”)** and minimal on-device eval pack.
  **Stack:** AWS/GCP/Azure ML, Terraform/Pulumi, NVIDIA CUDA/cuDNN, Triton-lang, Jetson; ONNX Runtime, TFLite, Core ML.

## 13) Collaboration & Leadership

**Principles:** clear contracts, high-signal docs, reusable patterns, and blameless learning loops.

* **Technical leadership — Strong:** roadmaps, ADRs, design reviews, mentoring, risk burndown; cross-team alignment.
* **Cross-functional fluency — Must:** product/data/platform/legal/security; shared SLOs.
* **Documentation — Must:** READMEs, runbooks, model/data cards, user docs; C4 & sequence diagrams.
  **Stack:** Docs-as-code, ADR/RFC templates, Notion/Confluence, GitHub Projects/Jira.

## 14) Reinforcement Learning (Core) — **Primary Algorithm Track**

**Principles:** safe exploration, reward integrity, offline-first where possible, **OPE before online**, reproducibility, and SLOs for online learning.

* **Formalisms & basics — Must:** MDPs/POMDPs; value vs policy methods; Bellman operators; advantage/GAE; entropy regularization; baselines; reward scaling/normalization.
* **Algorithms — Must:** DQN (+ Double, Dueling, PER), PPO/TRPO, A2C/A3C, SAC/TD3 (continuous), distributional RL (C51/QR-DQN), deterministic vs stochastic policies.
* **Exploration — Must:** ε-greedy, UCB/Thompson (bandits), entropy bonuses; **curiosity/ICM/RND/RE3** awareness; count-based/exploration bonuses; **sparse-reward strategies** (curricula, shaping).
* **Reward design & integrity — Must:** potential-based shaping, avoid proxy gaming/reward hacking; **reward models** (pairwise preferences), label quality & drift monitoring, **sign & version rewards**.
* **Offline RL & logs — Must:** dataset curation & schema (episodes, masks, action spaces), dedupe, **coverage checks**; algorithms: **IQL, CQL, TD3+BC, AWAC**; penalties for out-of-distribution actions.
* **Off-policy evaluation (OPE) — Must:** IPS/SNIPS, Self-Normalized DR, Doubly Robust, **FQE**, fitted MDP; variance control & confidence intervals; guardrails for go/no-go.
* **Safety & constraints — Must:** constrained MDPs (Lagrangian), risk-sensitive RL (CVaR/percentile), **shields/validators**, action filters; **risk budgets & kill-switches**; catastrophe replay.
* **Multi-armed bandits — Must:** contextual bandits, uplift modeling, delayed feedback; serving guards & exploration budgets.
* **Multi-agent RL — Strong:** self-play, population-based training, credit assignment, CTDE (centralized training, decentralized execution).
* **Hierarchical & meta RL — Strong:** options framework, skills discovery; meta-gradients; model-based RL awareness (MPC, Dreamer).
* **Sim2Real & domain shift — Strong:** domain randomization, invariances, transfer; **stress tests under shift**.
* **Human feedback & alignment — Strong:** **RLHF/RLAIF**: preference data pipelines, reward model monitoring, KL-control; alternatives (DPO/RMs).
* **Productionization — Must:** vectorized envs, **environment registries** with signed versions, deterministic seeds/logs; **policy registries with signing**, canary & rollback; stateless serving (TorchScript/ONNX), **admission control for online learning**.
* **Use-cases**: recsys/ads (slates, budgets), pricing/auctions, operations (forecast → control), robotics/control, traffic/logistics, UI personalization, LLM tool-use controllers.
  **Stack:** Training: **RLlib, Stable-Baselines3, CleanRL, Acme, Tianshou, RLax/JAX, Brax, Isaac Gym/MuJoCo**, PettingZoo/MAgent (multi-agent). Offline/OPE: **d3rlpy, CORL, Reinforcement-Learning-Coach**, **FQE/Fitted Q** toolkits. Simulators: **Gymnasium**, **Brax**, **Isaac Gym**, **Unity/ML-Agents**, RecSim, **OpenSpiel**. Serving/ops: **Ray Serve**, FastAPI/gRPC, ONNX Runtime/TensorRT, Feast/feature stores for state, **Sigstore/cosign** for policy/reward signing.

## Tech Stack by Task (principles → examples)

### Languages & tooling

* **Principles:** typed, tested, reproducible; measure before optimizing.
* **Must:** Python, SQL, Bash; Git/GitHub; tests/lint/format/typing; packaging.
* **Strong:** C++/Rust/Go; Poetry/pip-tools; CI (GitHub Actions).
* **Awareness:** TypeScript for demos/frontends.

### Data & pipelines

* **Principles:** contracts & SLAs; idempotent backfills; handle **late/dup data**.
* **Must:** S3/GCS/ADLS; Parquet; Airflow/Prefect/Dagster; Kafka/Kinesis; data validation with **contract tests**.
* **Strong:** Delta/Iceberg; Spark/Databricks; lineage/catalogs (DataHub).
* **Awareness:** LakeFS/DVC, Monte Carlo.

### Modeling & training

* **Principles:** strong baselines; evals as code; reproducibility.
* **Must:** PyTorch; scikit-learn; XGBoost/LightGBM; experiment tracking (W\&B/MLflow).
* **Strong:** Lightning/Accelerate; DeepSpeed/FSDP; Optuna.
* **Awareness:** JAX (selective).

### **Reinforcement Learning** *(new core)*

* **Principles:** **offline-first + OPE**, safe exploration, **signed environment/policy/reward** versions, online learning behind strict gates.
* **Must:** PPO/DQN/SAC/TD3; **IQL/CQL/TD3+BC**; **IPS/DR/FQE**; vectorized envs; deterministic seeding; policy/experience registries.
* **Strong:** multi-agent (PettingZoo/OpenSpiel), hierarchical/options, population-based training, risk-sensitive (CVaR), simulators (Brax/Isaac/MuJoCo).
* **Awareness:** model-based RL (MPC/Dreamer), causal RL, offline-to-online warm-start strategies.

### GenAI / RAG / Agents

* **Principles:** **hybrid + rerank**; **prompt/index versioning + signing**; **citations**; safety gates.
* **Must:** FAISS/pgvector; Elastic/OpenSearch BM25 + kNN; rerankers; evaluation/guardrails; **judge hygiene in CI**.
* **Strong:** LangChain/LlamaIndex/LangGraph; Milvus/Weaviate/Pinecone; Redis caches.
* **Awareness:** AutoGen/CrewAI, Guidance.

### Serving & ops

* **Principles:** SLOs, canary/rollback, **model+prompt+index** registry, evals-in-prod, **signed artifacts**.
* **Must:** Docker; FastAPI/gRPC; basic K8s; model registry; Prometheus/Grafana/OTel; **Sigstore/cosign** verify.
* **Strong:** KServe/Seldon/BentoML; vLLM/TensorRT-LLM; Helm; TVM/ONNX Runtime.
* **Awareness:** Argo Workflows/ArgoCD.

### Security & privacy

* **Principles:** least privilege, threat model, attest & sign, monitor access.
* **Must:** Vault/Secrets Manager; IAM/OPA basics; PII redaction; **SLSA baseline**.
* **Strong:** DP/Opacus/TF-Privacy; Confidential VMs; **content provenance policy**.
* **Awareness:** private inference (PIR/FHE/MPC) patterns.

### Prototyping & UX

* **Principles:** fastest path to learn; instrument prototypes.
* **Must:** Streamlit/Gradio; Jupyter; matplotlib/Plotly.
* **Strong:** Next.js; product analytics.
* **Awareness:** design systems.

## “What Great Looks Like” — Behaviors & Proofs

* **Design doc:** problem framing, constraints, KPIs, risks, experiment plan, success/kill criteria.
* **Ablation table:** baseline → increments; cost & latency alongside quality.
* **Eval pack:** stratified offline sets + production slice monitors; golden questions; false-positive/negative catalog; **LLM-judge calibration notes & CIs**.
* **Repro kit:** `Makefile`/`nox`/`tox`, pinned env, data snapshot hash, seed logs.
* **Ops bundle:** Helm/K8s manifest; SLOs; alerts; **model+prompt+index** rollback plan.
* **Safety bundle:** threat model, PII map, red-team prompts, abuse mitigation, model/data cards, **DPIA & data residency**.
* **Cost view:** \$/req or \$/token over time; utilization dashboards; committed-use vs on-demand plan; **vendor fallback plan**.
* **Postmortems:** blameless write-ups with action items & owners.

## Pitfalls & Anti-Patterns

* Chasing SOTA without a baseline or cost guardrails.
* Leakage via time, joins, or label proxies.
* Unversioned data or mutable training sets; **unversioned/unsigned prompts/retrievers/indices**.
* “Works on my box” envs; non-deterministic seeds.
* LLM demos without **evals, citations, or safety gates**.
* Over-indexing on orchestration before proving value.
* Ignoring **per-slice** performance (fairness/long tail).
* Unbounded queues and no backpressure → cascading failures.
* No **DR drills**; no degrade modes for vendor limits/quotas.
* **Judges/rerankers** left unmonitored; no rollback for them.
* **RL-specific:** reward hacking via mis-specified proxies; unsafe exploration without risk budgets; deploying without **OPE**; non-stationary environments without drift guards; stale/off-policy data causing extrapolation error; unbounded online learning without kill-switch.

## Readiness Checklists (quick self-audit)

### Data & RAG

* [ ] PII & license audit complete; **data cards** exist; residency/DPIA done where needed.
* [ ] Retrieval: **hybrid (BM25+ANN) + rerank**; chunk strategy justified; dedupe & normalization done.
* [ ] Citations grounded to spans; reference resolver tested.
* [ ] **Prompt registry/hash/sign**; **retriever/index** versioned & signed; **joint rollback** plan.
* [ ] Freshness policy and cache invalidation defined; semantic/page caches sized & monitored.
* [ ] **Support-coverage, faithfulness\@k, abstention** tracked; CI fails on regression.

### Training

* [ ] AMP/bfloat16 where safe; dataloader utilization > 85%; no CPU bottleneck; tokenizer throughput measured.
* [ ] Checkpoints resumable; seeds fixed; metrics logged each step/epoch.
* [ ] PEFT chosen with memory math; quantization tested for accuracy deltas.
* [ ] Numerics guardrails (loss scaling, log-space ops, Kahan; **NaN sentinels**; **autodiff tests**).

### Serving & Reliability

* [ ] P50/P90/P99 tracked; admission control & circuit breakers.
* [ ] Shadow/canary before 100% rollout; **one-command rollback** for model+prompt+index(+judge/reranker).
* [ ] Observability includes **evals-in-prod**, per-slice alerts, and **cost per req/token**; **judge decision logs** with human audit sampling.
* [ ] **Backups & DR** for models/vectors/prompts/features; restore drill in last quarter.
* [ ] **Vendor fallback** configured; hard cost ceilings with auto cut-over.

### Security, Privacy, Compliance

* [ ] Threat model reviewed; tool calls sandboxed; SSRF guarded; **egress allowlists**.
* [ ] Secrets rotated; least-privilege IAM; **SBOM generated & verified**; **SLSA level** set.
* [ ] Artifacts **signed & verified** (Sigstore/cosign) in CI/CD.
* [ ] Red-team prompts pass; jailbreak filters tuned with feedback loops.
* [ ] Vendor SLAs/quotas monitored; **degrade modes** defined and tested.
* [ ] **Content provenance** detected/logged; policy applied when absent.

### Human Factors

* [ ] Response style guide; uncertainty/error messaging; consent/opt-out surfaced.
* [ ] Slice fairness alerts + mitigation playbook; reviewers trained; HITL UX measured.
* [ ] **Judge/reranker** disagreement rates monitored across slices.

### **Reinforcement Learning**

* [ ] **Environment registry** (signed, versioned) with seeds & determinism notes; sim fidelity documented; domain randomization plan.
* [ ] **Reward spec** reviewed; shaping is potential-based; reward model (if any) versioned/signed with label QA.
* [ ] **Offline dataset** coverage checks (state-action density, support); train/val/test splits by episode/time; leakage avoided.
* [ ] **OPE** implemented (IPS/DR/FQE) with CIs; go/no-go thresholds defined; variance controls in place.
* [ ] Rollout plan: **shadow/canary**, exploration budget, risk KPIs (safety violations, regret) + **kill-switch**.
* [ ] **Policy registry** (signed); one-command rollback; policy distillation/quantization tested for latency.
* [ ] Observability: return/regret, constraint violations, entropy, action distribution drift, reward-model drift if using RLHF.
* [ ] **Multi-agent**: credit assignment strategy; self-play stability; population diversity checks (if applicable).

## Interview Signals (strong filters) + **Scoring Rubric (100 pts)**

* **Outcomes & product sense (15):** KPI tree, kill criteria, cost/quality tradeoffs.
* **Data contracts & leakage defense (15):** PIT correctness, **contract tests**, audits.
* **Modeling fundamentals (10):** GBM + simple DL, calibration, per-slice metrics.
* **GenAI/RAG core (20):** hybrid + rerank + span-grounded citations, cache strategy, **signed** index/prompt versioning & rollback.
* **Eval discipline (10):** evals-as-code, **judge hygiene** (pairwise, anchors, CIs, human agreement).
* **Serving & SRE (10):** SLOs, canary/rollback, admission control, P90/P99 + cost.
* **Security & privacy (10):** threat model, SSRF/tool sandboxing, **SBOM + SLSA + signing**.
* **FinOps (5):** utilization, batching, KV cache, \$ per req/token dashboards, vendor fallback.
* **Leadership & docs (5):** ADRs, runbooks, postmortems.
* **Platform hygiene (10):** CI, tests, typed code, reproducible envs.
  **For RL-heavy roles:** swap “GenAI/RAG core (20)” with **“RL core (20)”**: MDP/POMDP, PPO/DQN/SAC, **OPE** (IPS/DR/FQE), offline→online ramp, exploration & risk budgets, policy/rew. signing, constraint monitoring & rollback.

**Signals to probe live:** Whiteboard→code custom loss or retrieval; implement eval harness with tests; spot two leakage vectors; sketch rollout/rollback & SLOs; enumerate injection vectors & mitigations; argue BM25+ANN+reranker and chunking trade-offs; show **prompt hashing/signing** and index versioning + joint rollback; **derive GAE**, set up **FQE** on a logged dataset, design a **safe canary** with risk KPIs and kill-switch.

## 90-Day Outcomes (by track)

**A) Applied ML (Product)**

* Ship **baseline → v1** with evals, runbook, and SLOs; achieve **≥ X%** KPI lift at **≤ \$Y/req**.
* Replace a naïve pipeline with **idempotent, backfillable** DAG + data quality checks.
* Stand up **per-slice alerts** (incl. fairness) tied to mitigation playbooks.

**B) ML Platform & Infra**

* Productionize **model+prompt+index** registry with **signing & verification**; enable **one-command rollbacks**.
* Land **observability** (metrics/logs/traces/**evals**) with P90/P99 & cost dashboards.
* Implement **DR** (backups/restores + drill) and **degrade modes** for vendor/API quotas.

**C) GenAI / RAG**

* Launch RAG with **hybrid + rerank + span citations** and measurable hallucination reduction; track **support-coverage/faithfulness**.
* Implement **PEFT finetune + quantized serving** with **≥ Y%** cost reduction at equal quality.
* Calibrated **LLM-as-judge** pipeline (pairwise, anchors, CI on win-rates) with **human agreement audits**.

**D) Reinforcement Learning**

* Stand up **offline RL** baseline (IQL/CQL/TD3+BC) on logged data with **OPE** and CIs; write reward spec & tests; sign artifacts.
* Ship **safe online canary**: exploration budget, constraint monitors (violations, regret), **kill-switch** + one-command rollback.
* Land **policy/experience/environment registry** with versions & signatures; dashboards: return/regret, entropy, drift; **distill/quantize** policy for latency.

## Minimum Viable **Musts** (90-day bar for a general Applied ML Engineer)

1. KPI tree + baseline before fancy.
2. Data contracts & **contract tests** (schema, ranges, drift, PII/licensing).
3. Repro kit + tracking (one-command runs; pinned data/env).
4. Leakage-proof CV + ablation habit.
5. GBM + simple DL proficiency; know tabular ≠ Transformer.
6. RAG basics: **hybrid + rerank + span citations**.
7. Prompting for **structured outputs** (JSON/schema) with robust parsing.
8. **Evals as code** (stratified sets, slice metrics, CI) with **judge hygiene**.
9. Serving 101: FastAPI/gRPC, canary, rollback, SLOs, cost per req/token.
10. Observability: traces/metrics/logs + **evals-in-prod** + per-slice alerts.
11. Security & privacy: threat model, secret mgmt, SSRF/tool sandboxing, **artifact signing**.
12. FinOps: utilization, batching, KV cache, **cost/quality dashboard** + vendor fallback.
    **For RL-heavy roles, add:**
13. **Offline-first RL** (IQL/CQL/TD3+BC) on logs + **OPE (IPS/DR/FQE)** with CIs.
14. **Safe online learning** behind gates: exploration & risk budgets, constraint monitors, **kill-switch & rollback**.
15. **Policy/Env/Reward versioning & signing**; deterministic seeds; drift & reward-hacking monitors.

## Repo-Ready Templates (drop-in)

* **/docs/design\_doc.md:** problem framing, constraints, KPIs, risks, experiment plan, success/kill criteria.
* **/docs/eval\_pack.md:** dataset versions, slice definitions, golden questions, judge prompts, CI gates, win-rate CIs, **human agreement audit plan**.
* **/ops/runbook.md:** deploy/canary/rollback, on-call, SLOs/alerts, DR & restore steps, degrade modes, **artifact verification (Sigstore)**.
* **/safety/safety\_checklist.md:** threat model, PII map, red-team prompts, jailbreak filters, DPIA/residency, **tool scopes & egress allowlists**.
* **/cost/cost\_dashboard\_spec.md:** metrics (\$/req, tokens/req, utilization), budgets, alerts, **vendor cut-over thresholds**.
* **/versioning/registries.md:** **model, prompt, retriever, index** schemas; hashing/**signing** rules; joint release gates & rollback.

## Appendices

### A) Licensing/IP Gotchas (quick reference)

* **CC-BY-SA**: derivative share-alike obligations—contaminate downstream.
* **Non-commercial** datasets in commercial use—**don’t**.
* **Field-of-use** restrictions on weights/models—respect or exclude.
* **Scraping T\&Cs**: robots.txt & contractual bans; ensure legal review.
* **Proof:** asset inventory, approvals, exclusion lists, and checks in CI.

### B) DR Drill Script (quarterly)

* Restore **model+prompt+index+features** from last snapshot.
* Rebuild search indices; verify embeddings/version hashes.
* Run smoke + evals; validate P90/P99 and win-rate/faithfulness deltas.
* Rotate keys; attest images; **verify signatures**; sign artifacts.

### C) Synthetic Data Governance (quick policy)

* Only use where it demonstrably improves coverage or safety; track provenance.
* Never train only-on-model-generated data without human-labeled anchors; flag in dataset cards.
* Monitor bias/drift amplification; include synthetic-vs-real slice metrics.
* Exclude synthetic-only items from gold test sets unless explicitly marked.

### D) Content Provenance & Watermarks (policy)

* Detect & log provenance (e.g., watermarks, C2PA) at ingestion & inference.
* Define trust thresholds and UI surfacing; degrade or abstain when provenance is required but missing.
* Store provenance with artifacts and outputs; include in audit exports.

### **E) RL Reward & Safety Quick Reference**

* **Reward spec:** is it potential-based? monotonic with the true objective? has counter-examples?
* **Hacking checks:** simulate exploits; adversarial rollouts; human sanity checks.
* **OPE thresholds:** define minimal uplift/regret bounds w/ CIs before online.
* **Safety:** constraint list, sensors, violation taxonomy; shields/action filters in serving.
* **Risk budgets:** max exploration %, violation rate ceilings, automatic rollback triggers.
* **Drift:** monitor state/action/reward distribution drift and reward-model calibration over time.
