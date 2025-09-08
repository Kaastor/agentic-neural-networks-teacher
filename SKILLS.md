# Applied AI Engineer (LLMs + SLMs + Agents + Vision)

*Focus: end‑to‑end ownership of LLM‑based applications (fine‑tuning, tool‑use/automation agents, retrieval), **Small Language Models (SLMs) incl. edge/on‑device**, and **Vision/Perception**. Build, evaluate, deploy, and operate systems that are **fast, safe, and cost‑effective**.*

## Legend

* **Must** = core for almost every applied role; demonstrably do this.
* **Strong** = frequently needed; expected for senior/lead or to own production.
* **Awareness** = know concepts/trade‑offs; ramp fast when needed.

> **Operating stance:** **Own my AI services end‑to‑end** (model/prompt/retriever/index/agents/UX), not company‑wide pipelines.

---

## Tracks & Leveling Matrix (calibrated for LLMs, SLMs & Vision)

Overlapping tracks share foundations but diverge in emphasis.

* **A) Applied ML (Product)**
* **B) Platform & Reliability (Light MLOps)**
* **C) LLMs / RAG / Agents (incl. SLMs)**
* **D) Vision & Perception (incl. Multimodal LMMs)**
* **E) Edge & On‑Device AI (SLMs & Vision)**
* **F) Search & Retrieval (IR for RAG)**

> Removed: RL and unrelated domains.

| Skill Area                             | L3 (Mid)                                                                                                                                                                                                                | L4 (Senior)                                                                                                                                               | L5 (Staff)                                                                                                                                           | L6 (Principal)                                                                                                           |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **A: Applied ML**                      | **Must:** data quality, **contract tests**, leakage‑free CV; GBM & basic DL; **evals‑as‑code**; KPI mapping.                                                                                                            | + **Strong:** experiment design; per‑slice metrics; online/offline metric linkage; guardrails.                                                            | + Leads cross‑team metric trees, launch guardrails, fairness/long‑tail playbooks.                                                                    | + Org‑wide standards; sets strategy/portfolio bets tied to KPIs.                                                         |
| **B: Platform & Reliability**          | **Must:** CI/CD, containers, FastAPI/gRPC, basic k8s, observability; **canary/rollback**; **artifact verification** via **Release & Provenance Policy**.                                                                | + **Strong:** cost dashboards, rate limits, SLOs/error budgets, multi‑model routing, multi‑vendor fallbacks.                                              | + Multi‑region patterns, DR drills, supply‑chain posture, encrypted stores, privacy budgets.                                                         | + Company‑wide reliability/safety strategy and audits.                                                                   |
| **C: LLMs/RAG/Agents (incl. SLMs)**    | **Must:** prompting, JSON/schema I/O, **hybrid (BM25+ANN) + rerank**, span‑grounded citations, **LLM‑judge hygiene in CI**, **prompt/retriever/index versioning**; tool‑use with approval gates; deterministic parsers. | + **Strong:** eval‑pack evolution; safe rollback; cost/latency/token SLOs; **judge/reranker monitoring**; cache taxonomy (output/semantic/embedding).     | + Leads SFT/PEFT/distillation; **SLM first** strategies; model/index/prompt fleet mgmt; joint rollback across **model+prompt+index(+judge/policy)**. | + Safety/governance programs; model/index fleet strategy; content provenance policy; org‑level budget & vendor strategy. |
| **D: Vision & Perception**             | **Must:** transfer learning (CNN/ViT), detection/segmentation basics, augmentations, OCR basics, **ONNX/TensorRT export**.                                                                                              | + **Strong:** PTQ/QAT, TensorRT optimization, video sampling & batching, multi‑task heads; VLMs (e.g., CLIP‑style) for retrieval.                         | + Leads real‑time perception stacks at scale; MLLM (vision‑language) integrations; multi‑camera QoS.                                                 | + Org‑wide perception roadmap & governance; camera/system standards and audits.                                          |
| **E: Edge & On‑Device AI**             | **Must:** model conversion (ONNX/Core ML/TFLite/gguf), PTQ, runtime basics (NNAPI/CoreML/Metal/WebGPU), offline modes; **acceptance tests** (e.g., p95 ≤ X ms on target device).                                        | + **Strong:** distillation & quantization trade‑offs (AWQ/GPTQ/GGUF/LLM.int8/4‑bit), KV‑cache mgmt on device; packaging & updates; privacy‑first logging. | + Leads edge program across platforms; telemetry & remote config; staged rollouts; secure model update/signing.                                      | + Org edge/SLM strategy; supplier silicon/NPU partnership input; field ops & compliance strategy.                        |
| **F: Search & Retrieval (IR for RAG)** | **Must:** BM25/Query‑Likelihood + embeddings, **two‑tower vs cross‑encoder**, ANN (HNSW/IVF), metrics (**Recall\@k/NDCG**) + **counterfactual IPS**; dedupe/normalize/chunking with overlap.                            | + **Strong:** hybrid tuning, reranker selection, exploration budgets, freshness policies, per‑slice retrieval metrics.                                    | + Leads retriever/reranker architectures, budgeted retrieval, versioning strategy & rollback; marketplace of indices across products.                | + Org discovery/search strategy; governance of data/indices & policy.                                                    |

**L4 → L5:** owns a **portfolio of services** with **joint rollback** across **model+prompt+index(+judge/policy)** and **multi‑vendor cut‑over drills**.

**L5 → L6:** sets org‑wide **trust & safety** posture (eval packs, audits, disclosure) and enforces via gates.

> **De‑emphasis:** Running company‑wide Airflow/Spark/feature stores is **not** part of this role.

---

## 0) Operating Principles (meta **Musts**)

* **Outcomes > models.** KPIs, user value, incremental shipping.
* **Data first.** Quality, lineage, governance, drift, cost.
* **Ruthless simplicity.** Baselines/ablations; cheapest thing that works.
* **Reproducibility.** One‑command runs (envs, seeds, data snapshots).
* **Observability.** Metrics/logs/traces/**evals** before scaling. (**RUM conditional** when you own a user surface.)
* **Safety & compliance by default.** PII, licensing, security posture, abuse cases.
* **FinOps.** Cost/latency SLOs; \$/req or \$/token.
* **Write it down.** Design docs, ADRs, runbooks, postmortems, model/data cards.
* **Version everything.** Data, code, **models, prompts, retrievers, indices, evals, policies**.
* **Provenance & integrity.** **Attest & verify** release artifacts; immutable IDs (`model@sha` etc.); audit trails.

**Proof bundle:** crisp design docs & experiment charters, ablations with cost/latency, runbooks, **cost dashboards**, postmortems, **registries with verified artifacts and provenance**.

---

## 1) Mathematical & ML Foundations

* **Linear algebra & calculus (optimization) — Must**
  Conditioning, optimizer schedules, normalization/initialization, FP16/bfloat16 guardrails.
  **Proof:** custom loss + gradient; autodiff correctness tests; NaN sentinels.

* **Probability & statistics — Must**
  Calibration, CIs vs significance, bootstrap, uncertainty surfacing.

* **Learning theory & generalization — Strong**
  Bias–variance, regularization, ablations (data/model/training).

* **Information & signal processing — Strong**
  Filtering, spectral methods, compression, sampling.

* **Time‑series & causality — Awareness → Strong as needed**
  Stationarity, leakage traps, proper CV, counterfactuals, **policy‑aware metrics**.

**Stack:** NumPy, SciPy, PyTorch, scikit‑learn, statsmodels, pymc.

---

## 2) Software Engineering Mastery

* **Language excellence (Python + SQL; TypeScript for tools/UI — Awareness) — Must**
  Type hints, asyncio, vectorization; SQL window functions/CTEs; schema contracts.

* **Code quality & scale — Must**
  Clean architecture, packaging, profiling; **Proof:** flamegraphs & speedups.

* **Concurrency & systems — Strong**
  Async vs threads vs procs; queues/backpressure; idempotency; graceful shutdown.

* **Tooling & hygiene — Must**
  Tests (unit/integration/property), lint/format, typing gates, CI, deterministic builds, **SBOM awareness**.

**Determinism & Repro Pack — Strong**
Seed control, numeric parity checks, tokenizer/version‑skew tests, quantized‑vs‑FP regression harness, GPU/CPU parity checks in CI.

---

## 3) Data Engineering & DataOps (AAE‑lite)

* **Ingestion & storage — Must**
  Data contracts; Parquet/Avro; point‑in‑time correctness; simple batch/cron.

* **Streaming / near‑real‑time — Awareness → Strong as needed**
  Kafka/PubSub basics; idempotent consumers; backfills & replay.

* **Pipelines — Awareness**
  Idempotency, retries, lineage; prefer managed services over platform building.

* **Quality & governance — Must**
  Schema/range/distribution checks; drift monitors; PII discovery/redaction; **contract tests that fail CI**.

* **Labeling & supervision — Strong**
  Active/weak supervision, HITL, programmatic labeling; **IAA & adjudication**; golden‑set curation.

---

## 4) Classical ML Bread‑and‑Butter

* **Algorithms — Must:** linear/logistic, trees/GBMs, SVMs, clustering, naïve Bayes; know when GBM beats a Transformer for tabular.
* **Modeling habits — Must:** robust CV, leakage prevention, class imbalance, calibration.
  **Stack:** scikit‑learn, XGBoost/LightGBM/CatBoost, Optuna.

---

## 5) Deep Learning Architectures (Vision & NLP)

* **Vision — Must:** CNNs, ViTs; detection/segmentation; augmentations; small‑data transfer; metric learning (contrastive/CLIP‑style).
* **NLP/LLMs — Must:** tokenization (SentencePiece/BPE/unigram), Transformers, embeddings, IR/ranking; summarization/QA/NER; **tokenizer throughput & prompt‑length budgeting**.
* **Multimodal/VLM/MLLM — Strong:** CLIP/SigLIP‑style encoders, image‑text contrastive; BLIP/LLaVA‑style adapters; grounding for RAG.
* **Speech/audio — Awareness;** **Time‑series DL — Awareness.**

**Stack:** PyTorch (core), HF Transformers/Diffusers, Lightning/Accelerate, TorchScript/ONNX.

---

## 6) Training, Scaling & Efficiency

* **Performance training — Must:** AMP/bfloat16, clipping, schedulers, checkpointing, dataloader throughput.
* **PEFT & compression — Must:** LoRA/QLoRA/PEFT; distillation; pruning/sparsity; memory–latency trade‑offs.
* **Distributed training — Awareness → Strong as needed:** DDP/FSDP/ZeRO; failure recovery; gradient checkpointing.
* **Schedulers & clusters — Awareness:** SLURM/k8s jobs; preemption‑safe; spot‑aware.
* **Cost awareness — Must:** utilization, bandwidth, pricing; **\$ per achieved metric** dashboards.

> **Accelerators scope:** Triton‑lang/kernels = **Awareness**. Prioritize **ONNX/TensorRT**, **vLLM/TensorRT‑LLM**, profiler literacy as **Strong**.

---

## 7) LLMs, RAG & Agents (Applied GenAI & SLMs)

**Principles:** retrieval before fine‑tune; **hybrid BM25+ANN + rerank**; span‑grounded citations; judges/rerankers are first‑class; **adhere to Release & Provenance Policy**; **SLM‑first** where feasible; privacy by design.

* **LLM lifecycle — Strong:** SFT vs RLHF/RLAIF/DPO (awareness of methods); safety tuning; eval gates; license & contamination hygiene.
* **Prompting craft — Must:** roles, few‑shot/CoT, tool use, routing; JSON/Schema outputs; deterministic parsing; **prompt registry + hashing/verification**.
* **Secrets & context hygiene — Must:** secret‑scoped prompts; context‑leak controls; **privacy budgets** for logs.
* **RAG — Must:** chunking (with overlap), **hybrid (BM25+ANN)**, reranking; embeddings hygiene; **cache taxonomy** (output/semantic/page/embedding) with eviction/validation; **retriever & index versioning**; **joint rollback**.
* **RAG metrics — Must:** **support‑coverage**, **faithfulness\@k**, **answerable vs unanswerable + abstention**; CI regression gates.
* **Judges & rerankers — Must:** versions, **slice drift monitoring**, rollback plans, cost/latency SLOs; log judge rationales.
* **Agents & orchestration — Strong:** planning, **approval gates**, **tool scopes**, state/memory; idempotent retries, DLQs; sandboxing; human‑in‑the‑loop controls.
* **Efficient inference — Must:** KV‑cache mgmt, paged attention, quantization (AWQ/GPTQ/GGUF/LLM.int8), speculative decoding, batching/streaming; **cost/latency SLOs**.
* **SLM specialization — Strong:** model zoo & licenses (3–7B), **CPU & single‑GPU serving** (llama.cpp/Ollama vs vLLM/TGI), GGUF/AWQ/GPTQ trade‑offs; tokenizer throughput; rope/pos‑scaling basics.

**Stack:** FAISS/HNSWlib/ScaNN, pgvector/Milvus/Weaviate/Pinecone; Elastic kNN; LangChain/LlamaIndex/LangGraph; vLLM/TensorRT‑LLM/TGI/llama.cpp/Ollama; FastAPI/gRPC; Redis; Ragas/Giskard/DeepEval/promptfoo; OpenTelemetry.

---

## 8) Vision Inference, Serving & Multimodal

* **Export & runtimes — Must:** TorchScript/ONNX; TensorRT; OpenVINO as needed; batch/stream video; pre/post‑processing correctness.
* **Optimization — Strong:** PTQ/QAT, layer fusion, mixed precision, NMS/ROI ops; CPU offload strategies.
* **Serving — Must:** REST/gRPC; admission control; rate limits; streaming for camera feeds; GPU affinity/NUMA; multi‑camera batching; real‑time QoS.
* **Vision‑RAG — Strong:** image embeddings (CLIP/SigLIP); hybrid retrieval; cross‑modal reranking; grounded captions.
* **MLLMs — Strong:** image‑text adapters; prompt templates for visual QA; citation spans with coordinates.

---

## 9) Inference, Serving & Reliability (End‑to‑End Ownership)

* **APIs & services — Must:** streaming, admission control, rate limiting, retries/timeouts/circuit breakers; schema contracts.
* **Client→Server latency & UX — Strong (conditional):** TTFT/TBT budgeting; **RUM** when you own the UI; streaming UX; a11y; error/uncertainty messaging.
* **Optimization — Strong:** ONNX/AOT, fusion, pinned memory/NUMA, KV paging, tokenizer throughput.
* **Rollouts — Must:** shadow/canary/blue‑green; feature flags; **model+prompt+index(+judge/policy)** registry integration; one‑command rollback.
* **Observability — Must:** traces/metrics/logs; golden signals; token/req cost; **evals in prod**; data slices & drift; **judge/policy decision logs** with sampled human audits.
* **Capacity & cost — Must:** autoscaling, throughput models (Little’s Law), batching, queueing; P90/P99 budgets; SLOs & error budgets.
* **DR & continuity — Must:** backup/restore for **models, vectors, prompts, policies**; **restore drills**; key rotation; degrade modes.
* **Vendor dependency — Must:** quotas/price alarms; **multi‑vendor abstraction**; automatic cut‑over within hard cost ceilings.

---

## 10) Experimentation, Evaluation & Product Analytics

* **Instrumentation & analytics — Must:** event schemas, stable IDs, sampling plans, retention/cohort analyses, guardrail metrics; novelty effects & long‑horizon backtests; **data storytelling** (one killer chart tying latency/cost/quality to a KPI).
* **Experiment design — Strong:** metrics trees, traffic allocation, power/duration, sequential/non‑inferiority tests; ramp policies.
* **Model eval — Must:** stratified sets, per‑slice metrics, calibration, **fairness & robustness**; versioned datasets; CI gates.
* **A/B & beyond — Strong:** CUPED/CUPAC, long‑tail monitoring, novelty decay.
* **LLM‑judge hygiene — Must:** judge calibration, position‑bias mitigation, **human agreement checks**, **CIs on win‑rates**; **log judge rationales** for audit.

---

## 11) Security, Privacy, Compliance & Safety

* **Security for ML systems — Must:** STRIDE + ML‑specific risks; supply chain (**SBOM**, **SLSA targets**), **artifact verification**; prompt injection/tool SSRF; sandboxing; **egress allowlists**.
* **Abuse/fraud & multi‑tenant safeguards — Must:** rate‑limit evasion patterns, token‑abuse detection, tenant isolation tests.
* **Privacy — Must:** PII discovery/redaction, data minimization, retention; DP basics; federated patterns; private inference awareness (TEEs/PIR/FHE/MPC = **Awareness**); **privacy budgets for logs**.
* **Governance & standards — Must:** model/dataset cards, approvals & audit trails, RBAC; red‑teaming; incident response; **EU AI Act** risk tiers & obligations; **NIST AI RMF / ISO 23894** (**Awareness**).
* **Licensing/IP hygiene — Must:** CC‑BY‑SA share‑alike, non‑commercial datasets, field‑of‑use limits, scraping T\&Cs; **proof via license inventory + CI checks**.
* **Content provenance — Strong:** detect/log provenance; trust policy; surface to users when relevant.

---

## 12) Product, UX & Human Factors (for Applied ICs)

* **Problem framing — Must:** constraints → objective; success metrics; risks; buy vs build; de‑scope paths.
* **HITL — Must:** reviewer UX, escalation, feedback capture; queue design; incentives.
* **Explainability & trust — Strong:** global/local explainers, uncertainty surfacing, **citations in RAG**; clear error/uncertainty messaging.
* **Change management — Strong:** on‑call, postmortems; versioned releases.
* **UX writing & style — Must:** response tone guidelines, opt‑outs/consent surfaces; streaming UX patterns.
* **Fairness in production — Must:** **alerts on slice regressions**, mitigation playbooks, human review loops.
* **Judge/feedback observability — Must:** disagreement rates, slice drift; human audits; rollback plan.

---

## 13) Cloud, Hardware & Edge

* **Cloud — Must:** IAM least‑privilege, VPCs/subnets, SGs, storage classes/lifecycle, quotas; cost allocation/chargeback.
* **Accelerators — Strong:** GPU/TPU fundamentals, utilization vs latency; **TensorRT‑LLM/vLLM/ONNX Runtime**; memory hygiene; pinned memory.
* **Edge/mobile — Strong:** model conversion (ONNX → TFLite/Core ML; HF → gguf for llama.cpp/Ollama), PTQ/QAT; thermals & throttling; offline modes; **acceptance tests** ("p95 ≤ X ms on target device"); update/signing protocol; hardware delegates (NNAPI/Metal/ANE/NPU); WebGPU/WebNN.

---

## 14) Collaboration & Leadership (IC track)

* **Technical leadership — Strong:** roadmaps, ADRs, design reviews, mentoring, risk burndown; cross‑team alignment.
* **Cross‑functional fluency — Must:** product/data/platform/legal/security; shared SLOs.
* **Documentation — Must:** READMEs, runbooks, model/data cards, diagrams.

---

## Tech Stack by Task (principles → examples)

**Languages & Tooling**

* **Must:** Python, SQL, Bash; Git/GitHub; tests/lint/format/typing; packaging.
* **Strong:** C++/Rust/Go (perf); Poetry/pip‑tools; GitHub Actions.
* **Awareness:** TypeScript/Next.js for agent UIs.

**Data & Pipelines (minimal)**

* **Must:** S3/GCS/ADLS; Parquet; light schedulers with **contract tests**.
* **Strong:** dbt; Great Expectations/Soda.
* **Awareness:** Kafka/Kinesis/PubSub; CDC; Airflow/Prefect/Dagster.

**Modeling & Training**

* **Must:** PyTorch; scikit‑learn; XGBoost/LightGBM; MLflow/W\&B.
* **Strong:** Lightning/Accelerate; Optuna; FSDP/DeepSpeed (when scale demands).
* **Awareness:** JAX.

**GenAI / RAG / Agents**

* **Must:** **hybrid + rerank**, span citations; evals/guardrails; **judge hygiene**; **secret‑scoped prompts**; **cache taxonomy**.
* **Strong:** LangChain/LlamaIndex/LangGraph; Milvus/Weaviate/Pinecone; Redis; streaming UX.
* **Awareness:** CrewAI/AutoGen/Guidance.

**Serving & Ops**

* **Must:** Docker; FastAPI/gRPC; basic k8s; **registry for model/prompt/retriever/index/policy**; Prometheus/Grafana/OTel; **artifact verification** per **Release & Provenance Policy**.
* **Strong:** vLLM/TensorRT‑LLM; TVM/ONNX Runtime; **RUM** when applicable.
* **Awareness:** KServe/BentoML/Seldon; Helm; Argo.

**Edge & SLMs**

* **Must:** llama.cpp/Ollama; gguf conversion; TFLite/Core ML; ONNX Runtime Mobile; quantization (int8/int4).
* **Strong:** speculative decoding with draft models; KV‑cache residency tuning; tokenizer throughput fixes; CPU affinity; WebGPU.
* **Awareness:** MLC‑LLM, ExecuTorch.

**Security & Privacy**

* **Must:** Vault/Secrets Manager; IAM/OPA basics; PII redaction; **SLSA baseline**; **privacy budgets** for logs; abuse/fraud protections; tenant isolation.
* **Strong:** DP/Opacus; Confidential VMs; content provenance policy.

**Prototyping & UX**

* **Must:** Jupyter; Streamlit/Gradio; matplotlib/Plotly (data storytelling).
* **Strong:** Product analytics & (conditional) RUM dashboards.