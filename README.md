# Prompt Injection Attack Simulator for LLM Security

A production-grade local LLM security testing framework powered by **Ollama**. Simulates prompt injection attacks and evaluates defense mechanisms — no external APIs required.

---

## Quick Start

### 1. Prerequisites
```bash
# Install Ollama
brew install ollama   # macOS
# Then pull models:
ollama pull llama3
ollama pull mistral
ollama serve          # keep running in a terminal
```

### 2. Setup
```bash
cd "Prompt Injection Attack Simulator for LLM Security"
/opt/homebrew/bin/python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start
```bash
# Option A: use the startup script
bash scripts/start.sh

# Option B: manual
uvicorn src.api.main:app --reload &
streamlit run dashboard/app.py
```

- **API Docs:** http://localhost:8000/docs
- **Dashboard:** http://localhost:8501

---

## Architecture

```
src/
├── llm/
│   ├── ollama_client.py   # Ollama REST wrapper, retry, streaming
│   └── victim.py          # Victim LLM (3 system prompt modes)
├── attacks/
│   ├── attack_types.py    # Enums + AttackPrompt dataclass
│   ├── attack_library.py  # 16 curated attacks (4 types)
│   └── generator.py       # AttackGenerator + adversarial loop
├── defenses/
│   ├── base.py            # Abstract base + DefenseResult
│   ├── rule_based.py      # Regex + keyword detection
│   ├── prompt_engineering.py  # Sanitization + strict system prompt
│   ├── llm_guard.py       # Second-LLM classifier (risk 0–1)
│   └── orchestrator.py    # Runs all defenses, aggregates results
├── evaluation/
│   └── scorer.py          # ASR, F1, confusion matrix, model compare
├── logging/
│   └── db.py              # SQLite WAL, threadsafe, full query API
└── api/
    ├── main.py            # FastAPI app + CORS + lifespan
    ├── schemas.py         # Pydantic request/response models
    ├── routes.py          # All endpoints
    └── simulator.py       # Core pipeline: attack→defense→victim→eval→log
dashboard/
└── app.py                 # Streamlit 7-page dashboard
```

---

## Attack Types

| Type | Count | Examples |
|------|-------|---------|
| `instruction_override` | 4 | classic_ignore, DAN, role_switch |
| `jailbreak` | 5 | fictional_framing, base64 obfuscation |
| `data_exfiltration` | 4 | reveal_system_prompt, training_data_leak |
| `indirect_injection` | 4 | document_injection, webpage_injection |

---

## Defense Strategies

| Strategy | Method | Latency |
|----------|--------|---------|
| Rule-based | 12 regex patterns + 8 keyword blocklist | ~0ms |
| Prompt Engineering | Sanitization + strict system prompt | ~0ms |
| LLM Guard | Second Ollama model classifies prompt (0–1 risk) | ~1–5s |

---

## API Endpoints

```
GET  /api/v1/health                    # Ollama status
GET  /api/v1/models                    # Available local models
GET  /api/v1/attacks                   # Attack library listing
POST /api/v1/defense/evaluate          # Test a prompt against all defenses
POST /api/v1/simulate                  # Full simulation run
POST /api/v1/simulate/single           # Single attack test
POST /api/v1/simulate/adversarial-loop # Auto-mutate until success or exhaustion
POST /api/v1/simulate/compare-models   # llama3 vs mistral side-by-side
GET  /api/v1/logs                      # Query attack logs
GET  /api/v1/logs/runs                 # Run summaries
```

---

## Evaluation Metrics

- **Attack Success Rate (ASR)** — % attacks that got compliant responses
- **Defense Detection Rate** — % malicious prompts correctly blocked
- **False Positive Rate** — % benign prompts wrongly blocked
- **Precision / Recall / F1** — standard classification metrics
- **Risk Score** — 0–1 continuous score per defense strategy

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```
