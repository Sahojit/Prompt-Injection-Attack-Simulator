# 🔐 Prompt Injection Attack Simulator

A simple local LLM security project that simulates **prompt injection attacks** and evaluates how well a defense system can block them.

Built using local models via Ollama — no external APIs required.

---

# 🚀 What This Project Does

This system:

* Simulates prompt injection attacks (e.g., "ignore previous instructions")
* Sends them to a local LLM
* Applies a defense filter
* Measures how many attacks are blocked

👉 In short:
**Attack → Defense → Result → Evaluation**

---

# 🧱 Tech Stack

* Python
* FastAPI (backend)
* Streamlit (dashboard)
* Ollama (local LLM runtime)

---

# ⚙️ Setup (One-Time)

## 1. Install Ollama

Download and install Ollama.

Then run:

```bash
ollama pull llama3
ollama serve
```

Keep this running.

---

## 2. Clone the Repo

```bash
git clone https://github.com/Sahojit/Prompt-Injection-Attack-Simulator.git
cd Prompt-Injection-Attack-Simulator
```

---

## 3. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux

# Windows:
.venv\Scripts\activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements_simple.txt
```

---

# ▶️ Run the Project

## Start Backend

```bash
uvicorn src.api.main:app --reload
```

## Start Dashboard

```bash
streamlit run dashboard/app.py
```

---

# 🌐 Access the App

* API Docs → http://localhost:8000/docs
* Dashboard → http://localhost:8501

👉 Open the dashboard in browser to see:

* attacks
* blocked vs successful
* evaluation results

---

# 👥 How Teammates Can Use It

Each teammate should:

1. Install Ollama
2. Pull model:

   ```bash
   ollama pull llama3
   ```
3. Clone repo
4. Install requirements
5. Run backend + dashboard

⚠️ Important:

* Ollama must be running locally
* Ports 8000 and 8501 should be free

---

# 📊 Example Flow

1. System sends attack prompt
2. Defense checks it
3. If safe → goes to LLM
4. If malicious → blocked
5. Results are logged and shown on dashboard

---

# 🧠 Project Idea

This project demonstrates:

* How prompt injection attacks work
* How simple defenses can reduce risk
* How to measure LLM security

---

# ⚠️ Limitations

* Uses simple rule-based defense
* Not fully secure (for learning/demo purposes)
* Can be extended with ML-based detection

---

# 📌 Future Improvements

* Add ML-based classifier
* Improve detection rules
* Add more attack types

---

# 🧑‍💻 Author

Sahojit Karmakar
