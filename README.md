# 🤖 AI Email Automation SLM (Small Language Model)

A 100% private, self-hosted, lightweight **Small Language Model (SLM)** designed specifically for **Autonomous Email Automation, Triage & Professional Drafting**.

---

## 🌟 Why an SLM instead of Giant Models / Third-Party APIs?
1. **Zero Data Leakage**: Emails containing sensitive financial data, customer details, and invoices never leave your private server.
2. **Predictable & Fixed Cost**: Zero token bills. Run unlimited email inference.
3. **Domain Specialization**: Outperforms generic LLMs by strictly outputting structured triage JSON and contextual business replies.
4. **Ultra-Low Latency**: Fast local inference on CPU or modest consumer GPUs.

---

## 🏗️ Project Architecture

```
AI-automation/
├── data/
│   ├── dataset_generator.py      # Generates synthetic & real email triage training pairs
│   └── email_slm_dataset.jsonl   # Formatted instruction-tuning dataset
│
├── model/
│   ├── config.py                 # Hyperparameters (d_model, n_layers, n_heads, etc.)
│   ├── tokenizer.py              # Custom BPE/Byte-fallback email tokenizer
│   └── slm_transformer.py       # Modern Decoder-Only Transformer (RoPE + RMSNorm + SwiGLU)
│
├── training/
│   ├── train_slm.py             # PyTorch training pipeline for custom SLM
│   └── finetune_sota_slm.py     # LoRA/PEFT pipeline for Qwen-2.5-1.5B / Llama-3.2-1B
│
├── engine/
│   └── email_engine.py          # Production inference pipeline (Triage + Reply generator)
│
├── run_demo.py                  # Interactive CLI to train, test, and run live emails
├── requirements.txt             # Dependencies
└── README.md
```

---

## ⚡ Quick Start

### 1. Generate Dataset
```bash
python data/dataset_generator.py
```

### 2. Train the Custom SLM (Local CPU / GPU)
```bash
python training/train_slm.py
```
*This will train the lightweight Transformer on the instruction pairs and save the best checkpoint to `checkpoints/best_email_slm.pt`.*

### 3. Run Interactive Demo
```bash
python run_demo.py
```
*You can pick sample emails (Billing disputes, Outages, Enterprise leads) or paste your own emails to inspect real-time triage and draft generation.*

---

## 🚀 Scaling to Production: Fine-Tuning SOTA 1B-3B SLMs
If you want industrial-grade generation (like **Qwen 2.5 1.5B** or **Llama 3.2 1B**):
1. Open Google Colab or RunPod.
2. Run `python training/finetune_sota_slm.py`.
3. Export the LoRA adapter or merge weights into GGUF format for deployment with **Ollama** or **vLLM**.

---

## 📬 Next Step: Connecting to Gmail / Outlook
Once the model is running:
- Use `google-api-python-client` with Gmail API.
- On receiving new emails via Webhook, pass the message text to `EmailAutomationEngine.process_email()`.
- Automatically save the generated response as a **Draft** in the user's inbox!
