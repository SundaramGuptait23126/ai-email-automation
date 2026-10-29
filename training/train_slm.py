"""
Multi-Core Accelerated Training Pipeline for Custom Email SLM.
Trained on the bulk multi-sector dataset (2,500+ pairs).
Optimized for 16-Core CPU parallel execution.
"""

import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import torch
from torch.utils.data import Dataset, DataLoader
from model.config import SLMConfig
from model.tokenizer import EmailTokenizer
from model.slm_transformer import EmailSLM

DATA_PATH = PROJECT_ROOT / "data" / "email_slm_dataset.jsonl"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Maximize multi-core CPU utilization
num_threads = max(1, (os.cpu_count() or 4) - 2)
torch.set_num_threads(num_threads)

class EmailInstructionDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer: EmailTokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                prompt = (
                    f"<|instruction|>\n{item['instruction']}\n"
                    f"<|input|>\n{item['input']}\n"
                    f"<|output|>\n{item['output']}\n<|eos|>"
                )
                tokens = tokenizer.encode(prompt)
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        return torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch, pad_id=0):
    max_len = max(len(x) for x in batch)
    padded = []
    for x in batch:
        pad_len = max_len - len(x)
        p = torch.cat([x, torch.full((pad_len,), pad_id, dtype=torch.long)])
        padded.append(p)
    batch_tensor = torch.stack(padded)
    return batch_tensor, batch_tensor.clone()

def train(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 4e-4,
    d_model: int = 192,
    n_layers: int = 4,
    n_heads: int = 6,
    vocab_size_target: int = 4096
):
    print("=" * 65)
    print("🚀 MULTI-CORE ACCELERATED CUSTOM EMAIL SLM TRAINING")
    print("=" * 65)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device} | Active CPU Threads: {num_threads}")

    # 1. Build Tokenizer from expanded dataset
    print("\n[Step 1/3] Building Tokenizer from 2,500+ sample corpus...")
    tokenizer = EmailTokenizer()
    texts_for_vocab = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts_for_vocab.append(d["instruction"] + " " + d["input"] + " " + d["output"])
    tokenizer.build_vocab_from_texts(texts_for_vocab, target_vocab_size=vocab_size_target)
    tokenizer.save(str(CHECKPOINT_DIR / "tokenizer.json"))
    print(f"✅ Tokenizer compiled! Vocabulary: {tokenizer.vocab_size} tokens")

    # 2. Configure Model Architecture
    config = SLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_model * 3,
        max_seq_len=512,
        dropout=0.05
    )

    model = EmailSLM(config).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Step 2/3] Initialized Transformer: {param_count / 1e6:.2f} Million Parameters")

    # 3. Prepare DataLoader
    dataset = EmailInstructionDataset(DATA_PATH, tokenizer, max_seq_len=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    print(f"Total Batches per Epoch: {len(dataloader)} (Batch size: {batch_size})")

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    # 5. Training Loop
    print("\n[Step 3/3] Training In Progress...")
    start_time = time.time()
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0.0
        model.train()

        for step, (inputs, targets) in enumerate(dataloader, 1):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            logits, loss = model(inputs, targets=targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if step % 75 == 0 or step == len(dataloader):
                cur_loss = total_loss / step
                print(f"  Epoch [{epoch:02d}/{epochs:02d}] Step [{step:03d}/{len(dataloader)}] - Loss: {cur_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.6f}")

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        print(f"==> Epoch [{epoch:02d}/{epochs:02d}] Completed in {epoch_time:.1f}s | Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "vocab_size": tokenizer.vocab_size
            }, CHECKPOINT_DIR / "best_email_slm.pt")
            print(f"  ⭐ New Best Checkpoint Saved! (Loss: {best_loss:.4f})")

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"🎉 TRAINING COMPLETE in {total_time:.1f} seconds! Best Loss: {best_loss:.4f}")
    print(f"Model Checkpoint saved to: {CHECKPOINT_DIR / 'best_email_slm.pt'}")
    print("=" * 65)

    return model, tokenizer

if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    train(epochs=epochs, batch_size=8)
