"""
complaint_prioritizer/model_core.py  (v2 — matches your actual dataset)

Schema detected from your data:
  Columns      : Complaint_Narrative, Product, Sector, Priority_Level, Urgency_Score
  Priority vals: Low (0) | Medium (1) | High (2)   ← 3 classes
  Urgency range: 1.0 – 10.0                         ← normalised to 0–1 inside
  Sectors seen : Aviation, Food & Beverage, Banking, Retail, Tech & Telecom
  Products seen: Zomato, AirIndia, SpiceJet, IndiGo, SBI, etc.

Key change from your old model:
  OLD → sector/product were baked-in category IDs  → breaks on new sectors
  NEW → sector/product injected as free English text → works on ANY sector/product
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, mean_absolute_error
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

class Config:
    BASE_MODEL      = "bert-base-uncased"   # or "distilbert-base-uncased" (faster)
    MAX_LEN         = 256
    BATCH_SIZE      = 32
    EPOCHS          = 5
    LR              = 2e-5
    WARMUP_RATIO    = 0.1
    DROPOUT         = 0.3
    NUM_CLASSES     = 3          # Low=0, Medium=1, High=2
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

    # Joint loss weights
    PRIORITY_LOSS_W = 0.6
    URGENCY_LOSS_W  = 0.4

    # Label maps (matching YOUR dataset exactly)
    LABEL2IDX = {"Low": 0, "Medium": 1, "High": 2}
    IDX2LABEL = {0: "Low", 1: "Medium", 2: "High"}


# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORMATTER  ← the core generalisation trick
# ─────────────────────────────────────────────────────────────────────────────

def format_input(sector: str, product: str, complaint_text: str) -> str:
    """
    Converts the three raw fields into one natural-language string.

    This mirrors the `combined_text` column you already built in your
    train/test CSVs, but with a cleaner separator style and lowercased
    sector/product so the model handles casing variants gracefully.

    Your old combined_text:
        "[Sector: Retail] [Product: Amazon Headphones] Complaint: ..."

    New format (same idea, more sentence-like for BERT):
        "Complaint context: sector is retail, product is amazon headphones.
         Customer issue: ..."

    WHY THIS GENERALISES:
        BERT was pre-trained on billions of English sentences.
        It already understands "sector is aerospace" even if you never
        trained on aerospace — because it knows what "aerospace" means.
        Categorical IDs (sector_id=3) carry zero such prior knowledge.
    """
    sector  = str(sector).strip().lower()
    product = str(product).strip().lower()
    text    = str(complaint_text).strip()
    return (
        f"Complaint context: sector is {sector}, product is {product}. "
        f"Customer issue: {text}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ComplaintDataset(Dataset):
    """
    Reads your CSVs with columns:
        Complaint_Narrative | Product | Sector | Priority_Level | Urgency_Score

    Urgency_Score is on 1–10 scale in your data → normalised to 0–1 here.
    Priority_Level strings → integer indices (Low=0, Medium=1, High=2).
    """

    def __init__(self, df: pd.DataFrame, tokenizer, config: Config,
                 is_inference: bool = False):
        self.tokenizer    = tokenizer
        self.config       = config
        self.is_inference = is_inference

        self.texts = [
            format_input(row["Sector"], row["Product"], row["Complaint_Narrative"])
            for _, row in df.iterrows()
        ]

        if not is_inference:
            self.priorities = df["Priority_Level"].map(config.LABEL2IDX).values.astype(int)
            # Normalise urgency from 1–10 → 0–1
            self.urgencies  = ((df["Urgency_Score"].values - 1) / 9.0).astype(float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.config.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        if not self.is_inference:
            item["priority"] = torch.tensor(self.priorities[idx], dtype=torch.long)
            item["urgency"]  = torch.tensor(self.urgencies[idx],  dtype=torch.float)
        return item


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  — single unified model replacing your two separate models
# ─────────────────────────────────────────────────────────────────────────────

class ComplaintPrioritizer(nn.Module):
    """
    Replaces BOTH your old models (my_bert_model_final + my_urgency_model_final)
    with one unified dual-head model.

    Architecture:
        BERT encoder
            ↓
        Shared projection (768 → 512)
           ↙              ↘
    Priority head       Urgency head
    (3-class softmax)   (sigmoid → 0–1)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config  = config
        self.encoder = AutoModel.from_pretrained(config.BASE_MODEL)
        H = self.encoder.config.hidden_size   # 768 for bert-base

        self.shared = nn.Sequential(
            nn.Linear(H, 512),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
        )
        self.priority_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(256, config.NUM_CLASSES),   # 3 classes
        )
        self.urgency_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 1),
            nn.Sigmoid(),                          # → 0–1
        )

    def forward(self, input_ids, attention_mask):
        out     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls     = out.last_hidden_state[:, 0, :]
        shared  = self.shared(cls)
        return self.priority_head(shared), self.urgency_head(shared).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, model: ComplaintPrioritizer, config: Config):
        self.model  = model.to(config.DEVICE)
        self.config = config
        self.ce     = nn.CrossEntropyLoss()
        self.mse    = nn.MSELoss()

    def _loss(self, p_logits, u_pred, p_true, u_true):
        return (self.config.PRIORITY_LOSS_W * self.ce(p_logits, p_true) +
                self.config.URGENCY_LOSS_W  * self.mse(u_pred, u_true))

    def train(self, train_loader, val_loader, save_dir="best_model"):
        optimizer = AdamW(self.model.parameters(), lr=self.config.LR, weight_decay=0.01)
        total     = len(train_loader) * self.config.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total * self.config.WARMUP_RATIO), total)

        best_val_loss = float("inf")
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, self.config.EPOCHS + 1):
            self.model.train()
            running = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                ids  = batch["input_ids"].to(self.config.DEVICE)
                mask = batch["attention_mask"].to(self.config.DEVICE)
                pt   = batch["priority"].to(self.config.DEVICE)
                ut   = batch["urgency"].to(self.config.DEVICE)
                logits, upred = self.model(ids, mask)
                loss = self._loss(logits, upred, pt, ut)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                running += loss.item()

            val_loss, metrics = self.evaluate(val_loader)
            print(f"Epoch {epoch}/{self.config.EPOCHS} | "
                  f"Train Loss: {running/len(train_loader):.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Priority Acc: {metrics['acc']:.3f} | "
                  f"Urgency MAE: {metrics['mae']:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{save_dir}/model.pt")
                print("  ✓ Saved best model")

    def evaluate(self, loader):
        self.model.eval()
        total, pp, pt_all, up, ut_all = 0.0, [], [], [], []
        with torch.no_grad():
            for batch in loader:
                ids  = batch["input_ids"].to(self.config.DEVICE)
                mask = batch["attention_mask"].to(self.config.DEVICE)
                pt   = batch["priority"].to(self.config.DEVICE)
                ut   = batch["urgency"].to(self.config.DEVICE)
                logits, upred = self.model(ids, mask)
                total += self._loss(logits, upred, pt, ut).item()
                pp.extend(torch.argmax(logits, 1).cpu().numpy())
                pt_all.extend(pt.cpu().numpy())
                up.extend(upred.cpu().numpy())
                ut_all.extend(ut.cpu().numpy())

        acc = np.mean(np.array(pp) == np.array(pt_all))
        mae = mean_absolute_error(ut_all, up)
        return total / len(loader), {"acc": acc, "mae": mae, "preds": pp, "trues": pt_all}


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE API
# ─────────────────────────────────────────────────────────────────────────────

class ComplaintAnalyzer:
    """
    Production inference class.
    Works with ANY sector + product at inference time.

    Urgency output is on the original 1–10 scale (matching your dataset).
    """

    def __init__(self, model_dir: str, config: Optional[Config] = None):
        self.config    = config or Config()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = ComplaintPrioritizer(self.config)
        self.model.load_state_dict(
            torch.load(f"{model_dir}/model.pt", map_location=self.config.DEVICE))
        self.model.to(self.config.DEVICE).eval()

    def predict_single(self, sector: str, product: str, complaint_text: str) -> dict:
        text = format_input(sector, product, complaint_text)
        enc  = self.tokenizer(
            text, max_length=self.config.MAX_LEN,
            padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits, urgency = self.model(
                enc["input_ids"].to(self.config.DEVICE),
                enc["attention_mask"].to(self.config.DEVICE))
        idx        = torch.argmax(logits, 1).item()
        confidence = torch.softmax(logits, 1).max().item()
        # De-normalise urgency back to 1–10 scale
        urgency_10 = round(urgency.item() * 9 + 1, 2)

        return {
            "priority_label": self.config.IDX2LABEL[idx],
            "urgency_score":  urgency_10,          # 1–10 like your original data
            "confidence":     round(confidence, 4),
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input df must have: Sector, Product, Complaint_Narrative
        Returns df with added columns: priority_label, urgency_score, confidence
        Sorted by urgency_score descending (most urgent first).
        """
        results = [
            self.predict_single(r["Sector"], r["Product"], r["Complaint_Narrative"])
            for _, r in df.iterrows()
        ]
        out = pd.concat([
            df.reset_index(drop=True),
            pd.DataFrame(results)
        ], axis=1)
        return out.sort_values("urgency_score", ascending=False).reset_index(drop=True)
