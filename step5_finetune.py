"""
STEP 5 (Optional) — step5_finetune.py
======================================
Fine-tune the trained model on a specific company's labeled complaints.

When to use:
    - A company gives you 200–500 of their own labeled complaints
    - You want higher accuracy specifically for their sector/products
    - Without affecting the base model's generalization

Even 200 samples is enough — the BERT backbone already understands
language. You're just calibrating the decision boundary.

Usage:
    python step5_finetune.py \
        --company_data company_labeled.csv \
        --output company_model \
        --freeze_bert          # recommended for < 500 samples

Required columns in company_labeled.csv:
    Sector | Product | Complaint_Narrative | Priority_Level | Urgency_Score
"""

import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from model_core import Config, ComplaintDataset, ComplaintPrioritizer, Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",   default="best_model")
    p.add_argument("--company_data", required=True)
    p.add_argument("--output",       default="company_model")
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch",        type=int,   default=16)
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--freeze_bert",  action="store_true",
                   help="Freeze BERT backbone, train only heads. "
                        "Use when company data < 500 samples.")
    return p.parse_args()


def main():
    args   = parse_args()
    config = Config()
    config.EPOCHS     = args.epochs
    config.BATCH_SIZE = args.batch
    config.LR         = args.lr

    df = pd.read_csv(args.company_data)
    print(f"Company data: {len(df)} rows")
    print(f"Sectors  : {df['Sector'].unique().tolist()}")
    print(f"Products : {df['Product'].unique().tolist()}")
    print(f"Priority : {df['Priority_Level'].value_counts().to_dict()}")

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["Priority_Level"], random_state=42)

    tokenizer    = AutoTokenizer.from_pretrained(args.base_model)
    train_ds     = ComplaintDataset(train_df, tokenizer, config)
    val_ds       = ComplaintDataset(val_df,   tokenizer, config)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE)

    # Load base model weights
    model = ComplaintPrioritizer(config)
    model.load_state_dict(
        torch.load(f"{args.base_model}/model.pt", map_location=config.DEVICE))

    if args.freeze_bert:
        for param in model.encoder.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n🔒 BERT backbone frozen. Trainable params: {trainable:,} (heads only)")
        print("   → Good for < 500 company samples (avoids overfitting)")
    else:
        total = sum(p.numel() for p in model.parameters())
        print(f"\nFine-tuning all {total:,} parameters.")
        print("   → Use --freeze_bert if you have fewer than 500 samples")

    trainer = Trainer(model, config)
    trainer.train(train_loader, val_loader, save_dir=args.output)

    os.makedirs(args.output, exist_ok=True)
    tokenizer.save_pretrained(args.output)
    print(f"\n✅ Fine-tuned model saved to: {args.output}/")
    print(f"   Use it with: python step4_inference.py --model {args.output} --input ...")


if __name__ == "__main__":
    main()
