import argparse
import os
import pandas as pd
import torch
from tqdm import tqdm

from src.model import CounterMisinformationGenerator
from src.reward import RewardScorer

def load_data(path):
    df = pd.read_csv(path, sep="\t")
    df = df.dropna()
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.tsv")
    parser.add_argument("--model", default="microsoft/DialoGPT-small")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_data(args.data)

    generator = CounterMisinformationGenerator(args.model)
    scorer = RewardScorer()

    # -------------------------------
    # Stage 1: supervised warm start
    # -------------------------------
    optimizer = torch.optim.AdamW(generator.model.parameters(), lr=args.lr)
    generator.model.train()

    print("\n[1/2] Supervised warm-up...")
    for epoch in range(args.warmup_epochs):
        total_loss = 0.0
        for _, row in tqdm(df.iterrows(), total=len(df)):
            optimizer.zero_grad()
            loss = generator.supervised_loss(row["misinformation"], row["counter_response"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Warm-up epoch {epoch+1}: loss={total_loss/len(df):.4f}")

    # -------------------------------
    # Stage 2: REINFORCE
    # L = -reward * log p(response|misinformation)
    # -------------------------------
    print("\n[2/2] Reinforcement learning...")
    for step in range(args.steps):
        row = df.iloc[step % len(df)]
        generator.model.eval()

        # Generate candidate action.
        response = generator.generate(
            row["misinformation"],
            max_new_tokens=args.max_new_tokens
        )

        # Compute paper-inspired multi-objective reward.
        reward, parts = scorer.total(row["misinformation"], response)

        generator.model.train()
        optimizer.zero_grad()

        logp = generator.response_logprob(row["misinformation"], response)
        loss = -reward * logp

        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 1 == 0:
            print(
                f"step={step+1:03d} reward={reward:.4f} "
                f"polite={parts['politeness']:.2f} "
                f"refute={parts['refutation']:.2f} "
                f"evidence={parts['evidence']:.2f} "
                f"fluency={parts['fluency']:.2f} "
                f"relevance={parts['relevance']:.2f}"
            )
            print("  generated:", response)

    save_path = os.path.join(args.output_dir, "misinfocorrect")
    generator.model.save_pretrained(save_path)
    generator.tokenizer.save_pretrained(save_path)
    print(f"\nSaved model to: {save_path}")

if __name__ == "__main__":
    main()
