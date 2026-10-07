import argparse
import pandas as pd
from src.model import CounterMisinformationGenerator
from src.reward import RewardScorer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_data.tsv")
    parser.add_argument("--model", default="outputs/misinfocorrect")
    args = parser.parse_args()

    df = pd.read_csv(args.data, sep="\t").dropna()
    generator = CounterMisinformationGenerator(args.model)
    scorer = RewardScorer()

    rows = []
    for _, row in df.iterrows():
        response = generator.generate(row["misinformation"], max_new_tokens=80)
        total, parts = scorer.total(row["misinformation"], response)
        rows.append({
            "misinformation": row["misinformation"],
            "generated_response": response,
            "total_reward": total,
            **parts
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv("outputs/evaluation.csv", index=False)
    print("\nSaved: outputs/evaluation.csv")

if __name__ == "__main__":
    main()
