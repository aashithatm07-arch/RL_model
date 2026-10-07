import argparse
from src.model import CounterMisinformationGenerator
from src.reward import RewardScorer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/misinfocorrect")
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    text = args.text or input("Enter misinformation post: ").strip()

    generator = CounterMisinformationGenerator(args.model)
    scorer = RewardScorer()

    response = generator.generate(text, max_new_tokens=80)
    total, parts = scorer.total(text, response)

    print("\nMisinformation:")
    print(text)
    print("\nGenerated counter-response:")
    print(response)
    print("\nReward breakdown:")
    for k, v in parts.items():
        print(f"{k:12s}: {v:.3f}")
    print(f"total       : {total:.3f}")

if __name__ == "__main__":
    main()
