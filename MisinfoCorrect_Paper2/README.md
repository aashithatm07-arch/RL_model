# MisinfoCorrect — Paper 2 Implementation

Implementation of:

**Bing He, Mustaque Ahamad, Srijan Kumar (2023)**  
*Reinforcement Learning-based Counter-Misinformation Response Generation: A Case Study of COVID-19 Vaccine Misinformation*  
WWW 2023.

Official paper/code:
- Paper: https://arxiv.org/abs/2303.06433
- Official code: https://github.com/claws-lab/MisinfoCorrect

## 1. What did I understand from the paper?

The paper proposes **MisinfoCorrect**, a reinforcement-learning-based response generator for misinformation. A transformer language model generates a counter-response, and the generated response receives rewards for five desired properties:

1. Politeness
2. Refutation of the misinformation
3. Evidence
4. Fluency
5. Relevance/coherence

The paper warm-starts a GPT-2/DialoGPT-style model using paired misinformation and counter-response data. During RL, the objective is:

`L(theta) = - r * log p(response | misinformation)`

where the total reward is a weighted combination of the five rewards.

## 2. How did I implement the proposed approach?

The implementation has two stages:

### Stage A — Supervised warm-up
`microsoft/DialoGPT-small` is fine-tuned on misinformation/counter-response pairs.

### Stage B — Reinforcement learning
For every training example:

1. The model receives a misinformation post.
2. It samples a candidate response.
3. The response is scored on politeness, refutation, evidence, fluency, and relevance.
4. The scores are combined into one reward.
5. The model is updated using:

`loss = -reward * log_probability(response | misinformation)`

This follows the main optimization idea in the paper.

## 3. What challenges did I face?

The original implementation requires a GPT-2/DialoGPT environment plus separately trained BERT reward classifiers. The original paper also reports experiments on a multi-GPU DGX-1 setup. Reproducing that exact environment is unnecessarily heavy for a short assessment.

The biggest practical challenges are:
- Large transformer memory requirements
- Training separate reward classifiers
- Reproducing the authors' exact data preprocessing
- Older dependencies in the original repository
- GPU availability

## 4. What modifications did I make?

The assignment permits smaller models, smaller datasets, reduced training, and proof-of-concept implementations.

Therefore:

- I use `DialoGPT-small` instead of the larger original setup.
- I use a compact TSV dataset for a fast demonstration.
- I use lightweight reward proxies for politeness, refutation and evidence.
- I keep the paper's GPT-2-style fluency reward idea using inverse-perplexity-style scoring.
- I keep semantic relevance using sentence embeddings and cosine similarity.
- Training steps are deliberately small.

**Important:** the reward objectives are faithful to the paper, but the first three reward implementations are approximations. The original paper trains BERT classifiers for these rewards.

## 5. If I had another opportunity, how would I improve/save time?

I would first reproduce the official repository using the authors' exact environment and pretrained reward classifiers. I would then verify the official dataset and preprocessing before starting any training. After that, I would run a tiny sanity-check experiment and only then scale up training.

---

# Project structure

```text
MisinfoCorrect_Paper2/
├── data/
│   └── sample_data.tsv
├── src/
│   ├── model.py
│   └── reward.py
├── train.py
├── generate.py
├── evaluate.py
├── requirements.txt
└── README.md
```

# Installation

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

# Run training

```bash
python train.py --steps 10 --warmup_epochs 1
```

For a stronger run:

```bash
python train.py --steps 50 --warmup_epochs 2
```

# Generate a response

```bash
python generate.py --model outputs/misinfocorrect
```

Or directly:

```bash
python generate.py --model outputs/misinfocorrect --text "COVID-19 vaccines contain microchips that track people."
```

# Evaluate

```bash
python evaluate.py --model outputs/misinfocorrect
```

The result is saved to:

```text
outputs/evaluation.csv
```

# Methodology

```text
Misinformation Post
        |
        v
DialoGPT-small
        |
        v
Candidate Counter-response
        |
        +----> Politeness Reward
        |
        +----> Refutation Reward
        |
        +----> Evidence Reward
        |
        +----> Fluency Reward
        |
        +----> Relevance Reward
        |
        v
   Total Reward
        |
        v
REINFORCE:
Loss = - Reward × log P(response | misinformation)
        |
        v
Updated Generator
```

# Expected assessment explanation

**Problem:** Ordinary users often respond to misinformation in rude or unsupported ways.

**Solution:** Generate a counter-response automatically and train the generator to prefer responses that are polite, refuting, evidenced, fluent and relevant.

**Why reinforcement learning?** There is no single perfect response for every misinformation post. Instead of forcing one exact answer, the model can generate candidates and receive a reward based on desirable properties.

**State:** The misinformation post.

**Action:** The generated counter-response.

**Policy:** GPT-2/DialoGPT-style transformer.

**Reward:** Combination of politeness, refutation, evidence, fluency and relevance.

**Loss:** `-reward × log probability`.

**Output:** A contextual counter-misinformation response.

## Note on safety

This is a research/educational implementation. Generated text should be checked against reliable sources before being used as a real-world fact-check.
