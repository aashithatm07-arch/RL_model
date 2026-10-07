import math
import re
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

REFUTE_WORDS = [
    "not true", "not correct", "incorrect", "false", "misleading",
    "does not", "do not", "doesn't", "don't", "no evidence",
    "unsupported", "not supported", "cannot", "can't", "wrong"
]

POLITE_WORDS = [
    "please", "understand", "concern", "respect", "thank", "sorry",
    "i understand", "i can see", "hope this helps"
]

EVIDENCE_WORDS = [
    "evidence", "study", "studies", "research", "data", "clinical",
    "trial", "trials", "documented", "tested", "monitoring",
    "according to", "researchers", "scientific"
]

class RewardScorer:
    """
    Assessment-friendly approximation of the paper's reward module.

    Original paper:
      - BERT politeness classifier
      - BERT pairwise refutation classifier
      - BERT pairwise evidence classifier
      - inverse GPT-2 perplexity fluency reward
      - BERT embedding cosine similarity relevance reward

    This implementation keeps the same five reward objectives, but uses
    lightweight lexical scorers for the first three so the project can run
    without separately training three BERT classifiers.
    """

    def __init__(self, device=None, embedding_model="all-MiniLM-L6-v2",
                 lm_name="distilgpt2"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.lm_tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.lm_model = AutoModelForCausalLM.from_pretrained(lm_name).to(self.device)
        self.lm_model.eval()
        if self.lm_tokenizer.pad_token is None:
            self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token

    @staticmethod
    def _phrase_score(text, phrases):
        t = text.lower()
        hits = sum(1 for p in phrases if p in t)
        return min(1.0, hits / 2.0)

    def politeness(self, response):
        # Reward respectful language and penalize obvious hostile wording.
        score = self._phrase_score(response, POLITE_WORDS)
        hostile = ["idiot", "stupid", "shut up", "dumb", "moron", "liar", "hate"]
        score -= 0.25 * min(1.0, sum(x in response.lower() for x in hostile))
        return max(0.0, min(1.0, score))

    def refutation(self, misinformation, response):
        # Lightweight proxy for the paper's pairwise BERT refutation classifier.
        score = self._phrase_score(response, REFUTE_WORDS)
        # Bonus when the response explicitly contrasts the claim.
        if "claim" in response.lower() or "statement" in response.lower():
            score += 0.1
        return max(0.0, min(1.0, score))

    def evidence(self, misinformation, response):
        return self._phrase_score(response, EVIDENCE_WORDS)

    @torch.no_grad()
    def fluency(self, response):
        if not response.strip():
            return 0.0
        enc = self.lm_tokenizer(
            response, return_tensors="pt", truncation=True, max_length=128
        ).to(self.device)
        labels = enc["input_ids"]
        out = self.lm_model(**enc, labels=labels)
        # exp(-loss) is proportional to inverse perplexity and is stable for scoring.
        return float(torch.exp(-out.loss).clamp(0, 1).item())

    def relevance(self, misinformation, response):
        if not misinformation.strip() or not response.strip():
            return 0.0
        emb = self.embedder.encode(
            [misinformation, response], convert_to_tensor=True, normalize_embeddings=True
        )
        sim = util.cos_sim(emb[0], emb[1]).item()
        return max(0.0, min(1.0, (sim + 1) / 2))

    def total(self, misinformation, response, weights=None):
        weights = weights or {
            "politeness": 1.0,
            "refutation": 1.0,
            "evidence": 1.0,
            "fluency": 1.0,
            "relevance": 1.0,
        }
        scores = {
            "politeness": self.politeness(response),
            "refutation": self.refutation(misinformation, response),
            "evidence": self.evidence(misinformation, response),
            "fluency": self.fluency(response),
            "relevance": self.relevance(misinformation, response),
        }
        total = sum(weights[k] * scores[k] for k in scores)
        return total / sum(weights.values()), scores
