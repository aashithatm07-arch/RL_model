import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class CounterMisinformationGenerator:
    def __init__(self, model_name="microsoft/DialoGPT-small", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_pair(self, misinformation, response=None, max_length=256):
        prompt = misinformation.strip() + self.tokenizer.eos_token
        if response is not None:
            text = prompt + " " + response.strip()
        else:
            text = prompt
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        ).to(self.device)

    @torch.no_grad()
    def generate(self, misinformation, max_new_tokens=80, temperature=0.9, top_p=0.92):
        enc = self.encode_pair(misinformation)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def response_logprob(self, misinformation, response, max_length=256):
        """
        Sum log p(response | misinformation) over response tokens.
        This is the log-probability term used by REINFORCE.
        """
        prompt = misinformation.strip() + self.tokenizer.eos_token
        full = prompt + " " + response.strip()

        full_enc = self.tokenizer(
            full, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)
        prompt_enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        token_log_probs = torch.log_softmax(logits, dim=-1)
        chosen = token_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        prompt_len = prompt_enc["input_ids"].shape[1]
        # Label index i corresponds to input token i+1.
        response_start = max(prompt_len - 1, 0)
        response_log_probs = chosen[:, response_start:]

        return response_log_probs.sum()

    def supervised_loss(self, misinformation, response, max_length=256):
        prompt = misinformation.strip() + self.tokenizer.eos_token
        full = prompt + " " + response.strip()
        enc = self.tokenizer(
            full, return_tensors="pt", truncation=True, max_length=max_length
        ).to(self.device)

        labels = enc["input_ids"].clone()
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        )["input_ids"]
        prompt_len = prompt_ids.shape[1]
        labels[:, :prompt_len] = -100

        out = self.model(**enc, labels=labels)
        return out.loss
