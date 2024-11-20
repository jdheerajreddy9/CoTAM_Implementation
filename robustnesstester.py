import random
import re
from typing import List, Dict
import openai
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import torch

class RobustnessTester:
    def __init__(self, tokenizer, model, device="cuda"):
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def add_typos(self, text: str, typo_prob: float = 0.2) -> str:
        """Introduce typos by randomly removing or swapping characters."""
        def random_typo(word):
            if len(word) < 2:
                return word
            idx = random.randint(0, len(word) - 1)
            return word[:idx] + word[idx + 1:] if random.random() < 0.5 else word[:idx] + word[idx + 1:] + word[idx]
        return " ".join([random_typo(word) if random.random() < typo_prob else word for word in text.split()])

    def add_noise(self, text: str, noise_type: str = "punctuation") -> str:
        """Add noise to text by injecting extra spaces or punctuation."""
        if noise_type == "punctuation":
            return text + random.choice([".", "!", "??"])
        elif noise_type == "spacing":
            words = text.split()
            return " " * random.randint(1, 5) + " ".join(words) + " " * random.randint(1, 5)
        return text

    def paraphrase_text(self, text: str) -> str:
        """Generate a paraphrase using GPT or any LLM."""
        # Example paraphrasing with OpenAI GPT (you can replace this with other APIs/models)
        prompt = f"Paraphrase the following text:\n\n{text}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()

    def generate_perturbations(self, text: str) -> List[str]:
        """Generate perturbed versions of a text."""
        return [
            self.add_typos(text),
            self.add_noise(text, "punctuation"),
            self.add_noise(text, "spacing"),
        ]

    def evaluate_robustness(
        self, texts: List[str], labels: List[int], batch_size: int = 16, num_perturbations: int = 3
    ) -> Dict[str, float]:
        """Evaluate the model's performance on perturbed inputs."""
        self.model.eval()
        all_preds, all_labels = [], []
        
        for text, label in zip(texts, labels):
            perturbations = self.generate_perturbations(text)[:num_perturbations]
            tokenized = self.tokenizer(perturbations, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**tokenized).logits
                preds = torch.argmax(outputs, axis=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend([label] * len(perturbations))
        
        return classification_report(all_labels, all_preds, output_dict=True)

    def visualize_robustness(self, robustness_results: Dict[str, float]):
        """Optionally visualize robustness metrics."""
        print("Robustness Evaluation Results:")
        for metric, value in robustness_results.items():
            print(f"{metric}: {value}")
