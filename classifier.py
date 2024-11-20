import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report
from typing import List

class TransformerClassifier:
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

    def train(self, train_texts: List[str], train_labels: List[int], batch_size: int = 16, epochs: int = 3):
        self.model.train()
        train_data = self._prepare_dataloader(train_texts, train_labels, batch_size)
        for epoch in range(epochs):
            for batch in train_data:
                input_ids, attention_mask, labels = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, texts: List[str], labels: List[int], batch_size: int = 16) -> dict:
        self.model.eval()
        val_data = self._prepare_dataloader(texts, labels, batch_size, shuffle=False)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_data:
                input_ids, attention_mask, labels = [t.to(self.device) for t in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        return classification_report(all_labels, all_preds, output_dict=True)

    def _prepare_dataloader(self, texts: List[str], labels: List[int], batch_size: int, shuffle: bool = True):
        tokenized = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        dataset = TensorDataset(tokenized["input_ids"], tokenized["attention_mask"], torch.tensor(labels))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
