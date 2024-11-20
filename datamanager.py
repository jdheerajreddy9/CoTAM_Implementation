from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import List, Tuple

class DataManager:
    def load_dataset(self, dataset_name: str = "sst2", size: int = 500, seed: int = 42) -> Tuple[List[str], List[int]]:
        """Load and sample dataset."""
        dataset = load_dataset("glue", dataset_name)["train"]
        sampled_data = dataset.shuffle(seed=seed).select(range(size))
        texts = [data["sentence"] for data in sampled_data]
        labels = [data["label"] for data in sampled_data]
        return texts, labels

    def prepare_data(
        self, texts: List[str], labels: List[int], test_size: float = 0.2, val_size: float = 0.1, seed: int = 42
    ) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
        """Split dataset into train, validation, and test sets."""
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=test_size + val_size, random_state=seed, stratify=labels
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=test_size / (test_size + val_size), random_state=seed, stratify=temp_labels
        )
        return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
