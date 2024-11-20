from sentence_transformers import SentenceTransformer
import openai
import random
from sklearn.metrics import classification_report
import logging
import torch

logging.basicConfig(level=logging.INFO)

from transformers import pipeline

class AttributeValidator:
    """Handles validation of target manipulation and non-target attribute preservation."""

    def __init__(self):

        self.sentiment_analyzer = pipeline("sentiment-analysis")

    def validate_target_attribute(self, original_text: str, modified_text: str, target_attribute: str) -> bool:
        """Validate if the target attribute has been successfully manipulated."""
        modified_sentiment = self.sentiment_analyzer(modified_text)[0]['label']
        return modified_sentiment.lower() == target_attribute.lower()

    def validate_non_target_preservation(self, original_text: str, modified_text: str) -> bool:
        """Validate if non-target attributes (e.g., tone, structure) are preserved."""
        original_sentiment = self.sentiment_analyzer(original_text)[0]['label']
        modified_sentiment = self.sentiment_analyzer(modified_text)[0]['label']
        return original_sentiment == modified_sentiment

class EnhancedCoTAM:
    """Enhanced Chain-of-Thought Attribute Manipulation with validation."""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.attribute_validator = AttributeValidator()

    def validate_manipulation(self, original_text: str, modified_text: str, target_attribute: str) -> bool:
        """Validate target attribute change and non-target preservation."""
        is_target_changed = self.attribute_validator.validate_target_attribute(
            original_text, modified_text, target_attribute
        )
        is_preservation_valid = self.attribute_validator.validate_non_target_preservation(
            original_text, modified_text
        )
        return is_target_changed and is_preservation_valid

    def process_text(self, text: str, target_attribute: str) -> str:
        """Generate manipulated text."""
        return f"Manipulated version of: {text} with {target_attribute}"


class EnhancedCoTAM:
    def __init__(self, api_key: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.api_key = api_key
        openai.api_key = api_key
        self.model = SentenceTransformer(embedding_model)

    def decompose_attributes(self, text: str) -> dict:
        """Decompose text into attributes."""
        attributes = {"sentiment": "positive" if "good" in text.lower() else "negative"}
        return attributes

    def validate_preservation(self, original_text: str, modified_text: str) -> bool:
        """Validate attribute preservation."""
        original_embedding = self.model.encode(original_text, convert_to_tensor=True)
        modified_embedding = self.model.encode(modified_text, convert_to_tensor=True)
        

        if len(original_embedding.shape) > 1:
            original_embedding = original_embedding.squeeze(0)
        if len(modified_embedding.shape) > 1:
            modified_embedding = modified_embedding.squeeze(0)
        

        similarity = torch.nn.functional.cosine_similarity(original_embedding, modified_embedding, dim=0)
        

        return similarity.item() > 0.85


    def generate_plan(self, text: str, target_attribute: str) -> str:
        """Generate a manipulation plan."""
        return f"""
Text: "{text}"
Target Attribute: {target_attribute}

Plan:
1. Identify key sentiment phrases.
2. Replace with alternatives reflecting the target attribute.
3. Ensure coherence and grammar."""

    def reconstruct_text(self, text: str, plan: str) -> str:
        """Reconstruct text with target attribute."""
        prompt = f"""
Original Text: "{text}"
Manipulation Plan: {plan}

Provide the modified text only."""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message['content'].strip()
