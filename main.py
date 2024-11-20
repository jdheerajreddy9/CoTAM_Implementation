import json
from datamanager import DataManager
from enhancedcotam import EnhancedCoTAM
from classifier import TransformerClassifier
from robustnesstester import RobustnessTester
import openai

def main():

    data_manager = DataManager()
    texts, labels = data_manager.load_dataset("sst2", size=500)
    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = data_manager.prepare_data(texts, labels)


    cotam = EnhancedCoTAM(api_key="Your API key here") #Please Enter Your API key here
    augmented_texts, augmented_labels = [], []
    logged_results = []


    for text, label in zip(train_texts, train_labels):
        attributes = cotam.decompose_attributes(text)
        target_attribute = "positive" if label == 0 else "negative"
        plan = cotam.generate_plan(text, target_attribute)
        modified_text = cotam.reconstruct_text(text, plan)
        if cotam.validate_preservation(text, modified_text):
            augmented_texts.append(modified_text)
            augmented_labels.append(1 - label)
        

        logged_results.append({
            "original_text": text,
            "original_label": label,
            "modified_text": modified_text,
            "target_attribute": target_attribute,
            "plan": plan,
            "attributes": attributes
        })


    with open("cotam_results.json", "w") as f:
        json.dump(logged_results, f, indent=2)


    combined_texts = train_texts + augmented_texts
    combined_labels = train_labels + augmented_labels


    classifier = TransformerClassifier()
    classifier.train(combined_texts, combined_labels)
    results = classifier.evaluate(test_texts, test_labels)
    print("Evaluation Results:", results)


    tester = RobustnessTester(classifier.tokenizer, classifier.model)
    robustness_results = tester.evaluate_robustness(test_texts, test_labels)
    print("Robustness Results:", robustness_results)

if __name__ == "__main__":
    main()
