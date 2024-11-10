from datasets import load_from_disk
from transformers import AutoTokenizer, RobertaForQuestionAnswering, pipeline
import re

# Load the tokenizer and the question-answering model
tokenizer = AutoTokenizer.from_pretrained('nur-dev/roberta-kaz-large')  # Or use another QA model if available
model = RobertaForQuestionAnswering.from_pretrained('nur-dev/roberta-kaz-large')

# Load your dataset
test_dataset = load_from_disk("/home/cyberdemon/Desktop/Datathon")

# Check the structure of the dataset
print(test_dataset['train'][0])  # Ensure it has 'context', 'question', and 'answer' keys

# Initialize the question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Evaluation lists
predictions = []
references = []

# Iterate over the dataset
for sample in test_dataset['train']:
    context = sample.get("context")
    question = sample.get("question")
    true_answer = sample.get("answer")

    # Skip any samples missing data
    if context and question and true_answer:
        # Make prediction
        pred = qa_pipeline({"context": context, "question": question})
        predictions.append({"answer": pred["answer"]})
        references.append({"answer": true_answer})

# Define function to calculate EM and F1
def compute_em_f1(predictions, references):
    exact_matches = 0
    f1_total = 0.0

    for pred, ref in zip(predictions, references):
        # Normalize text
        pred_answer = pred["answer"]
        true_answer = ref["answer"]

        # Exact Match
        if pred_answer == true_answer:
            exact_matches += 1

        # F1 Score
        pred_tokens = set(pred_answer.split())
        true_tokens = set(true_answer.split())
        common_tokens = pred_tokens & true_tokens

        if len(common_tokens) == 0:
            f1 = 0.0
        else:
            precision = len(common_tokens) / len(pred_tokens)
            recall = len(common_tokens) / len(true_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
        f1_total += f1

    exact_match = exact_matches / len(predictions) * 100
    f1_score = f1_total / len(predictions) * 100
    return exact_match, f1_score

# Calculate scores
em, f1 = compute_em_f1(predictions, references)
print(f"Exact Match (EM): {em:.2f}%")
print(f"F1 Score: {f1:.2f}%")
