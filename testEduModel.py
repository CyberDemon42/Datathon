import torch
from transformers import pipeline
from datasets import load_from_disk

test_dataset = load_from_disk("/home/cyberdemon/Desktop/Datathon")

# Перемещение модели на устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создание пайплайна для вопросно-ответной модели
qa_pipeline = pipeline(
    "question-answering",
    model="./fine_tuned_model",
    tokenizer="./fine_tuned_model",
    device=0 if torch.cuda.is_available() else -1  # Использование GPU, если доступен
)

for i in range(100):
    # Пример контекста и вопросов
    context = test_dataset['train'][i]['context']
    question = test_dataset['train'][i]['question']
    answer = test_dataset['train'][i]['answer']

    # Получение ответов на вопросы
    result = qa_pipeline(question=question, context=context)
    print(f"Вопрос: {question}\nОтвет: {result['answer']}\nПравильный: {answer}\n")
