import os
import torch
from torch.utils.data import DataLoader
from transformers import RobertaForQuestionAnswering, RobertaTokenizerFast, AdamW, DataCollatorWithPadding
from datasets import load_from_disk
import torch._dynamo

torch._dynamo.config.suppress_errors = True

# Установка количества потоков для использования всех доступных ядер
num_threads = os.cpu_count()  # Получаем количество доступных ядер
torch.set_num_threads(num_threads)
print(f"Using {num_threads} CPU threads.")

# Инициализация устройства (CPU)
device = torch.device("cpu")
print(f"Using device: {device}")

# Загрузка модели и токенайзера на CPU
model = RobertaForQuestionAnswering.from_pretrained('nur-dev/roberta-kaz-large').to(device)
tokenizer = RobertaTokenizerFast.from_pretrained('nur-dev/roberta-kaz-large')
model.gradient_checkpointing_enable()

# Если используется PyTorch 2.0+, компиляция модели для оптимизации памяти
if hasattr(torch, "compile"):
    model = torch.compile(model)

# Загрузка вашего датасета
ds = load_from_disk("/home/cyberdemon/Desktop/Datathon")

# Функция для токенизации и выделения позиций ответов
def preprocess_function(examples):
    inputs = [q + " " + c for q, c in zip(examples["question"], examples["context"])]
    model_inputs = tokenizer(
        inputs,
        max_length=128,  # Оптимальная длина
        padding="max_length",
        truncation=True
    )
    
    start_positions = []
    end_positions = []

    for answer, context in zip(examples["answer"], examples["context"]):
        start = context.find(answer)
        end = start + len(answer)
        
        start_positions.append(start)
        end_positions.append(end)

    model_inputs["start_positions"] = start_positions
    model_inputs["end_positions"] = end_positions

    return model_inputs

# Токенизация датасета
tokenized_dataset = ds["train"].map(preprocess_function, batched=True, remove_columns=ds["train"].column_names)

# Настройка коллатора для заполнения батчей
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Создание загрузчика данных
train_loader = DataLoader(tokenized_dataset, batch_size=2, shuffle=True, collate_fn=data_collator)

# Оптимизатор
optimizer = AdamW(model.parameters(), lr=5e-5)

# Градиентный накопитель
accumulation_steps = 16  # Эквивалентный размер батча будет 4

# Цикл обучения с использованием накопления градиентов
model.train()
for epoch in range(1):  # Можно увеличить количество эпох
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader):
        # Перемещение данных на CPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Разделение потерь для градиентного накопления
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        # Обновление весов каждые accumulation_steps шагов
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Печать потерь на каждые 10 шагов
        if step % 10 == 0:
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

# Сохранение обученной модели
os.makedirs("./fine_tuned_model", exist_ok=True)
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
