from transformers import RobertaTokenizerFast, RobertaForMaskedLM

tokenizer = RobertaTokenizerFast.from_pretrained('nur-dev/roberta-kaz-large')
model = RobertaForMaskedLM.from_pretrained('nur-dev/roberta-kaz-large')
