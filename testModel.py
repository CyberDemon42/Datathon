from transformers import pipeline
pipe = pipeline('fill-mask', model='nur-dev/roberta-kaz-large')
predicted = pipe("Қазіргі <mask> әлемдік деңгейдегі <mask> университеттері сапалы білім, зияткерлік және мәдени <mask> беретін <mask> <mask> <mask> ғана емес, сонымен қатар мемлекет үшін <mask> қабілетті адами капиталды құратын <mask>, ғылым және өндірісті интеграциялаудың <mask> <mask> болып табылады.")

for t in predicted:
  print(t[0]['score'], t[0]['token_str'])
