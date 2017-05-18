questions = {}
with open('questions.txt', 'r', encoding='utf-16') as f:
    for ligne in f.readlines():
        q_id, q_body = ligne.split('\t')
        q_id = int(q_id)
        if q_id not in questions:
            q_body = q_body.strip()
            if q_body == '':
                q_body = '(Question manquante)'
            questions[q_id] = q_body

for q_id in sorted(questions.keys()):
    print(str(q_id) + ': ' + questions[q_id])
    
with open('questions.csv', 'w') as f:
    for k, v in questions.items():
        f.write(str(k) + '\t' + v + '\n')
