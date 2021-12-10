import nltk

files = ['iliad.txt', 'odyssey.txt']
output_files = ['iliad_sents.txt', 'odyssey_sents.txt']

for i, file in enumerate(files):
    text = open(file).read()
    sents = nltk.tokenize.sent_tokenize(text, language="english")
    sents_processed = []
    for sent in sents:
        sents_processed.append(sent.replace('\n', ' '))
    
    filename = output_files[i]
    textfile = open(filename, 'w+')
    for sent in sents_processed:
        textfile.write(sent + '\n')
    textfile.close()