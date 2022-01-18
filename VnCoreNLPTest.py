from vncorenlp import VnCoreNLP
client = VnCoreNLP(address='http://127.0.0.1', port=8000)
word_list = client.tokenize('nhưng sự thực hiện vẫn còn chưa phù hợp')[0]
word_list

sentences = open('dataset/60CauRaw.txt', encoding='utf-8').readlines()

with open('ResultOfTest/vncoreNLP_tokens.txt', 'w', encoding='utf-8') as f:
    vncore_sentences = []
    for sentence in sentences:
        word_list = client.tokenize(sentence)[0]
        vncore_sentences.append(' '.join(word_list))
        for word in word_list: f.write(word + '\n')
        if sentence != sentences[-1]: f.write('\n')
    f.write('\n')
vncore_sentences[0:3]