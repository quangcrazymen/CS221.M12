import re
import unicodedata as ud
import ast as ast

def syllablize(sentence):
    word = '\w+'
    non_word = '[^\w\s]'
    digits = '\d+([\.,_]\d+)+'
    
    patterns = []
    patterns.extend([word, non_word, digits])
    patterns = f"({'|'.join(patterns)})"
    
    sentence = ud.normalize('NFC', sentence)
    tokens = re.findall(patterns, sentence, re.UNICODE)
    return [token[0] for token in tokens]

def load_n_grams(path):
    with open(path, encoding='utf8') as f:
        words = f.read()
        words = ast.literal_eval(words)
    return words

def longest_matching_L2R(sentence, TuGhep):
    syllables = syllablize(sentence)
    syl_len = len(syllables)
    
    curr_id = 0
    word_list = []
    done = False
    
    while (curr_id < syl_len) and (not done):
        curr_word = syllables[curr_id]
        if curr_id >= syl_len - 1:
            word_list.append(curr_word)
            done = True
        else:
            next_word = syllables[curr_id + 1]
            pair_word = ' '.join([curr_word.lower(), next_word.lower()])
    
            if pair_word in TuGhep:
                word_list.append('_'.join([curr_word, next_word]))
                curr_id += 2
            else:
                word_list.append(curr_word)
                curr_id += 1

    return word_list

def longest_matching_R2L(sentence, TuGhep):
    syllables = syllablize(sentence)
    syl_len = len(syllables)
    
    curr_id = syl_len-1
    word_list = []
    done = False
    
    while (curr_id >0) and (not done):
        curr_word = syllables[curr_id]
        if curr_id == 0:
            word_list.insert(0,curr_word)
            done = True
        else:
            next_word = syllables[curr_id -1]
            pair_word = ' '.join([next_word.lower(),curr_word.lower()])
            if curr_id >= 1:
                if pair_word in TuGhep:
                    word_list.insert(0,'_'.join([next_word,curr_word]))
                    curr_id -= 2
                else:
                    word_list.insert(0,curr_word)
                    curr_id -= 1
    return word_list
TuGhep = load_n_grams('Dictionary/TuGhep.txt')
print(longest_matching_L2R('h???c sinh m???i h???c', TuGhep))
print(longest_matching_L2R('nh??ng s??? th???c hi???n v???n c??n ch??a ph?? h???p', TuGhep))
print(longest_matching_L2R('B???ng thuy???n tr?????ng ???????ng bi???n h???ng V v?? h???ng IV c???a ch??? ch???ng c?? gi?? tr??? g?? khi ?????ng trong bu???ng l??i con t??u du l???ch n??y .',TuGhep))

#========================================================
print(len(syllablize('h???c sinh m???i h???c')))

# sentences = open('60CauRaw.txt', encoding='utf-8').readlines()
# tokenize_sentences = [sentence.split(' ') for sentence in sentences]

# with open('longest_matching_tokens.txt', 'w', encoding='utf-8') as f:
#     longest_matching_sentences = []
#     for sentence in sentences:
#         word_list = longest_matching_L2R(sentence, TuGhep, tri_grams)
#         longest_matching_sentences.append(' '.join(word_list))
#         for word in word_list: f.write(word + '\n')
#         if sentence != sentences[-1]: f.write('\n')
#     f.write('\n')
# longest_matching_sentences[0:3]

# count_longest_matching_compounds = 0
# for sentence in longest_matching_sentences:
#     for word in sentence.split():
#         if '_' in word: count_longest_matching_compounds += 1
        
# print('S??? l?????ng t??? gh??p khi t??ch t??? b???ng thu???t to??n Longest Matching:', count_longest_matching_compounds)

# with open('60cauTVManual.txt', 'r', encoding='utf-8') as f:
#     manual_tokenize_sentences = []
#     sentence = ''
#     for word in f:
#         if word == '\n': 
#             manual_tokenize_sentences.append(sentence.strip())
#             sentence = ''
#         else: sentence += word.replace('\n', ' ')
# manual_tokenize_sentences[0:3]

# count_manual_tokenize_compounds = 0
# for sentence in manual_tokenize_sentences:
#     for word in sentence.split():
#         if '_' in word: count_manual_tokenize_compounds += 1
# print('S??? l?????ng t??? gh??p khi t??ch t??? th??? c??ng:', count_manual_tokenize_compounds)

# S??? c??u ti???ng vi???t trong file
# words = open('../../OurDatasets/LargeVLSPData.txt', encoding='utf-8').readlines()
# count_longest_matching_compounds = 0
# for word in words:
#     if '\t\n' in word: count_longest_matching_compounds += 1
# print(count_longest_matching_compounds)
# print(words[0:41])

#T???ng s??? c??u: 16835
