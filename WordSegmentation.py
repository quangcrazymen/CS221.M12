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
print(longest_matching_L2R('học sinh mới học', TuGhep))
print(longest_matching_L2R('nhưng sự thực hiện vẫn còn chưa phù hợp', TuGhep))
print(longest_matching_L2R('Bằng thuyền trưởng đường biển hạng V và hạng IV của chị chẳng có giá trị gì khi đứng trong buồng lái con tàu du lịch này .',TuGhep))

#========================================================
print(len(syllablize('học sinh mới học')))

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
        
# print('Số lượng từ ghép khi tách từ bằng thuật toán Longest Matching:', count_longest_matching_compounds)

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
# print('Số lượng từ ghép khi tách từ thủ công:', count_manual_tokenize_compounds)

# Số câu tiếng việt trong file
# words = open('../../OurDatasets/LargeVLSPData.txt', encoding='utf-8').readlines()
# count_longest_matching_compounds = 0
# for word in words:
#     if '\t\n' in word: count_longest_matching_compounds += 1
# print(count_longest_matching_compounds)
# print(words[0:41])

#Tổng số câu: 16835
