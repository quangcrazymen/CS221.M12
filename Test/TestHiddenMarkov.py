# Tính số lượng từ trong file:
import math
from collections import defaultdict
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


# Đọc dữ liệu:
def preprocess(vocabs_dict, path):
    data = []
    file = open(path, encoding="utf-8").readlines()

    for index, word in enumerate(file):
        if not word.split():
            word = "--n--"
            data.append(word)
            continue
        elif word.strip() not in vocabs_dict:
            word = "--unk--"
            data.append(word)
            continue
        data.append(word.strip())
    return data


# Từ điển:
vocabs = open("Dict.txt", encoding="utf-8").read().split("\n")
vocabs_dict = {}
index = 0

for word in sorted(vocabs):
    if word not in vocabs_dict:
        vocabs_dict[word] = index
        index += 1
print(vocabs_dict)

train_words = preprocess(vocabs_dict, "train_word.txt")
#print("Số lượng từ trong tập train_words:", countWord(train_words))

#print("Các từ không nằm trong từ điển: ")

#Pos tagging:
#training:
def seperate_word_tag(word_tag,vocabs_dict):
    if not word_tag.split():
        word = '--n--'
        tag = '--s--'
    else:
        word, tag= word_tag.split()
        if word not in vocabs_dict: word = '--unk--'
    return word,tag

def create_dictionaries(train_gold, vocab):
    emission_counts = defaultdict(int)
    transition_counts = defaultdict(int)
    tag_counts = defaultdict(int)

    prev_tag = '--s--'
    for word_tag in train_gold:
        word, tag = seperate_word_tag(word_tag,vocab)

        transition_counts[(prev_tag,tag)] +=1
        emission_counts[(tag, word)] += 1
        tag_counts[tag] += 1
        prev_tag = tag
    
    return transition_counts, emission_counts, tag_counts 

#Tập train gold:
#train_gold = open('dataset/50CauGanNhanTrain_Gold.txt', encoding='utf-8').readlines()
train_gold = open('train_gold.txt', encoding='utf-8').readlines()
#print('Số lượng từ trong tập train_gold:', countWord(train_gold))
train_gold[0:5]

transition_counts, emission_counts, tag_counts = create_dictionaries(train_gold, vocabs_dict)
states = sorted(tag_counts.keys())
print('Số nhãn:', len(states))
print(states)

print("Transition examples: ")
for example in list(transition_counts.items())[:3]:
    print(example)

print("Emission examples: ")
for example in list(emission_counts.items())[:3]:
    print (example)


#Ma trận A:
def create_transition_matrix(alpha, tag_counts, transition_counts):
    all_tags = sorted(tag_counts.keys())
    num_tags = len(all_tags)

    A = np.zeros((num_tags, num_tags))
    
    for i in range(num_tags):
        for j in range(num_tags):
            count = 0
            key = (all_tags[i], all_tags[j])
            if key in transition_counts: count = transition_counts[key]
                
            count_prev_tag = tag_counts[all_tags[i]]
            A[i, j] = (count + alpha) / (count_prev_tag + alpha * num_tags)
    return A

alpha = 1
for i in range(len(states)): tag_counts.pop(i, None)
    
A = create_transition_matrix(alpha, tag_counts, transition_counts)
df = pd.DataFrame(
    A[0:, 0:], 
    index = states[0:], 
    columns = states[0:]
)
df.head()
print(df)

def create_emission_matrix(alpha, tag_counts, emission_counts, vocabs):
    all_tags = sorted(tag_counts.keys())
    num_tags = len(tag_counts)
    num_words = len(vocabs)
    
    B = np.zeros((num_tags, num_words))
    
    for i in range(num_tags):
        for j in range(num_words):
            count = 0
            key = (all_tags[i], vocabs[j])
            if key in emission_counts.keys(): count = emission_counts[key]
                
            count_tag = tag_counts[all_tags[i]]
            B[i, j] = (count + alpha) / (count_tag + alpha * num_words)
    return B

cidx  = ['em', 'tập', 'đánh', 'đàn', 'chó','chạy','ngoài','sân','trường','rợp','bóng','cây','cao','cả']
rvals = ['N', 'V', 'E', 'A']
cols = [vocabs_dict[word] for word in cidx]
rows = [states.index(tag) for tag in rvals]

for i in range(len(states)): tag_counts.pop(i, None)
B = create_emission_matrix(alpha, tag_counts, emission_counts, list(vocabs_dict))

df = pd.DataFrame(B[np.ix_(rows, cols)], index=rvals, columns=cidx)
print(df)


A = np.array([sublist[1:].tolist() for sublist in A])###############
B = B[1:]#####################

test_words = preprocess(vocabs_dict, 'test.txt')
print('Số lượng từ trong tập test_words:', len(test_words))
test_words[0:5]

def viterbi_initialize(states, tag_counts, A, B, corpus, vocabs_dict):
    num_tags = len(tag_counts)
    s_idx = states.index('--s--')
    
    best_probs = np.zeros((num_tags, len(corpus)))
    best_paths = np.zeros((num_tags, len(corpus)), dtype=int)
    
    for i in range(num_tags):
        if A[s_idx, i - 1] == 0: best_probs[i, 0] = float('-inf')
        else: 
            index = vocabs_dict[corpus[0]]
            best_probs[i, 0] = math.log(A[s_idx, i - 1]) + math.log(B[i - 1, index])
    return best_probs, best_paths

best_probs_train, best_paths_train = viterbi_initialize(states, tag_counts, A, B, train_words, vocabs_dict)
print('best_probs_train[0, 0]:', best_probs_train[0, 0]) 
print('best_paths_train[2, 3]:', best_paths_train[2, 3])

best_probs_test, best_paths_test = viterbi_initialize(states, tag_counts, A, B, test_words, vocabs_dict)
print('best_probs_test[0, 0]:', best_probs_test[0, 0]) 
print('best_paths_test[2, 3]:', best_paths_test[2, 3])


########################FORWARD###################################

def viterbi_forward(A, B, corpus, best_probs, best_paths, vocabs_dict):
    num_tags = best_probs.shape[0]
    
    for i in range(1, len(corpus)): 
            
        for j in range(num_tags):
            best_prob_i = float('-inf')
            best_path_i = None
            
            for k in range(num_tags):
                index = vocabs_dict[corpus[i]]
                prob = best_probs[k, i - 1] + math.log(A[k, j - 1]) + math.log(B[j - 1, index])

                if prob > best_prob_i:
                    best_prob_i = prob
                    best_path_i = k
                    
            best_probs[j, i] = best_prob_i
            best_paths[j, i] = best_path_i
            
    return best_probs, best_paths

#print(train_words)
#print('Vocab dict: ',vocabs_dict)
best_probs_train, best_paths_train = viterbi_forward(A, B, train_words, best_probs_train, best_paths_train, vocabs_dict)

print('best_probs_train[0, 1]:', best_probs_train[0, 1]) 
print('best_paths_train[0, 4]:', best_paths_train[0, 4])

best_probs_test, best_paths_test = viterbi_forward(A, B, test_words, best_probs_test, best_paths_test, vocabs_dict)
print('best_probs_test[0, 1]:', best_probs_test[0, 1]) 
print('best_paths_test[0, 4]:', best_paths_test[0, :4])

########################BACKWARD###################################

def viterbi_backward(best_probs, best_paths, states):
    m = best_paths.shape[1] 
    z = [None] * m
    pred = [None] * m
    
    best_prob_for_last_word = float('-inf')
    num_tags = best_probs.shape[0]
    
    for k in range(num_tags):
        if best_probs[k, m - 1] > best_prob_for_last_word:
            best_prob_for_last_word = best_probs[k, m - 1]
            z[m - 1] = k
            
    pred[m - 1] = states[z[m - 1]]
    for i in range(m - 1, -1, -1):
        z[i - 1] = best_paths[z[i], i]
        pred[i - 1] = states[z[i - 1]]
    return pred

    train_pred = viterbi_backward(best_probs_train, best_paths_train, train_words, states)
train_pred = viterbi_backward(best_probs_train, best_paths_train, train_words, states)
test_pred = viterbi_backward(best_probs_test, best_paths_test, states)

print('Dự đoán cho test_pred:')
print(test_words)
print(test_pred)