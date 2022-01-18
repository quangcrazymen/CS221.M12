from turtle import color
import pandas as pd
import matplotlib.pyplot as plt

manual_tokens = open("dataset/50CauTrain_Word.txt", encoding="utf-8").readlines()  # readlines plural

# Number of words:
def countWord(file):
    notAWord = 0
    for word in file:
        if word == "\n" or word == "--n--":
            notAWord += 1
    return len(file) - notAWord


print(countWord(manual_tokens))
# Summarize tags:
def plot_tag_counts(gold):
    tags = [word_tag.split()[1] for word_tag in gold if word_tag.split()]
    tag_counts = pd.DataFrame(tags)[0].value_counts()
    tag_counts.plot.bar(rot=0, width=0.7, legend=False, figsize=(15, 5),color="#393ed4")
    plt.show()
    return pd.DataFrame(tag_counts).T.assign(Total=tag_counts.sum()) 

#file train 50 Cau:
train_gold = open('dataset/50CauGanNhanTrain_Gold.txt', encoding='utf-8').readlines()
print(plot_tag_counts(train_gold))

#file train 5000 cau:
train5000_gold = open('dataset/LargeVLSPData.txt', encoding='utf-8').readlines()
print(plot_tag_counts(train5000_gold))
