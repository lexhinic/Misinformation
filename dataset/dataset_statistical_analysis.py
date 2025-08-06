import json 
import string

'''
This script is used to analyze the dataset and count the number of samples, vocabulary size and number of words.
'''
# train dataset
#samples = []
count = 0
text = ''
with open("/home/stu4/Misinformation/dataset/train.jsonl", "r") as file:
    for line in file:
        data = json.loads(line.rstrip())
        #samples.append(data)
        count += 1
        #if count == 10:
            #break
        text += data["claim"]
        
#print(samples)
print("Number of samples in train dataset: ", count)

def count_vocabulary_size(s):
    s = s.lower()
    translator = str.maketrans('', '', string.punctuation)
    s = s.translate(translator)
    words = s.split()
    vocabulary = set(words)
    return (len(vocabulary), len(words))

print("Vocabulary size of train dataset: ", count_vocabulary_size(text)[0])
print("Number of words in train dataset: ", count_vocabulary_size(text)[1])

# test dataset
#samples = []
count = 0
text = ''
with open("/home/stu4/Misinformation/dataset/shared_task_test.jsonl", "r") as file:
    for line in file:
        data = json.loads(line.rstrip())
        #samples.append(data)
        count += 1
        #if count == 10:
            #break
        text += data["claim"]
        
#print(samples)
print("Number of samples in test dataset: ", count)

print("Vocabulary size of test dataset: ", count_vocabulary_size(text)[0])
print("Number of words in test dataset: ", count_vocabulary_size(text)[1])

