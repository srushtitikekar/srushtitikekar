import json
import numpy as np
import os
import torch
import re
from torch.utils.data import Dataset
def process_training_data():
    filepath = '/Users/srushtitikekar/Downloads/MLDS_hw2_1_data/'
    with open(filepath + 'training_label.json', 'r') as f:
        file = json.load(f)

    w_count = {}
    for data in file:
        for sentence in data['caption']:
            word_sent = re.sub('[.!,;?]]', ' ', sentence).split()
            for w in word_sent:
                w = w.replace('.', '') if '.' in w else w
                if w in w_count:
                    w_count[w] += 1
                else:
                    w_count[w] = 1

    D = {}
    for w in w_count:
        if w_count[w] > 4:
            D[w] = w_count[w]
    tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i_w = {i + len(tokens): w for i, w in enumerate(D)}
    w_i = {w: i + len(tokens) for i, w in enumerate(D)}
    for token, index in tokens:
        i_w[index] = token
        w_i[token] = index
    return i_w, w_i, D

def Process_Train():
    word_cnt= {}
    dictionary = {}

    with open('/Users/srushtitikekar/Downloads/MLDS_hw2_1_data/training_label.json', 'r') as f:
        
        file = json.load(f)

    # Process Data Cleaning for the training Labels like removing full stops and tokenize the data

    for f in file:
        for sent in f['caption']:
            sentence = sent.split()
            for w in sentence:

                #w = w.replace('.', '') 
                w = w.replace('.', '') if '.' in w else w
                if w in word_cnt:
                    word_cnt[w] += 1
                else:
                    word_cnt[w] = 1

    
    for w in word_cnt:
        if word_cnt[w] > 3:
            word_cnt[w] = word_cnt[w]
    tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    index_word = {i + len(tokens): w for i, w in enumerate(dictionary)}
    word_index = {w: i + len(tokens) for i, w in enumerate(dictionary)}
    for token, idx in tokens:
        word_index[idx] = token
        index_word[idx] = idx
        
    return index_word, word_index, dictionary

def sentence_split1(statement, dictionary, word_index):
    statement =statement.split()
    for i in range(len(statement)):
        if statement[i] not in dictionary:
            statement[i] = 3
        else:
            statement[i] = word_index[statement[i]]
    statement.insert(0, 1)
    statement.append(2)
    return statement


def doPair(train_label, dictionary, word_index):
    pairs=[]
    #train_label = '/Users/srushtitikekar/Downloads/MLDS_hw2_1_data/training_label.json'
    with open(train_label, 'r') as file:
        label = json.load(file)
    for l in label:
        for sent in l['caption']:
            sent = sentence_split1(sent, dictionary, word_index)
            pairs.append((l['id'], sent))
    return pairs

#    label_json = '/Users/srushtitikekar/Downloads/MLDS_hw2_1_data/' + label_file
#    caption_new=[]
#    with open(label_json, 'r') as f:
#        label = json.load(f)
#    for d in label:
#        for s in d['caption']:
#            s = sentence_split1(s, word_dict, w2i)
#            annotated_caption.append((d['id'], s))
#    return annotated_caption

def video_npy(train_data):
    video_data = {}
    all_files = os.listdir(train_data)
    for f in all_files:
        v = np.load(os.path.join(train_data, f))
        video_data[f.split('.npy')[0]] = v
    return video_data


class Training_Data(Dataset):

    def __init__(self, train_labels, train_file, directory, word_index):


        self.train_labels = train_labels
        self.train_file = train_file
        self.dictionary= directory
        self.video = video_npy(train_labels)
        self.word_index = word_index
        self.data_pair = doPair(train_file, directory, word_index)
        
    def __len__(self):
        return len(self.data_pair)
    
    def __getitem__(self, index):
        assert (index < self.__len__())
        video_file_name, sent = self.data_pair[index]
        d = torch.Tensor(self.video[video_file_name])
        d += torch.Tensor(d.size()).random_(0, 2000)/10000.
        return torch.Tensor(d), torch.Tensor(sent)


