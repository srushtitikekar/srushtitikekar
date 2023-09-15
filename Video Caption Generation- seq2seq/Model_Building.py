
import numpy as np
import torch
import random
import os
import json
import pickle
import multiprocessing
import torch.optim as opti
import torch.nn as nnet
from torch.autograd import Variable
import torch.nn.functional as funct
from torch.utils.data import DataLoader, Dataset
import re
from Processing import Process_Train,sentence_split1,process_training_data,Training_Data
import torch.nn.functional as Funct
from scipy.special import expit
from bleu_eval import BLEU
import sys





def to_batch(d):
    d.sort(key=lambda x: len(x[1]), reverse=True)
    video_data, capt = zip(*d) 
    video_data = torch.stack(video_data, 0)

    L = [len(cap) for cap in capt]
    append_zeros = torch.zeros(len(capt), max(L)).long()
    for i, cap in enumerate(capt):
        end = L[i]
        append_zeros[i, :end] = cap[:end]

    return video_data, append_zeros, L


class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])
    def __len__(self):
        return len(self.avi)
    def __getitem__(self, idx):
        return self.avi[idx]


class attention_impl(nnet.Module):
    def __init__(self, h_size):
        super(attention_impl, self).__init__()
        
        self.h_size = h_size
        self.F1 = nnet.Linear(2*h_size, h_size)
        self.F2 = nnet.Linear(h_size, h_size)

        self.to_weight = nnet.Linear(h_size, 1, bias=False)

    def forward(self, h_state, e_out):
        b_size, sequence_len, feat = e_out.size()
        h_state = h_state.view(b_size, 1, feat).repeat(1, sequence_len, 1)
        input_match = torch.cat((e_out, h_state), 2).view(-1, 2*self.h_size)

        data = self.F1(input_match)
        data = self.F2(data)
        att_weights = self.to_weight(data)
        att_weights = att_weights.view(b_size, sequence_len)
        att_weights = Funct.softmax(att_weights, dim=1)
        context = torch.bmm(att_weights.unsqueeze(1), e_out).squeeze(1)
        
        return context


class Encoder(nnet.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.L1 = nnet.Linear(4096, 512)

        self.dropout = nnet.Dropout(0.3)

        self.lstm = nnet.GRU(512, 512, batch_first=True)

    def forward(self, In):
        b_size, sequence_len, feat = In.size()   

        In = In.view(-1, feat)

        In = self.L1(In)

        In = self.dropout(In)

        In = In.view(b_size, sequence_len, 512)

        Out, h_state = self.lstm(In)

        

        return Out, h_state


class Decoder(nnet.Module):
    def __init__(self, h_size, out_size, v_size, w_dim, dropout=0.3):

        super(Decoder, self).__init__()

        self.h_size = 512
        self.out_size = out_size
        self.v_size = v_size
        self.w_dim = w_dim

        self.embedding = nnet.Embedding(out_size, 1024)
        self.dropout = nnet.Dropout(0.3)
        self.lstm = nnet.GRU(h_size+w_dim, h_size, batch_first=True)
        self.attention = attention_impl(h_size)
        print("h_size :",h_size)
        self.final_out = nnet.Linear(h_size, out_size)


    def forward(self, encoder_h_state, encoder_out, targets=None, mode='train',steps=None):
        print(type(encoder_h_state))
        _, b_size, _ = encoder_h_state.size()
        print(type(encoder_h_state))

        if encoder_h_state is None:
            decoder_h_state= None
        else:
            decoder_h_state = encoder_h_state
          
        seq_long=[]
        seq_pred = []
        decoder_in_word = Variable(torch.ones(b_size, 1)).long()
     

        target1 = self.embedding(targets)
        _, sequence_len, _ = target1.size()

        for idx in range(sequence_len-1):
            limit = self.TFR(training_steps=steps)
            if random.uniform(0.05, 0.995) > limit: # returns a random float value between 0.05 and 0.995
                current_in_word = target1[:, idx]  
            else: 
                current_in_word = self.embedding(decoder_in_word).squeeze(1)

            con = self.attention(decoder_h_state, encoder_out)
            lstm_input = torch.cat([current_in_word, con], dim=1).unsqueeze(1)
            lstm_out, decoder_h_state = self.lstm(lstm_input, decoder_h_state)
            prob = self.final_out(lstm_out.squeeze(1))
            seq_long.append(prob.unsqueeze(1))
            decoder_in_word = prob.unsqueeze(1).max(2)[1]

        seq_long = torch.cat(seq_long, dim=1)
        seq_pred = seq_long.max(2)[1]
        return seq_long, seq_pred
        
    def infer(self, encoder_h_state, encoder_out):
        _, b_size, _ = encoder_h_state.size()
        if encoder_h_state is None:
            decoder_h_state= None
        else:
            decoder_h_state = encoder_h_state
        

        decoder_in_word = Variable(torch.ones(b_size, 1)).long()

        seq_Prob = []
        seq_pred= []
        seq_len_assumption=28

        
        for i in range(seq_len_assumption-1):

            current_in_word = self.embedding(decoder_in_word).squeeze(1)
            con = self.attention(decoder_h_state, encoder_out)
            lstm_input = torch.cat([current_in_word, con], dim=1).unsqueeze(1)
            lstm_output, decoder_h_state = self.lstm(lstm_input, decoder_h_state)
            log = self.final_out(lstm_output.squeeze(1))
            seq_Prob.append(log.unsqueeze(1))
            decoder_current_input_word = log.unsqueeze(1).max(2)[1]

        seq_Prob = torch.cat(seq_Prob, dim=1)
        seq_pred = seq_Prob.max(2)[1]
        return seq_Prob, seq_pred

    def TFR(self, training_steps):
        return (expit(training_steps/20 +0.85)) # inverse of the logit function


class Encoder_Decoder(nnet.Module):
    def __init__(self, encoder, decoder):
        super(Encoder_Decoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, video_feat, mode, t_sentences=None, steps=None):
        encoder_out, encoder_h_state = self.encoder(video_feat)
        if mode == 'train':
            seq_Prob, seq_pred = self.decoder(encoder_h_state, encoder_out,t_sentences, mode, steps)
        elif mode == 'inference':
            seq_Prob, seq_pred =  self.decoder.infer(encoder_h_state, encoder_out)
        return seq_Prob, seq_pred

def calc_loss(loss_func, d1, d2, length):
    b_size = len(d1)
    p_cat = None
    groundT = None
    flag = True

    for b in range(b_size):
        predict = d1[b]
        ground_truth = d2[b]
        sequence_len = length[b] -1

        predict = predict[:sequence_len]
        ground_truth = ground_truth[:sequence_len]
        if flag:
            p_cat = predict
            groundT = ground_truth
            flag = False
        else:
            p_cat = torch.cat((p_cat, predict), dim=0)
            groundT = torch.cat((groundT, ground_truth), dim=0)

    loss = loss_func(p_cat, groundT)
    avg_loss = loss/b_size

    return loss, avg_loss


def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    video_data, captions = zip(*data) 
    video_data = torch.stack(video_data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return video_data, targets, lengths

class dataset_test(Dataset):
    def __init__(self, test_filepath):
        self.video = []
        files = os.listdir(test_filepath)
        for f in files:
            key1 = f.split('.npy')[0]
            value1 = np.load(os.path.join(test_filepath, f))
            self.video.append([key1, value1])
    def __len__(self):

        return len(self.video)
    
    def __getitem__(self, index):
        return self.video[index]


def test(test_loader, model, i2w):
    model.eval()
    sentence_all = []
    for batch_idx, batch in enumerate(test_loader):
        idx, video_f = batch

        idx, video_f = idx, Variable(video_f).float()

        seq_logProb, seq_pred = model(video_f, mode='inference')
        test_pred= seq_pred
        
        r= [[i2w[x.item()] if i2w[x.item()] != '<UNK>' else 'something' for x in s] for s in test_pred]
        print(r)
        r = [' '.join(sentence).split('<EOS>')[0] for sentence in r]
        rr = zip(idx, r)
        for k in rr:
            sentence_all.append(k)
    return sentence_all




def train_model(model, epoch, loss_func, model_param, opti,data_loader_train ):
    model.train()
    print('Epoch ', epoch)
    
    for b_index, b in enumerate(data_loader_train):
        
        video_feats, g_truth, len = b
        video_feats, g_truth = Variable(video_feats), Variable(g_truth)
        
        opti.zero_grad()
        seq_Prob, seq_predictions = model(video_feats,t_sentences=g_truth, mode = 'train', steps = epoch)
        g_truth = g_truth[:, 1:]  
        loss,avg_loss = calc_loss(loss_func, seq_Prob, g_truth, len)
        print('Batch -> ', b_index, ' , Loss -> ', loss.item())
        loss.backward()
        opti.step()

    loss = loss.item()
    print('Total Loss : ', loss)
    
def index_to_word_data_forward(I_W):
    return I_W



def process_model(file1,file2):
    indexToword, wordToindex, dictionary = process_training_data()
    #print("indexToword : ",indexToword)
    #print("wordToindex : ",wordToindex)

    pickle_file = open('indexToword.pickle-1', 'wb')
    pickle.dump(indexToword, pickle_file, protocol = pickle.HIGHEST_PROTOCOL)
    

    train_label = '/Users/srushtitikekar/Downloads/MLDS_hw2_1_data/training_data/feat/'
    train_data = '/Users/srushtitikekar/Downloads/MLDS_hw2_1_data/training_label.json'
    train_dataset = Training_Data(train_label, train_data, dictionary, wordToindex)



    data_loader_train = DataLoader(train_dataset, batch_size=150, shuffle=True,collate_fn=to_batch)

    epoch = 10



    encoder = Encoder()

  
    decoder = Decoder(512, len(indexToword) +4, len(indexToword) +4, 1024, 0.3)

    
    model=Encoder_Decoder(encoder,decoder)



    loss_fn = nnet.CrossEntropyLoss()
    parameters = model.parameters()
    Opt = opti.Adam(parameters, lr=0.0001)
    x = nnet.CrossEntropyLoss()
    for i in range(epoch):
        train_model(model, i+1, loss_fn, parameters, Opt, data_loader_train)
    

    filepath = '/Users/srushtitikekar/Downloads/MLDS_hw2_1_data/testing_data/feat'
    dataset = dataset_test('{}'.format(file1))
    data_test_load = DataLoader(dataset, batch_size=150, shuffle=True, num_workers=8)



    testing_loader = DataLoader(dataset, batch_size=150, shuffle=True, num_workers=8)

    ss = test(data_test_load, model, indexToword)

    with open(file2, 'w') as out:
        for id, sentence in ss:
            out.write('{},{}\n'.format(id, sentence))


    test_labels = json.load(open('/Users/srushtitikekar/Downloads/MLDS_hw2_1_data/testing_label.json'))
    output_file = file2
    result = {}
    with open(output_file,'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            id = line[:comma]
            caption = line[comma+1:]
            result[id] = caption

    bleu=[]
    for item in test_labels:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(result[item['id']],captions,True))
        bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    print("Average bleu score is " + str(average))


    
if __name__ == "__main__":

    process_model(sys.argv[1],sys.argv[2])


