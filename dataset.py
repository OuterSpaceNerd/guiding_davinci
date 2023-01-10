import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np 
from transformers import BertTokenizerFast
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import tensorflow as tf

class ChineseDataset(Dataset):
    def __init__(self, data_pth, toker, maxline=False):
        fout = open(data_pth, 'r')
        lines = fout.readlines()
        tokens = []
        masks = []
        labels = []
        first_inputs = []
        first_masks = []
        self.ll = []
        token_type_ids = []
        position_ids = []

        if maxline == False:
            maxline = len(lines)
        
        for i in tqdm(range(maxline)):
            try:
                src = (lines[i].split('\t')[0]).strip()
                tgt = (lines[i].split('\t')[1]).strip()
                src_token = toker.encode(src)
                tgt_token = toker.encode(tgt)
            except: # only src
                continue
           
            # try:
            #     src_token = toker.encode(src)
            #     tgt_token = toker.encode(tgt)
            # except:
            #     continue # nan
            temp_token = src_token + tgt_token[1:] # CLS sent1 SEP sent2 SEP 
            # temp_mask = [1] * len(temp_token)
            temp_mask = [1 for i in range(len(temp_token))]
            # temp_label = [-100] * (len(src_token)) + tgt_token[1:]
            temp_label = [-100 for i in range(len(src_token))] + tgt_token[1:] 
#             temp_label = temp_token  ## change to all sentences
            if len(temp_token) >= 40: continue
            # if len(src_token) >= 25: continue
            tokens.append(temp_token[:])
            masks.append(temp_mask[:])
            labels.append(temp_label[:])
            first_inputs.append(src_token[:-1])
            first_masks.append([1 for i in range(len(src_token[:-1]))] )
            self.ll.append(len(src_token[:-1])) ##?
            token_type_ids.append([0 for i in range(len(src_token))] + [1 for i in range(len(tgt_token[1:]))])
            position_ids.append(list(range(len(temp_token))))


        # self.post = pad_sequence([torch.LongTensor(x) for x in tokens], batch_first=True, padding_value=0)
        # self.mask = pad_sequence([torch.LongTensor(x) for x in masks], batch_first=True, padding_value=0)
        # self.label = pad_sequence([torch.LongTensor(x) for x in labels], batch_first=True, padding_value=-100)
        # self.first_input = pad_sequence([torch.LongTensor(x) for x in first_inputs], batch_first=True, padding_value=0)
        # self.first_mask = pad_sequence([torch.LongTensor(x) for x in first_masks], batch_first=True, padding_value=0)
        # self.token_type_id = pad_sequence([torch.LongTensor(x) for x in token_type_ids], batch_first=True, padding_value=0)
        # self.position_id = pad_sequence([torch.LongTensor(x) for x in position_ids], batch_first=True, padding_value=0)
        self.post = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in tokens], value=0))
        self.mask = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in masks], value=0))
        self.label = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in labels], value=-100))
        self.first_input = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in first_inputs], value=0))
        self.first_mask = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in first_masks], value=0))
        self.token_type_id = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in token_type_ids], value=0))
        self.position_id = torch.LongTensor(tf.keras.preprocessing.sequence.pad_sequences([torch.LongTensor(x) for x in position_ids], value=0))

        
    def __getitem__(self, index):

        return self.post[index], self.mask[index], self.label[index], self.token_type_id[index], self.first_input[index], self.first_mask[index], self.ll[index], self.position_id[index]

    def __len__(self):
        return len(self.post)
    
    def sample(self, x):
        idxs = torch.randint(high=len(self.post), size=(x,))
#         print("idxs: ", idxs)
        
        postbatch = torch.LongTensor([self.post[idx].tolist() for idx in idxs])
        maskbatch = torch.LongTensor([self.mask[idx].tolist() for idx in idxs])
        labelbatch = torch.LongTensor([self.label[idx].tolist() for idx in idxs])
        token_type_idbatch = torch.LongTensor([self.token_type_id[idx].tolist() for idx in idxs])
        first_inputbatch = torch.LongTensor([self.first_input[idx].tolist() for idx in idxs])
        first_maskbatch = torch.LongTensor([self.first_mask[idx].tolist() for idx in idxs])
        llbatch = torch.LongTensor([self.ll[idx] for idx in idxs])
        position_idbatch = torch.LongTensor([self.position_id[idx].tolist() for idx in idxs])
        
        return postbatch, maskbatch, labelbatch, token_type_idbatch, first_inputbatch, first_maskbatch, llbatch, position_idbatch


if __name__ == '__main__':
    data_pth = 'data/traditional_corpus/test-weibo-v3.tsv'
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    dataset = ChineseDataset(data_pth, tokenizer)

    print(len(dataset))
    
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    for inputs_id, mask, label, ll in train_dataloader:
        print(type(inputs_id), inputs_id.shape)
        print(inputs_id[0])
        print(mask[0])
        print(label[0])
        print(ll[0])
        break


