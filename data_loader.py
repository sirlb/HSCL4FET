# -*- coding: utf-8 -*-
import json
import random
import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

def data_path(data_file):
    if data_file == "bbn":
        train_data_file = "data/box4et/bbn/bbn_train.json"
        dev_data_file = "data/box4et/bbn/bbn_dev.json"
        test_data_file = "data/box4et/bbn/bbn_test.json"
        type_vocab_file = 'data/box4et/ontology/bbn_types.txt'
    elif data_file == "figer":
        train_data_file = "data/box4et/figer/figer_train.json"
        dev_data_file = "data/box4et/figer/figer_dev.json"
        test_data_file = "data/box4et/figer/figer_test.json"
        type_vocab_file = 'data/box4et/ontology/figer_types.txt'

    elif data_file == "ontonote":
        train_data_file = "data/box4et/onto/onto_train.json"
        dev_data_file = "data/box4et/onto/onto_dev.json"
        test_data_file = "data/box4et/onto/onto_test.json"
        type_vocab_file = 'data/box4et/ontology/ontonotes_types.txt'
        
    return train_data_file, dev_data_file, test_data_file, type_vocab_file


def load_data(data_path):
    data = []
    with open(data_path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            left_context_text = line['left_context_text']
            word = line['word']
            right_context_text = line['right_context_text']
            y_category = line['y_category']
            data.append((i,[left_context_text,word,right_context_text],y_category))
    return data

def load_vocab_file(filename):
    with open(filename, encoding='utf-8') as f:
        vocab = [line.strip() for line in f]
    id2type = {i: t for i, t in enumerate(vocab)}
    type2id = {t: i for i, t in enumerate(vocab)}
    return id2type, type2id


def truncate_sequences(sequences, maxlen):
    left_context_tokens,word_tokens,right_context_tokens = sequences
    seq_len = len(left_context_tokens) + len(word_tokens) + len(right_context_tokens) + len(word_tokens) + 3
    if seq_len > maxlen:
        ex_len = seq_len - maxlen
        right_context_tokens = right_context_tokens[:max(0, len(right_context_tokens) - ex_len)]
        ex_len = len(left_context_tokens) + len(word_tokens) + len(right_context_tokens) + len(word_tokens) + 3 - maxlen
        if ex_len > 0:
            left_context_tokens = left_context_tokens[min(ex_len, len(left_context_tokens) - 1):]
        ex_len = len(left_context_tokens) + len(word_tokens) + len(right_context_tokens) + len(word_tokens) + 3 - maxlen
        if ex_len > 0:
            word_tokens = []
            word_tokens = word_tokens[:maxlen-3]
    sequences =  ['[CLS]'] + left_context_tokens + word_tokens + right_context_tokens + ['[ENTITY]'] + word_tokens + ['[SEP]']
    return sequences


def sequence_padding(sequences, batch_first=True, padding_value=0):
    if isinstance(sequences,list):
        sequences = [torch.from_numpy(np.array(sequence)) for sequence in sequences]
    sequences = pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
    return sequences

def onehot_encode(batch_class_ids, n_classes):
    batch_size = len(batch_class_ids)
    encode_y = np.zeros((batch_size, n_classes), dtype=np.float32)
    for index, class_ids in enumerate(batch_class_ids):
        for cid in class_ids:
            encode_y[index][cid] = 1.0
    encode_y = torch.from_numpy(encode_y)
    return encode_y



def get_same_index(list_):
    colunmDict={}
    for i in zip(list_,list(range(len(list_)))):
        if list_.count(i[0]) > 1:
            if i[0] not in colunmDict:
                colunmDict.update({i[0]:[i[1]]})
            else:
                colunmDict[i[0]].append(i[1])
    return colunmDict 

def remove_entity(entity_index,remove_entity_index):
    entity_index_set = set([i for i in range(len(entity_index))])
    remove_entity_index_set = set(remove_entity_index)
    index = list(entity_index_set - remove_entity_index_set)
    entity_index = [entity_index[i] for i in index]
    return entity_index
  


def pos_neg_pairs(labels):
    if isinstance(labels, list):
        labels = torch.from_numpy(np.array(labels))
    indices_tuple = lmu.get_all_pairs_indices(labels)
    a1, p, a2, n = indices_tuple
    
    pos_pairs = [(i,j) for i,j in zip(a1.tolist(),p.tolist())]
    neg_pairs = [(i,j) for i,j in zip(a2.tolist(),n.tolist())]
    return pos_pairs, neg_pairs
    



class DataBatchLoader:
    def __init__(self, 
                 samples, 
                 batch_size, 
                 n_iter, 
                 type2id,
                 tokenizer,
                 use_more_granular,
                 fine_grained_type,
                 coarse_grained_type,
                 dataset = 'bbn', 
                 maxlen=128, 
                 shuffle=False, 
                 n_steps=-1):
        self.samples = samples
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.n_samples = len(self.samples)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
       
        self.n_steps = n_iter * self.n_batches
        if n_steps > 0:
            self.n_steps = n_steps
        self.maxlen = maxlen
        self.dataset = dataset
        
        self.type2id = type2id
        self.tokenizer = tokenizer
        self.use_more_granular = use_more_granular
        self.fine_grained_type = fine_grained_type
        self.coarse_grained_type = coarse_grained_type
        
    def __iter__(self):
        return self.batch_iter()

    def batch_iter(self):
        for step in range(self.n_steps):
            batch_idx = step % self.n_batches
            batch_beg, batch_end = batch_idx * self.batch_size,   \
                                   min((batch_idx + 1) * self.batch_size, self.n_samples)
            batch_data = self.samples[batch_beg:batch_end]
            #  
            if batch_beg == self.n_samples and self.shuffle: 
                random.shuffle(self.samples)

            batch_data = self.collect_fn(batch_data)
            yield batch_data
            
            
    def collect_fn(self, batch_data):
        batch_tokens, batch_token_ids, batch_labels, batch_label_ids = [], [], [], []
        batch_attention_mask, batch_token_type_ids = [], []
        batch_data_dict = dict()
        
        for data in batch_data:
            sample_index, sample_x, sample_y = data
            
            left_context_text, word, right_context_text = sample_x
            sample_x = self.tokenizer.tokenize(left_context_text), \
                        self.tokenizer.tokenize(word), \
                        self.tokenizer.tokenize(right_context_text)
            sample_x = truncate_sequences(sample_x, self.maxlen)  
            sample_x_id = self.tokenizer.convert_tokens_to_ids(sample_x)
            sample_y_id = [self.type2id[y] for y in sample_y]
            
            entity_id_index = sample_x.index('[ENTITY]') 
            
            batch_tokens.append(sample_x)
            batch_token_ids.append(sample_x_id)
            batch_labels.append(sample_y)
            batch_label_ids.append(sample_y_id)
        
            batch_attention_mask.append([1] * len(sample_x_id))
            batch_token_type_ids.append([0] * (entity_id_index - 1) + 
                                        [1] * (len(sample_x_id) - (entity_id_index -1))
                                        )
        batch_label_src = batch_label_ids
        batch_label_ids = onehot_encode(batch_label_ids, len(self.type2id))
        batch_label_ids = sequence_padding(batch_label_ids) 
        batch_token_ids = sequence_padding(batch_token_ids)
        
        batch_attention_mask = sequence_padding(batch_attention_mask)
        batch_token_type_ids = sequence_padding(batch_token_type_ids)
        
        batch_data_dict.update(
                {
                    'batch_tokens': batch_tokens,
                    'batch_token_ids': batch_token_ids,
                    'batch_attention_mask': batch_attention_mask,
                    'batch_token_type_ids': batch_token_type_ids,
                    'batch_labels': batch_labels,
                    'batch_label_ids': batch_label_ids,
                    'batch_label_src': batch_label_src
                    
                    }
            )
        # 1.
        data = self.fine_grained_contrastive_collect_fn(batch_labels)
        batch_data_dict.update(data)
        # 2. 
        data = self.coarse_grained_contrastive_collect_fn(batch_labels)
        batch_data_dict.update(data)
        #print(batch_token_type_ids.shape)
        
        return batch_data_dict
    
    # 
    def fine_grained_contrastive_collect_fn(self, batch_labels):
        def processing_contrastive_data(coarse_grained_types,type_index):
            contrastive_data = dict()
            for coarse_grained_type in coarse_grained_types:
                entity_index = []
                entity_label = []
                for i, labels in enumerate(batch_labels):
                    for label in labels:
                        sub_label = label.split("/")
                        if len(sub_label) >= type_index: 
                            entity_index.append(i)
                            #entity_label.append('/'.join(sub_label[:type_index]))
                            entity_label.append(label)
                            
                remove_entity_index = sum([v for i,v in get_same_index(entity_index).items()],[])
                entity_index = remove_entity(entity_index,remove_entity_index)
                entity_label = remove_entity(entity_label,remove_entity_index) 
                entity_label_id = [self.type2id[label] for label in entity_label]
                pos_pairs, neg_pairs = pos_neg_pairs(entity_label_id)
                #  
                if len(set(entity_label)) < 2 and len(entity_label) == len(set(entity_label)):   
                    continue
                    
                
                contrastive_data.update(
                    {
                        coarse_grained_type:{
                                'entity_index': entity_index,
                                'entity_label': entity_label,
                                'entity_label_id': entity_label_id,
                                'pos_pairs': pos_pairs,
                                'neg_pairs': neg_pairs
                        }
                    
                    })
            return contrastive_data
     
        if self.fine_grained_type == 'all':
            coarse_grained_types = ['all']
        elif self.fine_grained_type == 'same_coarse_grained_calculate':
            if self.use_more_granular:
                coarse_grained_types = [label.split("/")[2] for i, labels in enumerate(batch_labels) for label in labels]
            else:
                coarse_grained_types = [label.split("/")[1] for i, labels in enumerate(batch_labels) for label in labels]
            coarse_grained_types = list(set(coarse_grained_types))
            #print(coarse_grained_types)
        if self.use_more_granular:
            type_index = 4
        else:
            type_index = 3
        contrastive_data = processing_contrastive_data(coarse_grained_types,type_index)
        data = {'fine_grained_contrastive':contrastive_data}
        return data
        
    def coarse_grained_contrastive_collect_fn(self, batch_labels):
        def processing_contrastive_data(coarse_grained_types,type_index):
            contrastive_data = dict()
            for coarse_grained_type in coarse_grained_types:
                entity_index = []
                entity_label = []
                for i, labels in enumerate(batch_labels):
                    for label in labels:
                        sub_label = label.split("/")
                        if len(sub_label) >= type_index: 
                            entity_index.append(i)
                            entity_label.append('/'.join(sub_label[:type_index]))
                            
                            
                remove_entity_index = sum([v for i,v in get_same_index(entity_index).items()],[])
                entity_index = remove_entity(entity_index,remove_entity_index)
                entity_label = remove_entity(entity_label,remove_entity_index)     
                entity_label_id = [self.type2id[label] for label in entity_label]
                pos_pairs, neg_pairs = pos_neg_pairs(entity_label_id)
                
                #  
                if len(set(entity_label)) < 2 and len(entity_label) == len(set(entity_label)):   
                    entity_index = []
                    entity_label = []
                    entity_label_id = []
                    pos_pairs = []
                    neg_pairs = []
                    
                contrastive_data.update(
                    {
                        coarse_grained_type:{
                                'entity_index': entity_index,
                                'entity_label': entity_label,
                                'entity_label_id': entity_label_id,
                                'pos_pairs': pos_pairs,
                                'neg_pairs': neg_pairs
                        }
                    
                    })
            return contrastive_data
        
        if self.coarse_grained_type == 'all':
            coarse_grained_types = ['all']
    
        if self.use_more_granular:
            type_index = 3
        else:
            type_index = 2
        contrastive_data = processing_contrastive_data(coarse_grained_types,type_index)
        data = {'coarse_grained_contrastive':contrastive_data}
        

        return data
