# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from transformers import BertConfig, BertTokenizer,BertModel
      
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses import NTXentLoss
 

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

class FineGrainedContrastiveLoss(_Loss):
    def __init__(self,device):
        super(FineGrainedContrastiveLoss, self).__init__()
        self.device = device
        
    def forward(self, batch_data, entity_embedding):
        fgc_loss = 0
        fine_grained_contrastive_data = batch_data.get('fine_grained_contrastive')
        contrastive_count = 0
        for _, data in fine_grained_contrastive_data.items():
            fine_grained_contrastive_entity_index =  data.get('entity_index')
            fine_grained_contrastive_label =  data.get('entity_label_id')
            if len(fine_grained_contrastive_entity_index) == 0:
                continue
            fine_grained_contrastive_entity_index = torch.from_numpy(np.array(fine_grained_contrastive_entity_index)).to(self.device)
            contrastive_entity_embedding = entity_embedding.index_select(0, fine_grained_contrastive_entity_index)
            contrastive_entity_embedding  = contrastive_entity_embedding[:,0,:] 
            fine_grained_contrastive_label = torch.tensor(fine_grained_contrastive_label, device=self.device)
            fgc_loss += NTXentLoss(temperature=0.1)(contrastive_entity_embedding, fine_grained_contrastive_label)
            contrastive_count += 1
            
        fgc_loss = fgc_loss / contrastive_count if contrastive_count else 0
        return fgc_loss
    
class CoarseGrainedContrastiveLoss(_Loss):
    
    def __init__(self,device):
        super(CoarseGrainedContrastiveLoss, self).__init__()
        self.device = device
        
    def forward(self, batch_data, entity_embedding):
        cgc_loss = 0
        coarse_grained_contrastive_data = batch_data.get('coarse_grained_contrastive')
        contrastive_count = 0
        for _, data in coarse_grained_contrastive_data.items():
            coarse_grained_contrastive_entity_index =  data.get('entity_index')
            coarse_grained_contrastive_label =  data.get('entity_label_id')
            if len(coarse_grained_contrastive_entity_index) == 0:
                continue
            coarse_grained_contrastive_entity_index = torch.from_numpy(np.array(coarse_grained_contrastive_entity_index)).to(self.device)
            contrastive_entity_embedding = entity_embedding.index_select(0, coarse_grained_contrastive_entity_index)
            contrastive_entity_embedding  = contrastive_entity_embedding[:,0,:] 
            coarse_grained_contrastive_label = torch.tensor(coarse_grained_contrastive_label, device=self.device)
            cgc_loss += NTXentLoss(temperature=0.1)(contrastive_entity_embedding, coarse_grained_contrastive_label)
            contrastive_count += 1
            
        cgc_loss = cgc_loss / contrastive_count if contrastive_count else 0
        return cgc_loss



class FineGrainedAlignLoss(_Loss):
    
    def __init__(self,device):
        super(FineGrainedAlignLoss, self).__init__()
        self.device = device
        
    def forward(self, batch_data, entity_embedding):
        fine_grained_contrastive_pos_pairs = batch_data.get('fine_grained_contrastive').get('pos_pairs')
        x = [x for x,y in fine_grained_contrastive_pos_pairs]
        y = [y for x,y in fine_grained_contrastive_pos_pairs]
        x = entity_embedding.index_select(0, torch.from_numpy(np.array(x)).to(self.device))
        y = entity_embedding.index_select(0, torch.from_numpy(np.array(y)).to(self.device))
        fine_grained_align_loss = align_loss(x,y)
        
        return fine_grained_align_loss

class FineGrainedUniformLoss(_Loss):
    
    def __init__(self,device):
        super(FineGrainedUniformLoss, self).__init__()
        self.device = device
        
    def forward(self, batch_data, entity_embedding):
        fine_grained_contrastive_neg_pairs = batch_data.get('fine_grained_contrastive').get('neg_pairs')
        fine_grained_uniform_loss = 0
        for i,(x,y) in enumerate(fine_grained_contrastive_neg_pairs):
            x = entity_embedding.index_select(0, torch.from_numpy(np.array(x)).to(self.device))[:,0,:]
            y = entity_embedding.index_select(0, torch.from_numpy(np.array(y)).to(self.device))[:,0,:]
            z = torch.cat([x,y],dim=0)
            fine_grained_uniform_loss += uniform_loss(F.normalize(z))
        fine_grained_uniform_loss = fine_grained_uniform_loss / (i+1)
        return fine_grained_uniform_loss
   
    
class CoarseGrainedAlignLoss(_Loss):
    
    def __init__(self,device):
        super(CoarseGrainedAlignLoss, self).__init__()
        self.device = device
        
    def forward(self, batch_data, entity_embedding):
        
        coarse_grained_contrastive_pos_pairs = batch_data.get('coarse_grained_contrastive').get('pos_pairs')
        
        x = [x for x,y in coarse_grained_contrastive_pos_pairs]
        y = [y for x,y in coarse_grained_contrastive_pos_pairs]
        x = entity_embedding.index_select(0, torch.from_numpy(np.array(x)).to(self.device))
        y = entity_embedding.index_select(0, torch.from_numpy(np.array(y)).to(self.device))
        coarse_grained_align_loss = align_loss(x,y)
        
        return coarse_grained_align_loss

class CoarseGrainedUniformLoss(_Loss):
    
    def __init__(self,device):
        super(CoarseGrainedUniformLoss, self).__init__()
        self.device = device
        
    def forward(self, batch_data, entity_embedding):
        coarse_grained_contrastive_neg_pairs = batch_data.get('coarse_grained_contrastive').get('neg_pairs')
        coarse_grained_uniform_loss = 0
        for i,(x,y) in enumerate(coarse_grained_contrastive_neg_pairs):
            x = entity_embedding.index_select(0, torch.from_numpy(np.array(x)).to(self.device))[:,0,:]
            y = entity_embedding.index_select(0, torch.from_numpy(np.array(y)).to(self.device))[:,0,:]
            z = torch.cat([x,y],dim=0)
            coarse_grained_uniform_loss += uniform_loss(F.normalize(z,dim=1))
                
        coarse_grained_uniform_loss = coarse_grained_uniform_loss / (i+1)
        return coarse_grained_uniform_loss
    

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

    
class EntityTypingLosses(nn.Module):
    def __init__(self,
                 device,
                 task=['tag', 'cgc', 'fgc'],
                 task_weight=[1,1,1],
                 task_weight_type='auto',
                 calc_align_uniform_loss=False
                 ):
        super(EntityTypingLosses, self).__init__()
        self.task = task
        self.task_weight = task_weight
        self.task_weight_type = task_weight_type
        self.calc_align_uniform_loss = calc_align_uniform_loss
        
        self.tag_loss_func = nn.BCEWithLogitsLoss()
        self.fgc_loss_func = FineGrainedContrastiveLoss(device)
        self.cgc_loss_func = CoarseGrainedContrastiveLoss(device)
        
        self.fine_grained_align_loss_func = FineGrainedAlignLoss(device)
        self.fine_grained_uniform_loss_func = FineGrainedUniformLoss(device)
        self.coarse_grained_align_loss_func = CoarseGrainedAlignLoss(device)
        self.coarse_grained_uniform_loss_func = CoarseGrainedUniformLoss(device)
        self.awl = AutomaticWeightedLoss(len(task))
        
        
        if self.task_weight_type == 'auto':
            self.task_weight = '1,1,1'
    
    def forward(self,*inputs):
        logits,targets,batch_data, entity_embedding = inputs
        
        loss = []
        tag_loss = self.tag_loss_func(logits,targets)
        loss.append(tag_loss * self.task_weight[0])
        
        
        if 'fgc' in self.task:
            fgc_loss = self.fgc_loss_func(batch_data, entity_embedding)
            loss.append(fgc_loss * self.task_weight[1])
        else:
            fgc_loss = 0
            
        if 'cgc' in self.task:
            cgc_loss = self.cgc_loss_func(batch_data, entity_embedding)
            loss.append(cgc_loss * self.task_weight[2])
        else:
            cgc_loss = 0
        
        if self.calc_align_uniform_loss:
            fine_grained_align_loss = self.fine_grained_align_loss_func(batch_data, entity_embedding)
            fine_grained_uniform_loss = self.fine_grained_uniform_loss_func(batch_data, entity_embedding)
            coarse_grained_align_loss = self.coarse_grained_align_loss_func(batch_data, entity_embedding)
            coarse_grained_uniform_loss = self.coarse_grained_uniform_loss_func(batch_data, entity_embedding)
        else:
            fine_grained_align_loss = 0
            fine_grained_uniform_loss = 0
            coarse_grained_align_loss = 0
            coarse_grained_uniform_loss = 0
            
        if self.task_weight_type == 'auto':
            loss = self.awl(*loss)
        else:
            loss = sum(loss)
        loss_all = {
                'loss': loss,
                'tag_loss': tag_loss,
                'fgc_loss': fgc_loss,
                'cgc_loss': cgc_loss,
                'fine_grained_align_loss': fine_grained_align_loss,
                'fine_grained_uniform_loss': fine_grained_uniform_loss,
                'coarse_grained_align_loss': coarse_grained_align_loss,
                'coarse_grained_uniform_loss': coarse_grained_uniform_loss
                
            
            }
        return loss_all

class EntityEmbedding(nn.Module):
    def __init__(self):
        super(EntityEmbedding, self).__init__()
        
    def get_position(self, x,value=None,return_start=True):        
        y = torch.zeros(x.shape).to(x.device)
        y[x == value] = 1
        z = (y == 1).nonzero(as_tuple = False)
        o = z.view(-1).tolist()
        if return_start:
            return o[:1]
        return o


    def get_entity_embedding(self, sequence_output,segment_ids,entity_embedding_type='ENTITY'):
        embedding_all_entity = []
        current_batch_size = sequence_output.shape[0]
        for batch_idx  in range(current_batch_size):
            if entity_embedding_type == 'ENTITY':
                entity_index = self.get_position(segment_ids[batch_idx ],1,return_start=True)
                embedding_et = sequence_output[batch_idx , entity_index, :].unsqueeze(0)
            elif entity_embedding_type == 'CLS':
                embedding_et = sequence_output[batch_idx , 0:1, :].unsqueeze(0)
            elif entity_embedding_type == 'entity_mean':
                entity_index = self.get_position(segment_ids[batch_idx ],1,return_start=False)[1:]
                embedding_et = sequence_output[batch_idx , entity_index, :].mean(0,keepdim=True).unsqueeze(0)
            elif entity_embedding_type == 'ENTITY_CLS_cat':
                entity_index = self.get_position(segment_ids[batch_idx ],1,return_start=True)
                embedding_ENTITY = sequence_output[batch_idx , entity_index, :].unsqueeze(0)
                embedding_CLS = sequence_output[batch_idx , 0:1, :].unsqueeze(0)
                embedding_et = torch.cat((embedding_ENTITY, embedding_CLS), dim=2)
            elif entity_embedding_type == 'entity_mean_CLS_cat':
                entity_index = self.get_position(segment_ids[batch_idx ],1,return_start=False)[1:]
                embedding_entity_mean = sequence_output[batch_idx, entity_index, :].mean(0,keepdim=True).unsqueeze(0)
                embedding_CLS = sequence_output[batch_idx , 0:1, :].unsqueeze(0)
                embedding_et = torch.cat((embedding_entity_mean, embedding_CLS), dim=2)    
              
            embedding_all_entity.append(embedding_et)
        embedding_all_entity = torch.cat(embedding_all_entity, 0)
        
        return embedding_all_entity

    def forward(self, *inputs):
        sequence_output,segment_ids,entity_embedding_type = inputs
        entity_embedding = self.get_entity_embedding(sequence_output,segment_ids,entity_embedding_type)
        return entity_embedding
    

class EntityTypingModel(nn.Module):
    def __init__(self, 
                 num_class, 
                 model_path, 
                 device,
                 task=['tag', 'cgc', 'fgc'], 
                 task_weight=[1,1,1],
                 task_weight_type='auto',
                 entity_embedding_type='ENTITY',
                 ):
        super().__init__()
        
        if type(task) == str:
            self.task = [task]
        elif type(task) == list:
            self.task = list(set(task))
        self.task_weight = task_weight
        self.task_weight_type = task_weight_type
        self.device = device
        
        config = BertConfig.from_pretrained(model_path)
        self.model = BertModel(config).from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[ENTITY]"]})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.newfc_tag = nn.Linear(config.hidden_size, num_class)
        
        calc_align_uniform_loss = False
        self.entity_typing_loss_func = EntityTypingLosses(self.device,self.task,
                                                          self.task_weight,
                                                          self.task_weight_type,
                                                          calc_align_uniform_loss)
        self.entity_embedding_func = EntityEmbedding()
        self.entity_embedding_type = entity_embedding_type
        
        
    def forward(self, inputs):
        input_ids = inputs.get('batch_token_ids').to(self.device)
        attn_mask = inputs.get('batch_attention_mask').to(self.device)
        segment_ids = inputs.get('batch_token_type_ids').to(self.device)
        target = inputs.get('batch_label_ids').to(self.device)
            
        sequence_output  = self.model(input_ids, attn_mask, segment_ids)[0]
        pred = self.newfc_tag(sequence_output[:, 0, :])
        
        # compute train task loss     
        if self.training:
            entity_embedding = self.entity_embedding_func(sequence_output,segment_ids,self.entity_embedding_type)
            entity_typing_loss = self.entity_typing_loss_func(pred,target,inputs, entity_embedding)
            return pred, entity_typing_loss
        else:
            return pred
    

        