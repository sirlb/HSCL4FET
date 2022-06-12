# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import torch

import argparse
from model import EntityTypingModel
from data_loader import DataBatchLoader,load_data,load_vocab_file,data_path

def set_seed(seed=None):
    def __set_seed__(seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  
            torch.cuda.manual_seed_all(seed) 
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
    if seed is None:
        seed = random.randint(0,100000)
    __set_seed__(seed)
    return seed


  
def configure_optimizers(named_params, learning_rate, w_decay):
    from transformers.optimization import AdamW
    if w_decay == 0:
        return AdamW([p for _, p in named_params], lr=learning_rate, correct_bias=False)
    no_decay = ['bias', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': w_decay},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    assert len(optimizer_grouped_parameters[1]['params']) != 0
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, weight_decay=w_decay, correct_bias=False)
    return optimizer


class Metrics():
    
    def calc_strict_f1(self,true_and_prediction):
        num_entities = len(true_and_prediction)
        correct_num = 1e-8
        for true_labels, predicted_labels in true_and_prediction:
            correct_num += set(true_labels) == set(predicted_labels)
        strict_precision = strict_recall = correct_num / (num_entities + 1e-8)
        strict_f1 = 2 * strict_precision * strict_recall / float(strict_precision + strict_recall + 1e-8)
        return strict_precision, strict_recall, strict_f1



    def calc_loose_macro_f1(self,true_and_prediction):
        p, r = 0., 0.
        pred_example_count, gold_example_count = 1e-8, 1e-8
        pred_label_count = 0.
        for true_labels, predicted_labels in true_and_prediction:
            if len(predicted_labels) > 0:
                pred_example_count += 1
                pred_label_count += len(predicted_labels)
                per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
                p += per_p
            if len(true_labels) > 0:
                gold_example_count += 1
                per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
                r += per_r
                
        macro_precision = p / pred_example_count
        macro_recall = r / gold_example_count
        macro_f1 = 2 * macro_precision * macro_recall / float(macro_precision + macro_recall + 1e-8)
        return macro_precision, macro_recall, macro_f1
    
    
    def calc_loose_micro_f1(self,true_and_prediction):
        num_predicted_labels = 1e-8
        num_true_labels = 1e-8
        num_correct_labels = 0
        pred_example_count = 0
        for true_labels, predicted_labels in true_and_prediction:
            if len(predicted_labels) > 0:
                pred_example_count += 1
            num_predicted_labels += len(predicted_labels)
            num_true_labels += len(true_labels)
            num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
    
        if pred_example_count == 0:
            return 0,0,0
        micro_precision = num_correct_labels / num_predicted_labels
        micro_recall = num_correct_labels / num_true_labels
        micro_f1 = 2 * micro_precision * micro_recall / float(micro_precision + micro_recall + 1e-8)
        return micro_precision, micro_recall, micro_f1
        

def get_contrastive_label():
    if use_more_granular:
        type_index = 4
    else:
        type_index = 3
        
    coarse_grained_label = [] 
    fine_grained_label = [] 
    for label in type2id:
        sub_label = label.split("/")
        if len(sub_label) >= type_index: 
            fine_grained_label.append(label)
        else:
            coarse_grained_label.append(label)
            
    participate_fine_grained_contrastive_label = []
    participate_coarse_grained_contrastive_label = []  
    batch_iter = DataBatchLoader(train_data, batch_size, n_iter=1,
                                 type2id=type2id,
                                 tokenizer=tokenizer,
                                 use_more_granular=use_more_granular,
                                 fine_grained_type=fine_grained_type,
                                 coarse_grained_type=coarse_grained_type,
                                 dataset=dataset,shuffle=True)
    for batch in batch_iter:
        fine_grained_contrastive = batch.get('fine_grained_contrastive')
        coarse_grained_contrastive = batch.get('coarse_grained_contrastive')
        for key in fine_grained_contrastive:
            participate_fine_grained_contrastive_label.extend(
                                fine_grained_contrastive[key]['entity_label']
                             )
        for key in coarse_grained_contrastive:
            participate_coarse_grained_contrastive_label.extend(
                                coarse_grained_contrastive[key]['entity_label']
                            )
    
    participate_fine_grained_contrastive_label = list(set(participate_fine_grained_contrastive_label))
    participate_coarse_grained_contrastive_label = list(set(participate_coarse_grained_contrastive_label))
    no_participate_fine_grained_contrastive_label = \
            [label for label in type2id if label not in participate_fine_grained_contrastive_label]
    no_participate_coarse_grained_contrastive_label = \
            [label for label in type2id if label not in participate_coarse_grained_contrastive_label]
    
    label = {
        
            'fine_grained_label':fine_grained_label,
            'coarse_grained_label':coarse_grained_label,
            'participate_fine_grained_contrastive_label':participate_fine_grained_contrastive_label,
            'participate_coarse_grained_contrastive_label':participate_coarse_grained_contrastive_label,
            'no_participate_fine_grained_contrastive_label':no_participate_fine_grained_contrastive_label,
            'no_participate_coarse_grained_contrastive_label':no_participate_coarse_grained_contrastive_label
        
        }
    
    return label
    
class Evaluate():
    
    def __init_(self):
        self.metrics = Metrics()
 
        
        
    def evaluate(self, model, batch_iter):
        fine_grained_label = contrastive_label.get('fine_grained_label')
        coarse_grained_label = contrastive_label.get('coarse_grained_label')
        participate_fine_grained_contrastive_label = contrastive_label.get('participate_fine_grained_contrastive_label')
        participate_coarse_grained_contrastive_label = contrastive_label.get('participate_coarse_grained_contrastive_label')
        no_participate_fine_grained_contrastive_label = contrastive_label.get('no_participate_fine_grained_contrastive_label')
        no_participate_coarse_grained_contrastive_label = contrastive_label.get('no_participate_coarse_grained_contrastive_label')
    
        
        model.eval()
        gp_tups = list()
        for batch in batch_iter:
            with torch.no_grad():
                logits_batch = model(batch)
            logits_batch = logits_batch.data.cpu().numpy()  
            batch_label_src = batch.get('batch_label_src') 
            for i, logits in enumerate(logits_batch):
                idxs = np.squeeze(np.argwhere(logits > 0), axis=1)
                if len(idxs) == 0:
                    idxs = [np.argmax(logits)]
                if isinstance(idxs, np.ndarray):
                    idxs = list(idxs)
                
                # post-process
                if dataset == 'bbn' and post_process:
                    pre_label = [id2type[idx] for idx in idxs]
                    if '/ORGANIZATION' in pre_label and '/PERSON' in pre_label:
                        idxs.remove(type2id['/PERSON'])
                        pre_label.remove('/PERSON')
                        
                    if '/LOCATION' in pre_label and '/GPE' in pre_label:
                        idxs.remove(type2id['/LOCATION'])
                        pre_label.remove('/LOCATION')
                        
                    if '/FACILITY' in pre_label:
                        idxs[pre_label.index('/FACILITY')] = type2id['/FAC']   
                        pre_label[pre_label.index('/FACILITY')] = '/FAC'
                        
                    idxs = list(set(idxs))
            
                gp_tups.append((batch_label_src[i], idxs))
          
          
        gp_fine_grained_label = []  
        gp_coarse_grained_label = []  
        
        gp_participate_fine_grained_contrastive = [] 
        gp_no_participate_fine_grained_contrastive = []  
        gp_participate_coarse_grained_contrastive = []  
        gp_no_participate_coarse_grained_contrastive = []  
        
        for labels,preds in gp_tups:
            pre_label = [id2type[idx] for idx in preds]
            glod_label = [id2type[idx] for idx in labels]
             
            # 
            pre_fine_grained_label = [label for label in pre_label if label in fine_grained_label]
            glod_fine_grained_label = [label for label in glod_label if label in fine_grained_label]
            if glod_fine_grained_label:
                gp_fine_grained_label.append((glod_fine_grained_label,pre_fine_grained_label))
            # 
            pre_coarse_grained_label = [label for label in pre_label if label in coarse_grained_label]
            glod_coarse_grained_label = [label for label in glod_label if label in coarse_grained_label]
            if glod_fine_grained_label:
                gp_coarse_grained_label.append((glod_coarse_grained_label,pre_coarse_grained_label))
            
            # 
            pre_participate_fine_grained_contrastive_label = \
                    [label for label in pre_label if label in participate_fine_grained_contrastive_label]
            glod_participate_fine_grained_contrastive_label = \
                    [label for label in glod_label if label in participate_fine_grained_contrastive_label]
            if glod_participate_fine_grained_contrastive_label:
                gp_participate_fine_grained_contrastive.append(
                        (
                            glod_participate_fine_grained_contrastive_label,
                            pre_participate_fine_grained_contrastive_label
                        )
                    )
                
            # 
            pre_no_participate_fine_grained_contrastive_label = \
                    [label for label in pre_label if label in no_participate_fine_grained_contrastive_label]
            glod_no_participate_fine_grained_contrastive_label = \
                    [label for label in glod_label if label in no_participate_fine_grained_contrastive_label]
            if glod_no_participate_fine_grained_contrastive_label:
                gp_no_participate_fine_grained_contrastive.append(
                        (
                            glod_no_participate_fine_grained_contrastive_label,
                            pre_no_participate_fine_grained_contrastive_label
                        )
                    )
                
            # 
            pre_participate_coarse_grained_contrastive_label = \
                    [label for label in pre_label if label in participate_coarse_grained_contrastive_label]
            glod_participate_coarse_grained_contrastive_label = \
                    [label for label in glod_label if label in participate_coarse_grained_contrastive_label]
            if glod_participate_coarse_grained_contrastive_label:
                gp_participate_coarse_grained_contrastive.append(
                        (
                            glod_participate_coarse_grained_contrastive_label,
                            pre_participate_coarse_grained_contrastive_label
                        )
                    )
                
            # 
            pre_no_participate_coarse_grained_contrastive_label = \
                    [label for label in pre_label if label in no_participate_coarse_grained_contrastive_label]
            glod_no_participate_coarse_grained_contrastive_label = \
                    [label for label in glod_label if label in no_participate_coarse_grained_contrastive_label]
            if glod_no_participate_coarse_grained_contrastive_label:
                gp_no_participate_coarse_grained_contrastive.append(
                        (
                            glod_no_participate_coarse_grained_contrastive_label,
                            pre_no_participate_coarse_grained_contrastive_label
                        )
                    )
                
          
            
        metrics = dict()
        
        strict_precision, strict_recall, strict_f1 = self.metrics.calc_strict_f1(gp_tups)
        macro_precision, macro_recall, macro_f1 = self.metrics.calc_loose_macro_f1(gp_tups)
        micro_precision, micro_recall, micro_f1 = self.metrics.calc_loose_micro_f1(gp_tups)
        metrics.update({
            'total_metrics':{
                                        'strict_precision': round(strict_precision,6),
                                        'strict_recall': round(strict_recall,6),
                                        'strict_f1': round(strict_f1,6),
                                        'macro_precision': round(macro_precision,6),
                                        'macro_recall': round(macro_recall,6),
                                        'macro_f1': round(macro_f1,6),
                                        'micro_precision': round(micro_precision,6),
                                        'micro_recall': round(micro_recall,6),
                                        'micro_f1': round(micro_f1,6),
                                    
                                    }
            
            })
        
        strict_precision, strict_recall, strict_f1 = self.metrics.calc_strict_f1(gp_fine_grained_label)
        macro_precision, macro_recall, macro_f1 = self.metrics.calc_loose_macro_f1(gp_fine_grained_label)
        micro_precision, micro_recall, micro_f1 = self.metrics.calc_loose_micro_f1(gp_fine_grained_label)
        metrics.update({
            'fine_grained_contrastive_label_metrics':{
                                        'strict_precision': round(strict_precision,6),
                                        'strict_recall': round(strict_recall,6),
                                        'strict_f1': round(strict_f1,6),
                                        'macro_precision': round(macro_precision,6),
                                        'macro_recall': round(macro_recall,6),
                                        'macro_f1': round(macro_f1,6),
                                        'micro_precision': round(micro_precision,6),
                                        'micro_recall': round(micro_recall,6),
                                        'micro_f1': round(micro_f1,6),
                                    
                                    }
            
            })
        
        strict_precision, strict_recall, strict_f1 = self.metrics.calc_strict_f1(gp_coarse_grained_label)
        macro_precision, macro_recall, macro_f1 = self.metrics.calc_loose_macro_f1(gp_coarse_grained_label)
        micro_precision, micro_recall, micro_f1 = self.metrics.calc_loose_micro_f1(gp_coarse_grained_label)
        
        metrics.update({
            'coarse_grained_contrastive_label_metrics':{
                                        'strict_precision': round(strict_precision,6),
                                        'strict_recall': round(strict_recall,6),
                                        'strict_f1': round(strict_f1,6),
                                        'macro_precision': round(macro_precision,6),
                                        'macro_recall': round(macro_recall,6),
                                        'macro_f1': round(macro_f1,6),
                                        'micro_precision': round(micro_precision,6),
                                        'micro_recall': round(micro_recall,6),
                                        'micro_f1': round(micro_f1,6),
                                    
                                    }
            
            })
        
        strict_precision, strict_recall, strict_f1 = self.metrics.calc_strict_f1(gp_participate_fine_grained_contrastive)
        macro_precision, macro_recall, macro_f1 = self.metrics.calc_loose_macro_f1(gp_participate_fine_grained_contrastive)
        micro_precision, micro_recall, micro_f1 = self.metrics.calc_loose_micro_f1(gp_participate_fine_grained_contrastive)
        
        metrics.update({
            'participate_fine_grained_contrastive_label_metrics':{
                                        'strict_precision': round(strict_precision,6),
                                        'strict_recall': round(strict_recall,6),
                                        'strict_f1': round(strict_f1,6),
                                        'macro_precision': round(macro_precision,6),
                                        'macro_recall': round(macro_recall,6),
                                        'macro_f1': round(macro_f1,6),
                                        'micro_precision': round(micro_precision,6),
                                        'micro_recall': round(micro_recall,6),
                                        'micro_f1': round(micro_f1,6),
                                    
                                    }
            
            })
        
        strict_precision, strict_recall, strict_f1 = self.metrics.calc_strict_f1(gp_no_participate_fine_grained_contrastive)
        macro_precision, macro_recall, macro_f1 = self.metrics.calc_loose_macro_f1(gp_no_participate_fine_grained_contrastive)
        micro_precision, micro_recall, micro_f1 = self.metrics.calc_loose_micro_f1(gp_no_participate_fine_grained_contrastive)
        
        metrics.update({
            'no_participate_fine_grained_contrastive_label_metrics':{
                                        'strict_precision': round(strict_precision,6),
                                        'strict_recall': round(strict_recall,6),
                                        'strict_f1': round(strict_f1,6),
                                        'macro_precision': round(macro_precision,6),
                                        'macro_recall': round(macro_recall,6),
                                        'macro_f1': round(macro_f1,6),
                                        'micro_precision': round(micro_precision,6),
                                        'micro_recall': round(micro_recall,6),
                                        'micro_f1': round(micro_f1,6),
                                    
                                    }
            
            })
        
        strict_precision, strict_recall, strict_f1 = self.metrics.calc_strict_f1(gp_participate_coarse_grained_contrastive)
        macro_precision, macro_recall, macro_f1 = self.metrics.calc_loose_macro_f1(gp_participate_coarse_grained_contrastive)
        micro_precision, micro_recall, micro_f1 = self.metrics.calc_loose_micro_f1(gp_participate_coarse_grained_contrastive)
        
        metrics.update({
            'participate_coarse_grained_contrastive_label_metrics':{
                                        'strict_precision': round(strict_precision,6),
                                        'strict_recall': round(strict_recall,6),
                                        'strict_f1': round(strict_f1,6),
                                        'macro_precision': round(macro_precision,6),
                                        'macro_recall': round(macro_recall,6),
                                        'macro_f1': round(macro_f1,6),
                                        'micro_precision': round(micro_precision,6),
                                        'micro_recall': round(micro_recall,6),
                                        'micro_f1': round(micro_f1,6),
                                    
                                    }
            
            })
        
        strict_precision, strict_recall, strict_f1 = self.metrics.calc_strict_f1(gp_no_participate_coarse_grained_contrastive)
        macro_precision, macro_recall, macro_f1 = self.metrics.calc_loose_macro_f1(gp_no_participate_coarse_grained_contrastive)
        micro_precision, micro_recall, micro_f1 = self.metrics.calc_loose_micro_f1(gp_no_participate_coarse_grained_contrastive)
            
        metrics.update({
            'no_participate_coarse_grained_contrastive_label_metrics':{
                                        'strict_precision': round(strict_precision,6),
                                        'strict_recall': round(strict_recall,6),
                                        'strict_f1': round(strict_f1,6),
                                        'macro_precision': round(macro_precision,6),
                                        'macro_recall': round(macro_recall,6),
                                        'macro_f1': round(macro_f1,6),
                                        'micro_precision': round(micro_precision,6),
                                        'micro_recall': round(micro_recall,6),
                                        'micro_f1': round(micro_f1,6),
                                    
                                    }
            
            })
        
       
        print(json.dumps(metrics,ensure_ascii=False,indent=4))
            
        
        model.train()     
        return metrics
                  
def train():
    
    
    step = 0
    best_val_strict_f1 = 0
    best_val_macro_f1 = 0
    best_val_micro_f1 = 0
    
    best_test_strict_f1 = 0
    best_test_macro_f1 = 0
    best_test_micro_f1 = 0
    lr_reduced = False
    


    for batch in train_batch_iter:
        step += 1
        model.train()
        
        logits, entity_typing_loss = model(batch)
        loss = entity_typing_loss.get('loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        print(loss)
        
        if step % eval_interval == 0:
            val_metrics = eval_fnc.evaluate(model,  dev_batch_iter)
            val_total_metrics = val_metrics['total_metrics']
            val_strict_precision, val_strict_recall, val_strict_f1, \
                val_macro_precision, val_macro_recall, val_macro_f1,  \
                val_micro_precision, val_micro_recall, val_micro_f1 =  \
                val_total_metrics['strict_precision'],val_total_metrics['strict_recall'],val_total_metrics['strict_f1'],\
                val_total_metrics['macro_precision'],val_total_metrics['macro_recall'],val_total_metrics['macro_f1'],\
                val_total_metrics['micro_precision'],val_total_metrics['micro_recall'],val_total_metrics['micro_f1']

            test_metrics = eval_fnc.evaluate(model, test_batch_iter)
            test_total_metrics = test_metrics['total_metrics']    
            test_strict_precision, test_strict_recall, test_strict_f1, \
                test_macro_precision, test_macro_recall, test_macro_f1,  \
                test_micro_precision, test_micro_recall, test_micro_f1 =  \
                test_total_metrics['strict_precision'],test_total_metrics['strict_recall'],test_total_metrics['strict_f1'],\
                test_total_metrics['macro_precision'],test_total_metrics['macro_recall'],test_total_metrics['macro_f1'],\
                test_total_metrics['micro_precision'],test_total_metrics['micro_recall'],test_total_metrics['micro_f1']
                
        

            if selection_model_criteria == 'dev':
                if evaluation_metric == 'strict_f1':
                    if best_val_strict_f1 < val_strict_f1:
                        best_val_strict_f1 = val_strict_f1
                        best_test_strict_f1 = test_strict_f1
                       
                        
                        fine_grained_test_metrics = eval_fnc.evaluate(model, fine_grained_test_batch_iter)
                        no_fine_grained_test_metrics = eval_fnc.evaluate(model, no_fine_grained_test_batch_iter)
                       
                        if save_model:
                            torch.save(model.state_dict(),f'./save_model/{model_name}')
                elif evaluation_metric == 'macro_f1':
                    if best_val_macro_f1 < val_macro_f1:
                        best_val_macro_f1 = val_macro_f1
                        best_test_macro_f1 = test_macro_f1
                      
                        
                        fine_grained_test_metrics = eval_fnc.evaluate(model, fine_grained_test_batch_iter)
                        no_fine_grained_test_metrics = eval_fnc.evaluate(model, no_fine_grained_test_batch_iter)
                       
                        if save_model:
                            torch.save(model.state_dict(),f'./save_model/{model_name}')
                elif evaluation_metric == 'micro_f1':
                    if best_val_micro_f1 < val_micro_f1:
                        best_val_micro_f1 = val_micro_f1
                        best_test_micro_f1 = test_micro_f1
                       
                        
                        fine_grained_test_metrics = eval_fnc.evaluate(model, fine_grained_test_batch_iter)
                        no_fine_grained_test_metrics = eval_fnc.evaluate(model, no_fine_grained_test_batch_iter)
                       
                        if save_model:
                            torch.save(model.state_dict(),f'./save_model/{model_name}')
            
            
            elif selection_model_criteria == 'test':
                if evaluation_metric == 'strict_f1':
                    if best_val_strict_f1 < val_strict_f1:
                        best_val_strict_f1 = val_strict_f1
                    if best_test_strict_f1 < test_strict_f1:
                        best_test_strict_f1 = test_strict_f1
                        #
                        fine_grained_test_metrics = eval_fnc.evaluate(model, fine_grained_test_batch_iter)
                        no_fine_grained_test_metrics = eval_fnc.evaluate(model, no_fine_grained_test_batch_iter)
                        
                        if save_model:
                            torch.save(model.state_dict(),f'./save_model/{model_name}')
                elif evaluation_metric == 'macro_f1':
                    if best_val_macro_f1 < val_macro_f1:
                        best_val_macro_f1 = val_macro_f1
                    if best_test_macro_f1 < test_macro_f1:
                        best_test_macro_f1 = test_macro_f1
                        #
                        fine_grained_test_metrics = eval_fnc.evaluate(model, fine_grained_test_batch_iter)
                        no_fine_grained_test_metrics = eval_fnc.evaluate(model, no_fine_grained_test_batch_iter)
                        if save_model:
                            torch.save(model.state_dict(),f'./save_model/{model_name}')
                elif evaluation_metric == 'micro_f1':
                    if best_val_micro_f1 < val_micro_f1:
                        best_val_micro_f1 = val_micro_f1
                    if best_test_micro_f1 < test_micro_f1:
                        best_test_micro_f1 = test_micro_f1
                        # 
                        fine_grained_test_metrics = eval_fnc.evaluate(model, fine_grained_test_batch_iter)
                        no_fine_grained_test_metrics = eval_fnc.evaluate(model, no_fine_grained_test_batch_iter)
                     
                        if save_model:
                            torch.save(model.state_dict(),f'./save_model/{model_name}')
                    

            
            print('seed:',seed)
            print(
                    
                    '## step={} loss={:.6f} \n'
                    '   val_strict_precision = {:.6f}   val_strict_recall = {:.6f}  val_strict_f1 = {:.6f}  best_val_strict_f1 = {:.6f} \n'
                    '   val_macro_precision = {:.6f}   val_macro_recall = {:.6f}  val_macro_f1 = {:.6f}  best_val_macro_f1 = {:.6f} \n'
                    '   val_micro_precision = {:.6f}   val_micro_recall = {:.6f}  val_micro_f1 = {:.6f}  best_val_micro_f1 = {:.6f} \n'
                    '   test_strict_precision = {:.6f}  test_strict_recall = {:.6f} test_strict_f1 = {:.6f} best_test_strict_f1 = {:.6f} \n'
                    '   test_macro_precision = {:.6f}  test_macro_recall = {:.6f} test_macro_f1 = {:.6f} best_test_macro_f1 = {:.6f} \n'
                    '   test_micro_precision = {:.6f}  test_micro_recall = {:.6f} test_micro_f1 = {:.6f} best_test_micro_f1 = {:.6f}'
                    .format(
                        step, loss.item(),
                        val_strict_precision, val_strict_recall, val_strict_f1, best_val_strict_f1,
                        val_macro_precision, val_macro_recall, val_macro_f1, best_val_macro_f1,
                        val_micro_precision, val_micro_recall, val_micro_f1, best_val_micro_f1,
                        test_strict_precision, test_strict_recall, test_strict_f1, best_test_strict_f1,
                        test_macro_precision, test_macro_recall, test_macro_f1, best_test_macro_f1,
                        test_micro_precision, test_micro_recall, test_micro_f1, best_test_micro_f1
    
                    ))
            

            if dataset == 'ontonote':
                if test_macro_f1 > 0.77 and lr_reduced is False:
                    lr_scheduler.step()
                    lr_reduced = True
                    print('lr reduced')
            elif dataset == 'figer':
                if test_macro_f1 > 0.84 and lr_reduced is False:
                    lr_scheduler.step()
                    lr_reduced = True
                    print('lr reduced')
                    

eval_interval = 50
batch_size = 10
eval_batch_size = 192
lr_decay = 0.5
w_decay = 0.01
temperature = 0.1 
post_process = False

parser = argparse.ArgumentParser("参数")
parser.add_argument('-lr', '--learing_rate', type=float,default=1e-5)
parser.add_argument('-ds', '--dataset', default='ontonote')
parser.add_argument('-rs', '--rand_seed', type=int,default=83029)
parser.add_argument('-mb', '--model_backbone', default='bert-large-cased')
parser.add_argument('-ci', '--cuda_id', default='1')
parser.add_argument('-epoch', '--epochs',type=int,default=20)
parser.add_argument('-eval_metric', '--evaluation_metric', default='macro_f1') 
parser.add_argument('-eet', '--entity_embedding_type', default='ENTITY')  
parser.add_argument('-fgt', '--fine_grained_type', default='same_coarse_grained_calculate')
parser.add_argument('-cgt', '--coarse_grained_type', default='all')
parser.add_argument('-smc', '--selection_model_criteria', default='test')
parser.add_argument('-use_mg', '--use_more_granular', default=False)
parser.add_argument('-task_wt', '--task_weight_type', default='custom') #  custom
parser.add_argument('-task_w', '--task_weight', default='10,1,1')


parser.add_argument('-task', '--task', default='tag,cgc,fgc')

args = parser.parse_args()

learing_rate = args.learing_rate
dataset = args.dataset
rand_seed = args.rand_seed
model_backbone = args.model_backbone
cuda_id = args.cuda_id
epochs = args.epochs
evaluation_metric = args.evaluation_metric
fine_grained_type = args.fine_grained_type
entity_embedding_type = args.entity_embedding_type
coarse_grained_type = args.coarse_grained_type
selection_model_criteria = args.selection_model_criteria
use_more_granular = args.use_more_granular
task = args.task
task = task.split(',')

task_weight_type = args.task_weight_type
task_weight = args.task_weight

task_weight = [float(task_w) for task_w in task_weight.split(',')]

if dataset == 'bbn':
    args.epochs = 10
elif dataset == 'figer':
    args.epochs = 1
elif dataset == 'ontonote':
    args.epochs = 2
  
if dataset != 'ontonote':
    args.use_more_granular = False

# gpu
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# PATH
train_path,dev_path,test_path, type_vocab_path = data_path(dataset)
# seed
seed = set_seed(rand_seed)
save_model = True


train_data = load_data(train_path)
dev_data = load_data(dev_path)
test_data = load_data(test_path)

id2type, type2id = load_vocab_file(type_vocab_path)
num_class = len(id2type)


model_name = f'{dataset}-{model_backbone}-{evaluation_metric}-{task}-{learing_rate}-{rand_seed}.pth'
model = EntityTypingModel(num_class,model_path=model_backbone,device=device,
                          task=task,task_weight=task_weight,
                          task_weight_type=task_weight_type,
                          entity_embedding_type=entity_embedding_type)

model.to(device)  
tokenizer = model.tokenizer

# 
train_batch_iter = DataBatchLoader(
                                       train_data, 
                                       batch_size, 
                                       epochs,
                                       dataset = dataset,
                                       type2id=type2id,
                                       tokenizer=tokenizer,
                                       use_more_granular=use_more_granular,
                                       fine_grained_type=fine_grained_type,
                                       coarse_grained_type=coarse_grained_type,
                                       shuffle=True
                                   )
dev_batch_iter = DataBatchLoader(
                                    dev_data, 
                                    eval_batch_size, 
                                    1, 
                                    dataset = dataset,
                                    type2id=type2id,
                                    tokenizer=tokenizer,
                                    use_more_granular=use_more_granular,
                                    fine_grained_type=fine_grained_type,
                                    coarse_grained_type=coarse_grained_type
                                )
test_batch_iter = DataBatchLoader(
                                    test_data, 
                                    eval_batch_size, 
                                    1, 
                                    dataset = dataset,
                                    type2id=type2id,
                                    tokenizer=tokenizer,
                                    use_more_granular=use_more_granular,
                                    fine_grained_type=fine_grained_type,
                                    coarse_grained_type=coarse_grained_type
                                )
#
contrastive_label = get_contrastive_label()

fine_grained_test_data = [data for data in test_data 
                          if any(d in contrastive_label.get('fine_grained_label') for d in data[2]) ]
no_fine_grained_test_data = [item for item in test_data if item not in fine_grained_test_data]

fine_grained_test_batch_iter = DataBatchLoader(
                                                fine_grained_test_data, 
                                                eval_batch_size, 
                                                1, 
                                                dataset = dataset,
                                                type2id=type2id,
                                                tokenizer=tokenizer,
                                                use_more_granular=use_more_granular,
                                                fine_grained_type=fine_grained_type,
                                                coarse_grained_type=coarse_grained_type
                                    )
no_fine_grained_test_batch_iter = DataBatchLoader(
                                    no_fine_grained_test_data, 
                                    eval_batch_size, 
                                    1, 
                                    dataset = dataset,
                                    type2id=type2id,
                                    tokenizer=tokenizer,
                                    use_more_granular=use_more_granular,
                                    fine_grained_type=fine_grained_type,
                                    coarse_grained_type=coarse_grained_type
                                    )


optimizer = configure_optimizers(list(model.named_parameters()), learning_rate=learing_rate, w_decay=w_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: lr_decay)

eval_fnc = Evaluate()
    
    
if __name__ == '__main__':

    train()

    
