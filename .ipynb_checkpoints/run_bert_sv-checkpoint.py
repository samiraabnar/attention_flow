import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from attention_graph_util import *
import seaborn as sns
import itertools 
import matplotlib as mpl
import networkx as nx
import os
from util import constants

from absl import app
from absl import flags
import pandas as pd

from util.models import MODELS
from util.tasks import TASKS
from attention_graph_util import *
from util.config_util import get_task_params
from notebooks.notebook_utils import *
from util import inflect

from tqdm import tqdm
from scipy.stats import spearmanr
import math


rc={'font.size': 10, 'axes.labelsize': 10, 'legend.fontsize': 10.0, 
    'axes.titlesize': 32, 'xtick.labelsize': 20, 'ytick.labelsize': 16}
plt.rcParams.update(**rc)
mpl.rcParams['axes.linewidth'] = .5 #set the value globally

import torch
from transformers import *
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from transformers import DistilBertTokenizer, DistilBertModel




def offset_convertor(encoded_input_task, task_offset, task_encoder, tokenizer):
    string_part1 = task_encoder.decode(encoded_input_task[:task_offset])
    tokens_part1 = tokenizer.tokenize(string_part1)
    
    return len(tokens_part1)


def get_raw_att_relevance(full_att_mat, input_tokens, layer=-1, output_index=0):
    raw_rel = full_att_mat[layer].sum(axis=0)[output_index]/full_att_mat[layer].sum(axis=0)[output_index].sum()
    
    return raw_rel


def get_joint_relevance(full_att_mat, input_tokens, layer=-1, output_index=0):
    att_sum_heads =  full_att_mat.sum(axis=1) / full_att_mat.shape[1]
    joint_attentions = compute_joint_attention(att_sum_heads, add_residual=True)
    relevance_attentions = joint_attentions[layer][output_index]
    return relevance_attentions


def get_flow_relevance(full_att_mat, input_tokens, layer, output_index):
    
    input_tokens = input_tokens
    res_att_mat = full_att_mat.sum(axis=1)/full_att_mat.shape[1]
    res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None,...]
    res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]

    res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=input_tokens)
    
    A = res_adj_mat
    res_G=nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(res_G, {(i,j): A[i,j]}, 'capacity')


    output_nodes = ['L'+str(layer+1)+'_'+str(output_index)]
    input_nodes = []
    for key in res_labels_to_index:
        if res_labels_to_index[key] < full_att_mat.shape[-1]:
            input_nodes.append(key)
    
    flow_values = compute_node_flow(res_G, res_labels_to_index, input_nodes=input_nodes, output_nodes=output_nodes, length=full_att_mat.shape[-1])
    
    n_layers = full_att_mat.shape[0]
    length = full_att_mat.shape[-1]
    final_layer_attention = flow_values[(layer+1)*length:, layer*length:(layer+1)*length]
    relevance_attention_flow = final_layer_attention[output_index]

    return relevance_attention_flow


task_name = 'word_sv_agreement_lm'
task_params = get_task_params(batch_size=1)
task = TASKS[task_name](task_params, data_dir='../InDist/data')
cl_token = task.sentence_encoder().encode(constants.bos)
task_tokenizer = task.sentence_encoder()._tokenizer

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased',
                                        output_hidden_states=True,
                                        output_attentions=True)



all_examples_x = []
all_examples_vp = []
all_examples_y = []

all_examples_attentions = []
all_examples_blankout_relevance = []
all_examples_grads = []
all_examples_inputgrads = []
n_batches = 1000

all_examples_accuracies = []

infl_eng = inflect.engine()
verb_infl, noun_infl = gen_inflect_from_vocab(infl_eng, '../InDist/notebooks/wiki.vocab')

test_data = task.databuilder.as_dataset(split='validation', batch_size=1)
for examples in tqdm(test_data):
    sentence = task.sentence_encoder().decode(examples['sentence'][0])
    
    verb_position = examples['verb_position'][0].numpy()+1  #+1 because of adding cls.
    verb_position = offset_convertor(examples['sentence'][0], verb_position, task.sentence_encoder(), tokenizer)
    
    sentence = ['cls']+tokenizer.tokenize(sentence)+['sep']
    
    all_examples_vp.append(verb_position)
    sentence[verb_position] = tokenizer.mask_token
    tf_input_ids = tokenizer.encode(sentence)
    input_ids = torch.tensor([tf_input_ids])
    

    
    s_shape = input_ids.shape
    batch_size, length = s_shape[0], s_shape[1]
    actual_verb = examples['verb'][0].numpy().decode("utf-8")
    inflected_verb = verb_infl[actual_verb] 


    actual_verb_index = tokenizer.encode(tokenizer.tokenize(actual_verb))[1]
    inflected_verb_index = tokenizer.encode(tokenizer.tokenize(inflected_verb))[1]

    all_examples_x.append(input_ids)
    embeded_inputs = torch.autograd.Variable(model.distilbert.embeddings(input_ids), requires_grad=True)
    predictions = model(inputs_embeds=embeded_inputs)
    logits = predictions[0][0]

    
        
    probs = torch.nn.Softmax(dim=-1)(logits)
    actual_verb_score = probs[verb_position][actual_verb_index]
    inflected_verb_score = probs[verb_position][inflected_verb_index]
    
    main_diff_score = actual_verb_score - inflected_verb_score
    
    all_examples_accuracies.append(main_diff_score > 0)
    
    main_diff_score.backward()
    grads = embeded_inputs.grad
    grad_scores = abs(np.sum(grads.detach().numpy(), axis=-1))
    input_grad_scores = abs(np.sum((grads * embeded_inputs).detach().numpy(), axis=-1))
    all_examples_grads.append(grad_scores)
    all_examples_inputgrads.append(input_grad_scores)
    
    hidden_states, attentions = predictions[-2:]
    _attentions = [att.detach().numpy() for att in attentions]
    attentions_mat = np.asarray(_attentions)[:,0]

    all_examples_attentions.append(attentions_mat)
    
    # Repeating examples and replacing one token at a time with unk
    batch_size = 1
    max_len = input_ids.shape[1]
    
    # Repeat each example 'max_len' times
    x = input_ids
    extended_x = np.reshape(np.tile(x[:,None,...], (1, max_len, 1)),(-1,x.shape[-1]))

    # Create unk sequences and unk mask
    unktoken = tokenizer.encode([tokenizer.mask_token])[1]
    unks = unktoken * np.eye(max_len)
    unks =  np.tile(unks, (batch_size, 1))
    
    unk_mask =  (unktoken - unks)/unktoken
  
    # Replace one token in each repeatition with unk
    extended_x = extended_x * unk_mask + unks
    
    # Get the new output
    extended_predictions = model(torch.tensor(extended_x, dtype=torch.int64))
    extended_logits = extended_predictions[0]
    extended_probs = torch.nn.Softmax(dim=-1)(extended_logits)
    
    extended_correct_probs = extended_probs[:,verb_position,actual_verb_index]
    extended_wrong_probs =  extended_probs[:,verb_position,inflected_verb_index]
    extended_diff_scores = extended_correct_probs - extended_wrong_probs
    
    # Save the difference in the probability predicted for the correct class
    diffs = abs(main_diff_score - extended_diff_scores)

    all_examples_blankout_relevance.append(diffs.detach())
    n_batches -= 1
    if n_batches <= 0:
        break


print("compute raw relevance scores ...")
all_examples_raw_relevance = {}
for l in np.arange(0,6):
    all_examples_raw_relevance[l] = []
    for i in tqdm(np.arange(len(all_examples_x))):
        tokens = tokenizer.decode(all_examples_x[i][0].numpy())
        vp = all_examples_vp[i]
        length = len(tokens)
        attention_relevance = get_raw_att_relevance(all_examples_attentions[i], tokens, layer=l, output_index=vp)
        all_examples_raw_relevance[l].append(np.asarray(attention_relevance))

print("compute joint relevance scores ...")
all_examples_joint_relevance = {}
for l in np.arange(0,6):
    all_examples_joint_relevance[l] = []
    for i in tqdm(np.arange(len(all_examples_x))):
        tokens = tokenizer.decode(all_examples_x[i][0].numpy())
        vp = all_examples_vp[i]
        length = len(tokens)
        attention_relevance = get_joint_relevance(all_examples_attentions[i], tokens, layer=l, output_index=vp)
        all_examples_joint_relevance[l].append(np.asarray(attention_relevance))
    
print("compute flow relevance scores ...")
all_examples_flow_relevance = {}
for l in np.arange(0,6):
    all_examples_flow_relevance[l] = []
    for i in tqdm(np.arange(len(all_examples_x))):
        tokens = tokenizer.decode(all_examples_x[i][0].numpy())
        vp = all_examples_vp[i]
        length = len(tokens)
        attention_relevance = get_flow_relevance(all_examples_attentions[i], tokens, layer=l, output_index=vp)
        all_examples_flow_relevance[l].append(np.asarray(attention_relevance))
        
        
raw_sps_blank = []
raw_sps_grad = []
raw_sps_inputgrad = []

joint_sps_blank = []
joint_sps_grad = []
joint_sps_inputgrad = []

flow_sps_blank = []
flow_sps_grad = []
flow_sps_inputgrad = []


for l in np.arange(0,6):
    print("###############Layer ",l, "#############")
    print('raw blankout')
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_raw_relevance[l][i],all_examples_blankout_relevance[i].numpy())
        if not math.isnan(sp[0]):
            raw_sps_blank.append(sp[0])
        else:
            raw_sps_blank.append(0)
        
    print(np.mean(raw_sps_blank), np.std(raw_sps_blank))
    
    
    print('raw inputgrad')
    print(all_examples_raw_relevance[l][0].shape, all_examples_inputgrads[0][0].shape)
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_raw_relevance[l][i],all_examples_inputgrads[i][0])
        if not math.isnan(sp[0]):
            raw_sps_inputgrad.append(sp[0])
        else:
            raw_sps_inputgrad.append(0)
        
    print(np.mean(raw_sps_inputgrad), np.std(raw_sps_inputgrad))
    
    print('raw grad')
    print(all_examples_raw_relevance[l][0].shape, all_examples_grads[0][0].shape)
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_raw_relevance[l][i],all_examples_grads[i][0])
        if not math.isnan(sp[0]):
            raw_sps_grad.append(sp[0])
        else:
            raw_sps_grad.append(0)
        
    print(np.mean(raw_sps_grad), np.std(raw_sps_grad))
    
    print('joint blankout')
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_joint_relevance[l][i],all_examples_blankout_relevance[i].numpy())
        if not math.isnan(sp[0]):
            joint_sps_blank.append(sp[0])
        else:
            joint_sps_blank.append(0)
        
    print(np.mean(joint_sps_blank), np.std(joint_sps_blank))
    
    print('joint grad')
    print(all_examples_joint_relevance[l][0].shape, all_examples_grads[0][0].shape)
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_joint_relevance[l][i],all_examples_grads[i][0])
        if not math.isnan(sp[0]):
            joint_sps_grad.append(sp[0])
        else:
            joint_sps_grad.append(0)
        
    print(np.mean(joint_sps_grad), np.std(joint_sps_grad))
    
    print('joint inputgrad')
    print(all_examples_joint_relevance[l][0].shape, all_examples_inputgrads[0][0].shape)
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_joint_relevance[l][i],all_examples_inputgrads[i][0])
        if not math.isnan(sp[0]):
            joint_sps_inputgrad.append(sp[0])
        else:
            joint_sps_inputgrad.append(0)
        
    print(np.mean(joint_sps_inputgrad), np.std(joint_sps_inputgrad))
    
    print('flow')
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_flow_relevance[l][i],all_examples_blankout_relevance[i].numpy())
        
        if not math.isnan(sp[0]):
            flow_sps_blank.append(sp[0])
        else:
            flow_sps_blank.append(0)
        
    print(np.mean(flow_sps_blank), np.std(flow_sps_blank))
  
    print('flow grad')
    print(all_examples_joint_relevance[l][0].shape, all_examples_grads[0][0].shape)
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_flow_relevance[l][i],all_examples_grads[i][0])
        if not math.isnan(sp[0]):
            flow_sps_grad.append(sp[0])
        else:
            flow_sps_grad.append(0)
        
    print(np.mean(flow_sps_grad), np.std(flow_sps_grad))
    
    print('flow inputgrad')
    print(all_examples_joint_relevance[l][0].shape, all_examples_inputgrads[0][0].shape)
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_flow_relevance[l][i],all_examples_inputgrads[i][0])
        if not math.isnan(sp[0]):
            flow_sps_inputgrad.append(sp[0])
        else:
            flow_sps_inputgrad.append(0)
        
    print(np.mean(flow_sps_inputgrad), np.std(flow_sps_inputgrad))
    
np.save('all_examples_flow_relevance', all_examples_flow_relevance)
np.save('all_examples_joint_relevance', all_examples_joint_relevance)
np.save('all_examples_blankout_relevance', all_examples_blankout_relevance)
np.save('all_examples_grads', all_examples_grads)