import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import math
from tqdm import tqdm

# HuggingFace Transformer Library
from transformers import *

# Local imports
from attention_graph_util import *




tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

data = tfds.load('glue/sst2')

# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='sst-2')
valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, max_length=128, task='sst-2')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)
valid_dataset = valid_dataset.batch(64)
test_dataset = glue_convert_examples_to_features(data['test'], tokenizer, max_length=128, task='sst-2')
test_dataset = test_dataset.batch(1)





def spearmanr(x, y):
    """ `x`, `y` --> pd.Series"""
    x = pd.Series(x)
    y = pd.Series(y)
    assert x.shape == y.shape
    rx = x.rank(method='dense')
    ry = y.rank(method='dense')
    d = rx - ry
    dsq = np.sum(np.square(d))
    n = x.shape[0]
    coef = 1. - (6. * dsq) / (n * (n**2 - 1.))
    return [coef]

def get_raw_att_relevance(full_att_mat, input_tokens, layer=-1):
    att_sum_heads =  full_att_mat.sum(axis=1)/full_att_mat.shape[1]
    return att_sum_heads[layer].max(axis=0)
    

def compute_node_flow(G, labels_to_index, input_nodes, output_nodes,length):
    number_of_nodes = len(labels_to_index)
    flow_values=np.zeros((number_of_nodes,number_of_nodes))
    for key in output_nodes:
        if key not in input_nodes:
            current_layer = int(labels_to_index[key] / length)
            pre_layer = current_layer - 1
            u = labels_to_index[key]
            for inp_node_key in input_nodes:
                v = labels_to_index[inp_node_key]
                flow_value = nx.maximum_flow_value(G,u,v)
                flow_values[u][pre_layer*length+v ] = flow_value
            flow_values[u] /= flow_values[u].sum()
            
    return flow_values

def get_flow_relevance(full_att_mat, input_tokens, layer):
    
    res_att_mat = full_att_mat.sum(axis=1)/full_att_mat.shape[1]
    res_att_mat = res_att_mat + np.eye(res_att_mat.shape[1])[None,...]
    res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[...,None]

    res_adj_mat, res_labels_to_index = get_adjmat(mat=res_att_mat, input_tokens=input_tokens)
    
    A = res_adj_mat
    res_G=nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    for i in np.arange(A.shape[0]):
        for j in np.arange(A.shape[1]):
            nx.set_edge_attributes(res_G, {(i,j): A[i,j]}, 'capacity')

    output_nodes = []
    input_nodes = []
    for key in res_labels_to_index:
        if key.startswith('L'+str(layer+1)+'_'):
            output_nodes.append(key)
        if res_labels_to_index[key] < full_att_mat.shape[-1]:
            input_nodes.append(key)
    
    flow_values = compute_node_flow(res_G, res_labels_to_index, input_nodes, output_nodes, length=full_att_mat.shape[-1])
    
    n_layers = full_att_mat.shape[0]
    length = full_att_mat.shape[-1]
    final_layer_attention_raw = flow_values[(layer+1)*length: (layer+2)*length,layer*length:(layer+1)*length]
    relevance_attention_raw = final_layer_attention_raw.max(axis=0)

    return relevance_attention_raw
    
    
def get_joint_relevance(full_att_mat, input_tokens, layer):
    att_sum_heads =  full_att_mat.sum(axis=1)/full_att_mat.shape[1]
    joint_attentions = compute_joint_attention(att_sum_heads, add_residual=True)
    relevance_attentions = joint_attentions[layer].max(axis=0)
    return relevance_attentions


def get_nores_joint_relevance(full_att_mat, input_tokens, layer):
    att_sum_heads =  full_att_mat.sum(axis=1)/full_att_mat.shape[1]
    joint_attentions = compute_joint_attention(att_sum_heads, add_residual=False)
    relevance_attentions = joint_attentions[layer].max(axis=0)
    return relevance_attentions


model.config.output_attentions = True
model.config.output_hidden_states = True

model.distilbert.transformer.output_attentions = True
model.distilbert.transformer.output_hidden_states = True

for layer in model.distilbert.transformer.layer:
    layer.output_attentions = True    
    layer.output_hidden_states = True
    layer.attention.output_attentions = True
    layer.attention.output_hidden_states = True

# Read examples and save attention mats and input gradient scores.
all_examples_grads = []
all_examples_attentions = []
all_examples_x = []
n_examples = 100
for x,y in test_dataset:
    with tf.GradientTape() as tape:
        inputs_embeds = model.distilbert.embeddings(x['input_ids'])
        tape.watch(inputs_embeds)
        logits, hidden_states, attentions = model({'attention_mask':x['attention_mask'], 
                                                    'inputs_embeds':inputs_embeds,
                                                    'token_type_ids': x['token_type_ids']}, training=False
                                            )
        pindex = tf.argmax(logits, axis=-1)
        true_logits = logits[:,y[0]]
        print(true_logits)
    grads = tape.gradient(true_logits, inputs_embeds)[0]
    
    length = tf.reduce_sum(x['attention_mask'], axis=-1)[0]    
    all_examples_grads.append(tf.abs(tf.reduce_sum(grads, -1)[:length]))
    
    _attentions = [att.numpy() for att in attentions]
    attentions_mat = np.asarray(_attentions)[:,0]
        
    cropped_input = x['input_ids'][0, :length]
    all_examples_x.append(cropped_input)
    cropped_attention_mat = attentions_mat[:,:,:length,:length]
    all_examples_attentions.append(cropped_attention_mat)
        
    if n_examples == 0:
        break
    n_examples -= 1
    
    

print("compute raw relevance scores ...")
all_examples_raw_relevance = {}
for l in np.arange(0,6):
    all_examples_raw_relevance[l] = []
    for i in tqdm(np.arange(len(all_examples_x))):
        tokens = tokens = tokenizer.convert_ids_to_tokens(all_examples_x[i])
        length = len(tokens)
        attention_relevance = get_raw_att_relevance(all_examples_attentions[i], tokens, layer=l)
        all_examples_raw_relevance[l].append(np.asarray(attention_relevance))

print("compute joint relevance scores ...")
all_examples_joint_relevance = {}
for l in [0, 2, 4, 5]:
    all_examples_joint_relevance[l] = []
    for i in tqdm(np.arange(len(all_examples_x))):
        tokens = tokenizer.convert_ids_to_tokens(all_examples_x[i])
        length = len(tokens)
        attention_relevance = get_joint_relevance(all_examples_attentions[i], tokens, layer=l)
        all_examples_joint_relevance[l].append(np.asarray(attention_relevance))
    
        
        
for l in [0, 2, 4, 5]:
    print("###############Layer ",l, "#############")
    print('raw grad')
    print(all_examples_raw_relevance[l][0].shape, all_examples_grads[0].shape)
    raw_sps_grad = []
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_raw_relevance[l][i],all_examples_grads[i])
        raw_sps_grad.append(sp[0])
 
        
    print(np.mean(raw_sps_grad), np.std(raw_sps_grad))

    
    print('joint grad')
    print(all_examples_joint_relevance[l][0].shape, all_examples_grads[0].shape)
    joint_sps_grad = []
    for i in np.arange(len(all_examples_x)):
        sp = spearmanr(all_examples_joint_relevance[l][i],all_examples_grads[i])
        joint_sps_grad.append(sp[0])

        
    print(np.mean(joint_sps_grad), np.std(joint_sps_grad))

  

print("compute flow relevance scores ...")
all_examples_flow_relevance = {}
for l in [0, 2, 4, 5]:
    print("###############Layer ",l, "#############")
    flow_sps_grad = []
    all_examples_flow_relevance[l] = []
    for i in tqdm(np.arange(len(all_examples_x))):
        tokens = tokenizer.convert_ids_to_tokens(all_examples_x[i])
        length = len(tokens)            
        attention_relevance = get_flow_relevance(all_examples_attentions[i], 
                                                 tokens,layer=l)
        all_examples_flow_relevance[l].append(np.asarray(attention_relevance))
        sp = spearmanr(all_examples_flow_relevance[l][i],all_examples_grads[i])
        flow_sps_grad.append(sp[0])
        
    print(np.mean(flow_sps_grad), np.std(flow_sps_grad))
        
        
        