# Attention flow

This repository contain implementations of __Attention Rollout__ and __Attention Flow__ algorithms, which are post hoc methods to get more explanatory attention weights.

Attention Rollout  and Attention Flow recursively compute the token attentions in each layer of a given model given the embedding attentions as input. They differ in the assumptions they make about how attention weights in lower layers affect the flow of information to the higher layers and whether to compute the token attentions relative to each other or independently. 


* [Notebook showing how to apply these methods on a pretrained BERT model of huggingface Transformer library](https://github.com/samiraabnar/attention_flow/blob/master/bert_example.ipynb)


#### Here is the paper introducing these methods:
* [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)
