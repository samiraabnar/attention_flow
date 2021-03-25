# Attention flow

This repository contain implementations of __Attention Rollout__ and __Attention Flow__ algorithms, which are post hoc methods to get more explanatory attention weights.

Attention Rollout  and Attention Flow recursively compute the token attentions in each layer of a given model given the embedding attentions as input. They differ in the assumptions they make about how attention weights in lower layers affect the flow of information to the higher layers and whether to compute the token attentions relative to each other or independently. 


* [Colab showing how to apply these methods on a pretrained BERT model of huggingface Transformer library](https://colab.research.google.com/drive/1nG_6T3mMu9aI7_k_sCpayusONELtJrAP?usp=sharing)


#### Here is the paper introducing these methods:
* [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928)


#### Related projects:
* [An implementation of Attention Rollout for Vision Transformers by Jacob Gildenblat](https://github.com/jacobgil/vit-explain) (and a nice blog post on [`Exploring Explanaibality for Vision Transformers`](https://jacobgil.github.io/deeplearning/vision-transformer-explainability)).
