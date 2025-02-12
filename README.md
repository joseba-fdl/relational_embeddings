## Relational Embeddings for Language Independent Stance Detection
The large majority of the research performed on stance detection has been focused on developing more or less sophisticated text classification systems, even when many benchmarks are based on social network data such as Twitter. This paper aims to take on the stance detection task by placing the emphasis not so much on the text itself but on the interaction data available on social networks. More specifically, we propose a new method to leverage social information such as _friends_ and _retweets_ by generating Relational Embeddings, namely, dense vector representations of interaction pairs. Our experiments on seven publicly available datasets and four different languages (Basque, Catalan, Italian and Spanish) show that combining our relational embeddings with discriminative textual methods helps to substantially improve performance, obtaining state-of-the-art results for six out of seven evaluation settings, outperforming strong baselines based on Large Language Models, or other popular interaction-based approaches such as DeepWalk or node2vec.


### Data 
+ CIC: https://github.com/ixa-ehu/catalonia-independence-corpus
+ SardiStance: https://github.com/mirkolai/evalita-sardistance
+ VaxxStance: https://vaxxstance.github.io/

### Additional Data 
+ Relational information of CIC tweets' authors
+ VaxxStance ES RTs from user TLs

