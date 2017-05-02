# Very-Deep Pairwise Word Interaction Neural Networks for Modeling Textual Similarity

This repo contains the Torch implementation of the very-deep pairwise word interaction neural network for modeling textual similarity, as described in the following paper:

+ Hua He and Jimmy Lin. [Pairwise Word Interaction Modeling with Deep Neural Networks for Semantic Similarity Measurement.](http://www.aclweb.org/anthology/N16-1108) *Proceedings of the 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL/HLT 2016)*, pages 937-948.


This model does not require external resources such as WordNet or parsers, does not use sparse features, and achieves good accuracy on standard public datasets.

Installation and Dependencies
------------

- Please install Torch deep learning library. We recommend this local installation which includes all required packages our tool needs, simply follow the instructions here:
https://github.com/torch/distro

- Currently our tool only runs on CPUs, therefore it is recommended to use INTEL MKL library (or at least OpenBLAS lib) so Torch can run much faster on CPUs. 

- Our tool then requires Glove embeddings by Stanford. Please run fetch_and_preprocess.sh for downloading and preprocessing this data set (around 3 GBs).


Running
------------

- Command to run (training, tuning and testing all included): 
- ``th trainSIC.lua``

The tool will output pearson scores and also write the predicted similarity scores given each pair of sentences from test data into predictions directory.



