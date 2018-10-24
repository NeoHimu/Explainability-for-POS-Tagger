
# Description

This code release contains an implementation of two relevance decomposition methods, Layer-wise Relevance Propagation (LRP) and Sensitivity Analysis (SA), for a bidirectional LSTM on Sequence to Sequence Labeling Task (POS tagger in this case). This code can be used for explaining any Sequence to Sequence Labeling task like Word Sense Disambiguation or Machine Translation. Heatmapping is used here to emphasize on the word which are responsible for the final decision made by the neural network model.



## Dependencies

Python>=3.5 + Numpy + Matplotlib, or alternatively simply install Anaconda.

Using Anaconda you can e.g. create a Python 3.6 environment: conda create -n py36 python=3.6 anaconda

Then activate it with: source activate py36

Before being able to use the code, you might need to run in the terminal: export PYTHONPATH=$PYTHONPATH:$pwd



## Usage

The folder model/ contains a word-based bidirectional LSTM model, that was trained for [Parts of Speech Tagger on Brown Corpus](https://github.com/NeoHimu/POS-Tagger-using-HMM-and-Deep-Learning)

The notebook run_example.ipynb provides a usage example of the code, it performs LRP and SA on a test sentence.



## Acknowledgments
[Explaining Recurrent Neural Network Predictions in Sentiment Analysis by L. Arras, G. Montavon, K.-R. Müller and W. Samek, 2017](http://aclweb.org/anthology/W/W17/W17-5221.pdf)



## More information

For further research and projects involving LRP, visit [heatmapping.org](http://heatmapping.org)
