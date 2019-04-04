# SERN
A Self-Attentive Emotion Recognition Network

Modern deep learning approaches have achieved groundbreaking performance in modeling and classifying sequential data. 
Specifically, attention networks constitute the state-of-the-art paradigm for capturing long temporal dynamics. 
This paper examines the efficacy of this paradigm in the challenging task of emotion recognition in dyadic conversations. 
In contrast to existing approaches, our work introduces a novel attention mechanism capable of inferring the immensity of the effect of each past utterance on the current speaker emotional state.
The proposed attention mechanism performs this inference procedure without the need of a decoder network; this is achieved by means of innovative self-attention arguments. 
Our self-attention networks capture the correlation patterns among consecutive encoder network states, thus allowing to robustly and effectively model temporal dynamics over arbitrary long temporal horizons.
Thus, we enable capturing strong affective patterns over the course of long discussions.
We exhibit the effectiveness of our approach considering the challenging IEMOCAP benchmark. 
As we show, our devised methodology outperforms state-of-the-art alternatives and commonly used approaches, giving rise to promising new research directions in the context of Online Social Network (OSN) analysis tasks. 

# Requirements
In order to experiment with SERN you need to install the following dependencies:
1. Python 3.6
2. TensorFlow 1.13.1
3. Gensim 3.7.1
4. NLTK 3.4
5. Scikit-learn
