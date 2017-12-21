# Bi_LSTM_for_REALDISP

My first steps in machine learning to classify the activitys in the [REALDISP Activity Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/REALDISP+Activity+Recognition+Dataset). 

First I trained a Stacked Denoising Autoencoder (SDA) for dimension reduction and feature extraction. After that, I used the resulting latent representation of the features to train a Bidirectional-LSTM network for activity classification. 

I used [Keras](https://keras.io/) and other libs to implement my Network with [TensorFlow](https://www.tensorflow.org/) backend. 

Include:
- Dataset preparation
- Data visualisation 
- Train a SDA 
- Train a Bi-LSTM and classify the activitys 


### Citing
The Bidirectional-LSTM is a special recurrent neural network (RNN), which was defined by [J. Schmidhuber & A. Graves](http://www.sciencedirect.com/science/article/pii/S0893608005001206).
The Stacked Denoising Autoencoder was defined by [P. Vincent, H. Larochelle, I. Lajoie, Y. Bengio, et al.](http://www.jmlr.org/papers/v11/vincent10a.html).
