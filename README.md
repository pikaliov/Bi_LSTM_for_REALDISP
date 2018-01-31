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
- Bidirectional-LSTM: [Schmidhuber, J., Graves, A.: Framewise phoneme classification with bidirectional LSTM and other neural network architectures. In: Neural Networks. 2005, H. 5-6, S. 602…610](http://www.sciencedirect.com/science/article/pii/S0893608005001206)

- Stacked Denoising: [Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., et al.:  Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion, In: Journal of Machine Learning Research. 2010, H. 11, S. 3371…3408.](http://www.jmlr.org/papers/v11/vincent10a.html)
