import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant


class FakeNewsDetectionNet(keras.Model):

    def __init__(self, dim_input, dim_embeddings, dim_output, embeddings):
        super(FakeNewsDetectionNet, self).__init__()
        self.model_layers = [
            layers.Embedding(
                input_dim=dim_input,
                output_dim=dim_embeddings,
                embeddings_initializer=Constant(embeddings),
                trainable=False
            ),
            layers.LSTM(128),
            layers.Dense(
                units=128,
                activation='relu'
            ),
            layers.Dense(
                units=dim_output,
                activation='softmax'
            )
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
