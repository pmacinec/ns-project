import tensorflow.keras as keras


class FakeNewsDetectionNet(keras.Model):
    """
    Neural network model for fake news detection.

    :param dim_input: int, input dimension.
    :param dim_embeddings: int, embeddings dimension
        (e.g. 300 for fasttext).
    :param dim_output: int, output dimension
        (e.g. 2 for binary classification).
    :param embeddings: numpy.array, embeddings matrix of dimension
        (dim_input x dim_embeddings).
    """

    def __init__(self, dim_input, dim_embeddings, dim_output, embeddings):
        super(FakeNewsDetectionNet, self).__init__()
        self.model_layers = [
            keras.layers.Embedding(
                input_dim=dim_input,
                output_dim=dim_embeddings,
                embeddings_initializer=keras.initializers.Constant(embeddings),
                trainable=False
            ),
            keras.layers.LSTM(128),
            keras.layers.Dense(
                units=128,
                activation='relu'
            ),
            keras.layers.Dense(
                units=dim_output,
                activation='softmax'
            )
        ]

    def call(self, x):
        for layer in self.model_layers:
            x = layer(x)
        return x
