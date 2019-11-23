import tensorflow.keras as keras


class FakeNewsDetectionNet(keras.Model):
    """
    Neural network model for fake news detection.

    :param dim_input: int, input dimension.
    :param dim_embeddings: int, embeddings dimension
        (e.g. 300 for fasttext).
    :param embeddings: numpy.array, embeddings matrix of dimension
        (dim_input x dim_embeddings).
    :param lstm_units: int, number of units in LSTM layer.
    :param num_hidden_layers: int, number of hidden dense layers.
    """

    def __init__(
            self,
            dim_input,
            dim_embeddings,
            embeddings,
            lstm_units,
            num_hidden_layers
    ):
        super(FakeNewsDetectionNet, self).__init__()
        self.embedding_layer = keras.layers.Embedding(
                input_dim=dim_input,
                output_dim=dim_embeddings,
                embeddings_initializer=keras.initializers.Constant(embeddings),
                trainable=False,
                mask_zero=True
        )
        self.lstm_layer = keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_units)
        )
        self.dense_layers = [
            keras.layers.Dense(
                units=lstm_units,
                activation='relu'
            )
            for _ in range(num_hidden_layers)
        ]
        self.final_dense = keras.layers.Dense(
                units=1,
                activation='sigmoid'
        )

    def call(self, input):
        x = self.embedding_layer(input)
        mask = self.embedding_layer.compute_mask(input)
        x = self.lstm_layer(x, mask=mask)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.final_dense(x)

        return x
