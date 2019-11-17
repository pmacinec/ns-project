import tensorflow.keras as keras


class FakeNewsDetectionNet(keras.Model):
    """
    Neural network model for fake news detection.

    :param dim_input: int, input dimension.
    :param dim_embeddings: int, embeddings dimension
        (e.g. 300 for fasttext).
    :param embeddings: numpy.array, embeddings matrix of dimension
        (dim_input x dim_embeddings).
    """

    def __init__(self, dim_input, dim_embeddings, embeddings):
        super(FakeNewsDetectionNet, self).__init__()
        self.embedding_layer = keras.layers.Embedding(
                input_dim=dim_input,
                output_dim=dim_embeddings,
                embeddings_initializer=keras.initializers.Constant(embeddings),
                trainable=False,
                mask_zero=True
        )
        self.lstm_layer = keras.layers.LSTM(64)
        self.dense_layer = keras.layers.Dense(
                units=64,
                activation='relu'
        )
        self.final_dense = keras.layers.Dense(
                units=1,
                activation='sigmoid'
        )

    def call(self, input):
        x = self.embedding_layer(input)
        mask = self.embedding_layer.compute_mask(input)
        x = self.lstm_layer(x, mask=mask)
        x = self.dense_layer(x)
        x = self.final_dense(x)

        return x
