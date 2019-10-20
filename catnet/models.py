import tensorflow as tf


class CatNetClassifier(tf.keras.Model):
    def __init__(
        self,
        n_classes,
        numerical_features,
        categorical_features,
        category_sizes,
        embeddings_sizes,
        layers=[200, 50],
        activation="relu",
        wide_n_deep=False,
        return_predictions=True,
        name=None,
    ):
        super().__init__(name=name)

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.category_sizes = category_sizes
        self.embeddings_sizes = embeddings_sizes
        self.activation_fn = activation
        self.wide_n_deep = wide_n_deep
        self.return_predictions = return_predictions

        self.categorical_embeddings = [
            tf.keras.layers.Embedding(
                input_dim=category_size,
                output_dim=embeddings_size,
                name=f"{feature}_embedding",
            )
            for feature, category_size, embeddings_size in zip(
                categorical_features, category_sizes, embeddings_sizes
            )
        ]

        self.concatenate = tf.keras.layers.Concatenate()
        self.dense_layers = [
            tf.keras.layers.Dense(size, activation=activation) for size in layers
        ]

        if wide_n_deep:
            self.wide_n_deep_concatenate = tf.keras.layers.Concatenate()

        if return_predictions:
            self.predictions = tf.keras.layers.Dense(
                n_classes, activation="softmax", name="predictions"
            )

    @tf.function
    def call(self, inputs, training=None):

        if training is None:
            training = tf.keras.backend.learning_phase()

        all_features = self.numerical_features + self.categorical_features

        assert len(inputs) == len(all_features)

        inputs = dict(zip(all_features, inputs))

        # [print(f, t.shape) for (f, t) in inputs.items()]

        # exit()

        numerical_inputs = [inputs[feature] for feature in self.numerical_features]
        categorical_inputs = [inputs[feature] for feature in self.categorical_features]
        # [print(x.shape) for x in categorical_inputs]
        categorical_inputs = [tf.squeeze(input, axis=1) for input in categorical_inputs]

        # print(list(zip(categorical_features, category_sizes, embeddings_sizes)))

        # [print(x.shape) for x in categorical_inputs]

        categorical_embeddings = [
            embeddings(input)
            for input, embeddings in zip(
                categorical_inputs, self.categorical_embeddings
            )
        ]

        # [print(x.shape) for x in numerical_inputs + categorical_embeddings]

        embeddings = self.concatenate(numerical_inputs + categorical_embeddings)

        deep = embeddings

        for layer in self.dense_layers:
            deep = layer(deep)

        if self.wide_n_deep:
            net = self.wide_n_deep_concatenate([embeddings, deep])
        else:
            net = deep

        if self.return_predictions:
            return self.predictions(net)
        else:
            return net


def get_inputs(numerical_features, categorical_features):
    numerical_inputs = [
        tf.keras.layers.Input(shape=[1], dtype=tf.float32, name=input_name)
        for input_name in numerical_features
    ]

    # categorical
    categorical_input = [
        tf.keras.layers.Input(shape=[1], dtype=tf.int32, name=input_name)
        for input_name in categorical_features
    ]

    return numerical_inputs, categorical_input


def CatNetClassifier2(
    n_classes,
    numerical_features,
    categorical_features,
    category_sizes,
    embeddings_sizes,
    layers=[200, 50],
    dropout=None,
    activation="relu",
    wide_n_deep=False,
    name=None,
):

    if not hasattr(dropout, "__iter__"):
        dropout = [dropout] * len(layers)

    numerical_inputs, categorical_inputs = get_inputs(
        numerical_features, categorical_features
    )

    squeezed_categorical_inputs = [
        tf.keras.layers.Reshape(())(input) for input in categorical_inputs
    ]

    # print(list(zip(categorical_features, category_sizes, embeddings_sizes)))

    categorical_embeddings = [
        tf.keras.layers.Embedding(
            input_dim=category_size,
            output_dim=embeddings_size,
            name=f"{feature}_embedding",
        )(input)
        for input, feature, category_size, embeddings_size in zip(
            squeezed_categorical_inputs,
            categorical_features,
            category_sizes,
            embeddings_sizes,
        )
    ]

    # for t in numerical_inputs + categorical_embeddings:
    #     print(t.name, t.dtype, t.shape)

    # exit()

    embeddings = tf.keras.layers.concatenate(numerical_inputs + categorical_embeddings)

    deep = embeddings

    for n_neurons, layer_dropout in zip(layers, dropout):
        deep = tf.keras.layers.Dense(n_neurons, activation=activation)(deep)

        if layer_dropout is not None:
            deep = tf.keras.layers.Dropout(layer_dropout)(deep)

    if layers and wide_n_deep:
        net = tf.keras.layers.concatenate([embeddings, deep], axis=-1)
    else:
        net = deep

    net = tf.keras.layers.Dense(n_classes, activation="softmax", name="predictions")(
        net
    )

    return tf.keras.Model(
        inputs=numerical_inputs + categorical_inputs, outputs=[net], name=name
    )

