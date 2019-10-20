import tensorflow as tf
import typing as tp


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


def CatNetClassifier(
    n_classes,
    numerical_features: [str],
    categorical_features: tp.Dict[
        str, tp.Tuple[int, int]
    ],  # name: (category_size, embedding_size)
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

    categorical_embeddings = [
        tf.keras.layers.Embedding(
            input_dim=category_size,
            output_dim=embeddings_size,
            name=f"{feature}_embedding",
        )(input)
        for input, (feature, (category_size, embeddings_size)) in zip(
            squeezed_categorical_inputs, categorical_features.items()
        )
    ]

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

