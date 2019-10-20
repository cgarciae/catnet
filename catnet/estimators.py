import tensorflow as tf
import math

from . import models
from . import utils


class CatNetClassifier:
    def __init__(
        self,
        n_classes,
        numerical_features,
        categorical_features,
        embeddings_sizes=None,
        layers=[200, 50],
        dropout=None,
        activation="relu",
        wide_n_deep=False,
        optimizer=None,
        loss=None,
        metrics=["acc", "mean_iou", "f1_score"],
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        callbacks=[],
        validation_data=None,
        name=None,
    ):
        assert validation_data is not None, "Please provide validation_data"

        self.n_classes = n_classes
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.embeddings_sizes = embeddings_sizes
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.wide_n_deep = wide_n_deep
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.validation_data = validation_data
        self.name = name

    def fit(self, X, y):

        n_classes = len(y.unique())

        self.category_sizes = [
            max(X[feature].max(), self.validation_data[0][feature].max()) + 1
            for feature in self.categorical_features
        ]

        if self.embeddings_sizes is None:
            self.embeddings_sizes = ["minus1"] * len(self.categorical_features)
        elif (
            isinstance(self.embeddings_sizes, (int, float))
            or self.embeddings_sizes == "minus1"
        ):
            self.embeddings_sizes = [self.embeddings_sizes] * len(
                self.categorical_features
            )

        for i in range(len(self.embeddings_sizes)):

            if self.embeddings_sizes[i] == "minus1":
                self.embeddings_sizes[i] = self.category_sizes[i] - 1
            elif self.embeddings_sizes[i] < 1:
                self.embeddings_sizes[i] = math.ceil(
                    self.embeddings_sizes[i] * self.category_sizes[i]
                )

        if self.optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.metrics = [
            [
                utils.MeanIoU(n_classes, name="mean_iou")
                if x == "mean_iou"
                else utils.F1Score(n_classes, "micro")
                if x == "f1_score"
                else x
                for x in self.metrics
            ]
        ]

        numerical_inputs, categorical_inputs = models.get_inputs(
            self.numerical_features, self.categorical_features
        )

        self.inputs = numerical_inputs + categorical_inputs

        self.model = models.CatNetClassifier2(
            n_classes=self.n_classes,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            category_sizes=self.category_sizes,
            embeddings_sizes=self.embeddings_sizes,
            layers=self.layers,
            dropout=self.dropout,
            activation=self.activation,
            wide_n_deep=self.wide_n_deep,
        )

        output = self.model(self.inputs)

        # self.model = tf.keras.Model(inputs=self.inputs, outputs=[net], name=self.name)

        self.model.summary()

        self.model.compile(
            self.optimizer, loss="sparse_categorical_crossentropy", metrics=self.metrics
        )

        self.all_features = self.numerical_features + list(self.categorical_features)

        data = [X[feature].to_numpy().reshape(-1, 1) for feature in self.all_features]

        # for f, n in data.items():
        #     print(f, n.shape)

        # exit()

        history = self.model.fit(
            x=data,
            y=y.to_numpy(),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=self.callbacks,
            shuffle=True,
            validation_data=(
                [
                    self.validation_data[0][feature].to_numpy()
                    for feature in self.numerical_features
                    + list(self.categorical_features)
                ],
                self.validation_data[1].to_numpy(),
            ),
        )

        return history

    def predict_proba(self, X):

        return self.model.predict(
            [X[feature].to_numpy().reshape(-1, 1) for feature in self.all_features]
        )

    def predict(self, X):

        return self.model.predict(
            [X[feature].to_numpy().reshape(-1, 1) for feature in self.all_features]
        )

    def save(self, filepath):
        self.model.save(filepath)

