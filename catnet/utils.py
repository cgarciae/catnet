import tensorflow as tf
import tensorflow_addons as tfa


class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)


class F1Score(tfa.metrics.F1Score):
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)
        return tf.reduce_mean(
            super().__call__(y_true, y_pred, sample_weight=sample_weight)
        )
