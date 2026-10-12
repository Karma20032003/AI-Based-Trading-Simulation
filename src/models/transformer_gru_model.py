"""
Transformer-GRU hybrid sequence classifier.

A small multi-head self-attention (Transformer encoder) block extracts
context-aware representations of the input sequence, which are then fed
into a GRU layer to model temporal order, followed by a classification head.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import RANDOM_SEED, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE

tf.random.set_seed(RANDOM_SEED)


def _transformer_encoder(inputs, head_size=32, num_heads=2, ff_dim=64, dropout=0.2):
    x = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)

    ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = tf.keras.layers.Dense(inputs.shape[-1])(ff)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff + x)


def build_transformer_gru(input_shape, n_classes=3):
    inputs = tf.keras.Input(shape=input_shape)
    x = _transformer_encoder(inputs)
    x = tf.keras.layers.GRU(32)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_transformer_gru(X_train, y_train_enc, X_val, y_val_enc, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, verbose=0):
    model = build_transformer_gru(input_shape=X_train.shape[1:])
    es = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
    history = model.fit(
        X_train, y_train_enc,
        validation_data=(X_val, y_val_enc),
        epochs=epochs, batch_size=batch_size,
        callbacks=[es], verbose=verbose,
    )

    val_pred_enc = np.argmax(model.predict(X_val, verbose=0), axis=1)
    acc = accuracy_score(y_val_enc, val_pred_enc)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val_enc, val_pred_enc, average="macro", zero_division=0
    )
    metrics = {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "final_val_loss": float(history.history["val_loss"][-1]),
    }
    return model, metrics, history
