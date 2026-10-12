"""
GRU sequence classifier (BUY / HOLD / SELL) built with TensorFlow/Keras.
"""
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import RANDOM_SEED, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE

tf.random.set_seed(RANDOM_SEED)


def build_gru(input_shape, n_classes=3):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.GRU(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GRU(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_gru(X_train, y_train_enc, X_val, y_val_enc, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, verbose=0):
    model = build_gru(input_shape=X_train.shape[1:])
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
