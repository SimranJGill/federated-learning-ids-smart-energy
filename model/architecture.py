import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

def build_model(window_size, n_features, num_classes, dropout_rate=0.3):

    inputs = Input(shape=(window_size, n_features), name="input")

    # ── CNN Branch ────────────────────────────────────────────────
    cnn = layers.Conv1D(64, kernel_size=3, activation="relu",
                        padding="same")(inputs)
    cnn = layers.BatchNormalization()(cnn)
    cnn = layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = layers.Dropout(dropout_rate)(cnn)
    cnn = layers.Conv1D(128, kernel_size=3, activation="relu",
                        padding="same")(cnn)
    cnn = layers.BatchNormalization()(cnn)
    cnn = layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = layers.Dropout(dropout_rate)(cnn)
    cnn_out = layers.GlobalAveragePooling1D()(cnn)

    # ── BiLSTM Branch ─────────────────────────────────────────────
    bilstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True))(inputs)
    bilstm = layers.Dropout(dropout_rate)(bilstm)
    bilstm = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True))(bilstm)
    bilstm = layers.Dropout(dropout_rate)(bilstm)

    # ── Attention ─────────────────────────────────────────────────
    attn = layers.Dense(1, activation="tanh")(bilstm)
    attn = layers.Softmax(axis=1, name="attention_weights")(attn)
    attn = layers.Multiply()([bilstm, attn])
    bilstm_out = layers.GlobalAveragePooling1D()(attn)

    # ── Merge ─────────────────────────────────────────────────────
    merged = layers.Concatenate()([cnn_out, bilstm_out])
    x = layers.Dense(256, activation="relu")(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name="CNN_BiLSTM_Attention")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model