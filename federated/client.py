import flwr as fl
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model.architecture import build_model

class EnergyIoTClient(fl.client.NumPyClient):
    def __init__(self, client_id, X, y, num_classes):
        self.client_id  = client_id
        self.num_classes = num_classes
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(X, y, test_size=0.2,
                             random_state=client_id)
        window_size = X.shape[1]
        n_features  = X.shape[2]
        self.model  = build_model(window_size, n_features, num_classes)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        y_train_cat = tf.keras.utils.to_categorical(
            self.y_train, self.num_classes)
        y_val_cat   = tf.keras.utils.to_categorical(
            self.y_val, self.num_classes)
        history = self.model.fit(
            self.X_train, y_train_cat,
            epochs=config.get("local_epochs", 3),
            batch_size=config.get("batch_size", 64),
            validation_data=(self.X_val, y_val_cat),
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=2, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=1, verbose=0),
            ]
        )
        return self.model.get_weights(), len(self.X_train), {
            "loss":     float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
        }

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        y_val_cat = tf.keras.utils.to_categorical(
            self.y_val, self.num_classes)
        loss, acc = self.model.evaluate(
            self.X_val, y_val_cat, verbose=0)
        return loss, len(self.X_val), {"accuracy": acc}

def make_client_fn(shards, num_classes):
    def client_fn(cid: str):
        X, y = shards[int(cid)]
        return EnergyIoTClient(int(cid), X, y, num_classes)
    return client_fn