import tensorflow as tf


class WeightTracker(tf.keras.callbacks.Callback):
    def __init__(self, layer_names):
        super().__init__()
        self.layer_names = layer_names  # Nazwy warstw do śledzenia
        self.weights_history = {name: [] for name in layer_names}

    def on_epoch_end(self, epoch, logs=None):
        for name in self.layer_names:
            layer = self.model.get_layer(name)
            weights, biases = layer.get_weights()  # Pobranie wag i biasów
            self.weights_history[name].append(weights.copy())  # Zapis wag
