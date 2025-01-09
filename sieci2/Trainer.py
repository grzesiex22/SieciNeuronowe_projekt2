import math
import numpy as np
from WeightTracker import WeightTracker


class Trainer:
    def __init__(self, model, train_data, val_data, initial_batch_size, increase_epoch, increase_factor,
                 max_batch_size, layer_names):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.initial_batch_size = initial_batch_size
        self.increase_epoch = increase_epoch
        self.increase_factor = increase_factor
        self.max_batch_size = max_batch_size
        self.global_history = {
            'loss': [], 'val_loss': [],
            'accuracy': [], 'val_accuracy': [],
            'mse': [], 'val_mse': [],
            'recall': [], 'val_recall': [],
            'precision': [], 'val_precision': [],
            'auc': [], 'val_auc': [],
            'f1_score': [], 'val_f1_score': [],
            'batch_size': [], 'lr': [],
        }
        self.weight_tracker = None
        self.layer_names = layer_names
        self.weights_history = {name: [] for name in self.layer_names}

        # Early stopping parameters
        self.early_stopping_patience = 3
        self.early_stopping_best_val_loss = np.inf
        self.early_stopping_counter = 0
        self.best_weights = None  # Zmienna na przechowywanie najlepszych wag
        self.best_epoch = None

        # Reduce learning rate parameters
        self.reduce_lr_patience = 2
        self.reduce_lr_best_val_loss = np.inf
        self.reduce_lr_factor = 0.5
        self.reduce_lr_counter = 0

    def train(self, epochs):
        batch_size = self.initial_batch_size
        current_epoch = 0

        print(f"tarin_data: {self.train_data.samples}")
        print(f"val_data: {self.val_data.samples}")


        while current_epoch < epochs:
            print(f"\nStarting training with batch_size: {batch_size}")
            # Update batch size in data generators
            self.train_data.batch_size = batch_size
            self.val_data.batch_size = batch_size
            steps_per_epoch = math.floor(self.train_data.samples / batch_size)
            validation_steps = math.floor(self.val_data.samples / batch_size)

            # Dodanie callbacku podczas treningu
            self.weight_tracker = WeightTracker(layer_names=self.layer_names)

            # Train one epoch at a time
            for epoch in range(self.increase_epoch):
                if current_epoch >= epochs:
                    break

                print(f"\nEpoch {current_epoch + 1}/{epochs}")
                print(f"Current batch size: {self.train_data.batch_size}")
                current_lr = self.model.optimizer.get_config().get("learning_rate", None)
                print(f"Current learning rate: {current_lr}")


                history = self.model.fit(
                    self.train_data,
                    epochs=current_epoch + 1,
                    initial_epoch=current_epoch,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=self.val_data,
                    validation_steps=validation_steps,
                    verbose=1,
                    callbacks=[self.weight_tracker]
                )

                # Update global history
                for key in self.global_history.keys():
                    self.global_history[key].extend(history.history.get(key, []))
                # print(f'global history = {history.history}')

                # Obliczanie F1-score na podstawie Precision i Recall
                precision = history.history['precision'][-1]  # Lista wartości precyzji z historii treningu
                recall = history.history['recall'][-1]  # Lista wartości czułości z historii treningu
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                # Obliczanie F1-score dla każdej epoki
                self.global_history['f1_score'].append(f1_score)

                # Obliczanie val_F1-score na podstawie Precision i Recall
                val_precision = history.history['val_precision'][-1]  # Lista wartości precyzji z historii treningu
                val_recall = history.history['val_recall'][-1]  # Lista wartości czułości z historii treningu
                val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall) \
                    if (val_precision + val_recall) > 0 else 0
                # Obliczanie val_F1-score dla każdej epoki
                self.global_history['val_f1_score'].append(val_f1_score)

                # Add batch_size and learning_rate to global_history for this epoch
                self.global_history['batch_size'].append(batch_size)
                self.global_history['lr'].append(current_lr)

                current_val_loss = history.history['val_loss'][-1]
                print(f"Validation loss after epoch {current_epoch + 1}: {current_val_loss}")

                for name in self.layer_names:
                    layer = self.model.get_layer(name)
                    weights, biases = layer.get_weights()  # Pobranie wag i biasów
                    self.weights_history[name].append(weights.copy())  # Zapis wag

                # Early stopping check
                if current_val_loss < self.early_stopping_best_val_loss:
                    self.early_stopping_best_val_loss = current_val_loss
                    self.best_weights = self.model.get_weights()  # Zapis najlepszych wag
                    self.best_epoch = current_epoch+1
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping triggered. Training stopped. Best epoch: {self.best_epoch}")
                        self.model.set_weights(self.best_weights)  # Przywrócenie najlepszych wag
                        return self.global_history, self.weights_history
                print(
                    f"EarlyStopping counter: {self.early_stopping_counter}, Best val_loss: {self.early_stopping_best_val_loss}")

                # Reduce learning rate check
                if current_val_loss < self.reduce_lr_best_val_loss:
                    self.reduce_lr_best_val_loss = current_val_loss
                    self.reduce_lr_counter = 0
                else:
                    self.reduce_lr_counter += 1
                    if self.reduce_lr_counter >= self.reduce_lr_patience:
                        # Pobierz aktualny learning rate z konfiguracji lub dynamicznej wartości
                        current_lr = self.model.optimizer.learning_rate
                        if callable(current_lr):  # Jeśli jest dynamiczny (np. Schedule)
                            current_lr = current_lr(self.model.optimizer.iterations).numpy()
                        else:
                            current_lr = float(current_lr)

                        # Zmniejsz współczynnik uczenia
                        new_lr = current_lr * self.reduce_lr_factor
                        self.model.optimizer.learning_rate = new_lr  # Przypisz nową wartość
                        print(f"\nLearning rate reduced from {current_lr} to {new_lr}")
                        self.reduce_lr_counter = 0

                print(f"ReduceLR counter: {self.reduce_lr_counter}, Best val_loss: {self.reduce_lr_best_val_loss}")

                current_epoch += 1

            # Increase batch size if applicable
            if batch_size < self.max_batch_size:
                batch_size = int(batch_size * self.increase_factor)
                if batch_size > self.max_batch_size:
                    batch_size = self.max_batch_size

        print("\nTraining complete.")
        return self.global_history, self.weights_history
