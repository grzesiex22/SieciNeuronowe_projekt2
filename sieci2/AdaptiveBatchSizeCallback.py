from tensorflow.keras.callbacks import Callback

# Klasyczny sposób zmiany rozmiaru batcha po określonej liczbie epok
class AdaptiveBatchSizeCallback(Callback):
    def __init__(self, train_data, val_data, initial_batch_size, increase_epoch, increase_factor, max_batch_size):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.initial_batch_size = initial_batch_size
        self.increase_epoch = increase_epoch
        self.increase_factor = increase_factor
        self.max_batch_size = max_batch_size
        self.steps_per_epoch = train_data.samples // self.initial_batch_size
        self.validation_steps = val_data.samples // self.initial_batch_size

    def on_epoch_end(self, epoch, logs=None):
        # Zwiększamy batch_size w określonym momencie
        if epoch + 1 >= self.increase_epoch:
            new_batch_size = int(self.initial_batch_size * self.increase_factor)
            if new_batch_size <= self.max_batch_size:
                self.initial_batch_size = new_batch_size
                print(f"\nEpoch {epoch + 1}: Zwiększono batch_size do {self.initial_batch_size}")

                # Zmiana batch_size w generatorze danych
                self.train_data.batch_size = self.initial_batch_size
                self.val_data.batch_size = self.initial_batch_size

                # Wyliczenie nowych wartości steps_per_epoch i validation_steps
                self.steps_per_epoch = self.train_data.samples // self.initial_batch_size
                self.validation_steps = self.val_data.samples // self.initial_batch_size

                print(f"Nowe steps_per_epoch: {self.steps_per_epoch}, validation_steps: {self.validation_steps}")
            else:
                print(f"\nEpoch {epoch + 1}: Osiągnięto maksymalny rozmiar batch_size.")
