import Model
import Wykresy
import os
import re

def get_next_prefix(folder="models"):
    # Tworzenie folderu, jeśli nie istnieje
    os.makedirs(folder, exist_ok=True)

    # Pobierz listę plików w folderze
    files = os.listdir(folder)

    # Wyciągnięcie numerków z nazw plików
    prefix_numbers = []
    for file in files:
        match = re.match(r"(\d+)_.*", file)
        if match:
            prefix_numbers.append(int(match.group(1)))

    # Znajdź najwyższy numerek i zwróć o 1 większy
    return max(prefix_numbers, default=0) + 1


# Ścieżki do folderów 'train' i 'val'
train_dir = 'dataset/train'
val_dir = 'dataset/val'
test_dir = 'dataset/test'


model = Model.Model(train_path=train_dir, val_path=val_dir, test_path=test_dir)
model.load_data()
model.create_model()
model.train2(epochs=40)

n = get_next_prefix()
# dane do zapisu do plików
model.save_model_architecture(filename_prefix=n)
model.save_model(filename_prefix=n)
model.save_history2(filename_prefix=n)
model.save_weights_history_to_json(filename_prefix=n)
# wykresy
Wykresy.plot_training_history(history=model.history, filename_prefix=n)
Wykresy.plot_additional_training_history(history=model.history, filename_prefix=n)
Wykresy.plot_weights_average(weights_history=model.weights_history, layer_names=model.layer_names, filename_prefix=n)
