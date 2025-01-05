import json
import os
import tensorflow as tf
from keras.src.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from AdaptiveBatchSizeCallback import AdaptiveBatchSizeCallback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from Trainer import Trainer

class Model:
    def __init__(self, train_path, val_path, test_path):
        """
        Inicjalizacja klasy i wczytanie danych.

        :param train_path: Ścieżka do pliku CSV z danymi treningowymi.
        :param test_path: Ścieżka do pliku CSV z danymi testowymi.

        Publiczne atrybuty klasy:
        - train_data1, test_data1: Przechowują dane surowe i wstępnie przetworzone z pierwszego zestawu podejścia.
        - train_data2, test_data2: Przechowują dane surowe i wstępnie przetworzone z drugiego zestawu podejścia.
        - x1, y1: Przechowują cechy i etykiety dla pierwszego zestawu podejścia.
        - x2, y2: Przechowują cechy i etykiety dla drugiego zestawu podejścia.
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path

        # Dane dla podejścia 1 (bez 'cabin', z prefiksami biletów)
        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.model = None
        self.history = None
        self.classes = ['cat', 'dog', 'wild']

    def load_data(self):
        """Wczytaj dane z folderów z użyciem ImageDataGenerator."""
        # Augmentacja dla danych treningowych
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,  # Normalizacja pikseli
            rotation_range=40,  # Obrót
            width_shift_range=0.1,  # Przesunięcie poziome
            height_shift_range=0.1,  # Przesunięcie pionowe
            shear_range=0.2,  # Zniekształcenie
            zoom_range=0.2,  # Powiększenie
            horizontal_flip=True,  # Odbicie lustrzane
            fill_mode='nearest',  # Uzupełnianie
        )

        # Załaduj dane treningowe
        self.train_data = train_datagen.flow_from_directory(
            self.train_path,  # Ścieżka do folderu z danymi treningowymi
            target_size=(128, 128),  # Rozmiar obrazów, do którego je przeskalujemy
            batch_size=32,
            class_mode="categorical",  # Wieloklasowa klasyfikacja
        )

        # Załaduj dane walidacyjne bez augmentacji
        val_datagen = ImageDataGenerator(rescale=1. / 255)  # Tylko normalizacja

        self.val_data = val_datagen.flow_from_directory(
            self.val_path,  # Ścieżka do folderu z danymi treningowymi
            target_size=(128, 128),  # Rozmiar obrazów
            batch_size=32,
            class_mode="categorical",  # Wieloklasowa klasyfikacja
        )

        # Załaduj dane testowe bez augmentacji
        test_datagen = ImageDataGenerator(rescale=1. / 255)  # Tylko normalizacja

        self.test_data = test_datagen.flow_from_directory(
            self.test_path,  # Ścieżka do folderu z danymi testowymi
            target_size=(128, 128),
            batch_size=32,
            class_mode="categorical",
        )
        # x = self.train_data.samples
        print("Dane zostały wczytane.")


    def create_model(self):
        # Tworzymy model CNN
        self.model = tf.keras.Sequential([
            # Warstwa konwolucyjna
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Spłaszczanie wyników
            tf.keras.layers.Flatten(),

            # Warstwa w pełni połączona
            tf.keras.layers.Dense(128, activation='relu'),

            # Warstwa wyjściowa
            tf.keras.layers.Dense(len(self.train_data.class_indices), activation='softmax')
            # Liczba klas (np. 3: dog, cat, wild)
        ])

        # Tworzenie optymalizatora SGD z momentum
        optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

        # Kompilacja modelu
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'mse'])

        # Podsumowanie modelu
        self.model.summary()

    def train(self, epochs=10):

        if not self.model:
            print("Model has not been created")
            return

        # Tworzymy zmienny współczynnik uczenia
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',  # Monitorujemy stratę na zbiorze walidacyjnym
            factor=0.5,  # Redukcja współczynnika uczenia o połowę
            patience=2,  # Jeśli brak poprawy przez 2 epoki, zmniejsz współczynnik uczenia
            verbose=1  # Wyświetlanie komunikatów
        )

        # Tworzymy EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitorujemy stratę na zbiorze walidacyjnym
            patience=3,  # Czas oczekiwania na poprawę (3 epoki)
            restore_best_weights=True,  # Przywrócenie najlepszej wagi modelu
            verbose=1  # Wyświetlanie komunikatów
        )

        # # Tworzenie callbacku
        # adaptive_batch_size_callback = AdaptiveBatchSizeCallback(
        #     train_data=self.train_data,  # Generator danych treningowych
        #     val_data=self.val_data,  # Generator danych walidacyjnych
        #     initial_batch_size=32,  # Początkowy rozmiar batcha
        #     increase_epoch=1,  # Po jakiej epoce zwiększymy batch size
        #     increase_factor=1.2,  # O ile razy zwiększymy batch size
        #     max_batch_size=256  # Maksymalny dozwolony rozmiar batcha
        # )

        # Trenowanie modelu
        self.history = self.model.fit(
            self.train_data,  # Zestaw treningowy
            epochs=epochs,  # Liczba epok, możesz zwiększyć w zależności od czasu
            validation_data=self.val_data,  # Zestaw walidacyjny
            verbose=1,
            callbacks=[early_stopping, reduce_lr]  # Dodanie callbacków
        )

    def train2(self, epochs=10):

        if not self.model:
            print("Model has not been created")
            return

        trainer = Trainer(
            model=self.model,
            train_data=self.train_data,
            val_data=self.val_data,
            initial_batch_size=32,
            increase_epoch=3,
            increase_factor=1.35,
            max_batch_size=256
        )

        self.history = trainer.train(epochs=20)

    def save_model_architecture(self, filename_prefix="0", folder="models"):
        # Tworzenie brakujących folderów, jeśli nie istnieją
        os.makedirs(folder, exist_ok=True)

        # Zapis architektury modelu
        file_path = os.path.join(folder, f'{filename_prefix}_architecture.json')
        model_json = self.model.to_json()

        model_dict = json.loads(model_json)
        with open(file_path, "w") as json_file:
            json.dump(model_dict, json_file, indent=4)  # Dodajemy `indent=4`

    def save_model(self, filename_prefix="0", folder="models"):
        # Tworzenie brakujących folderów, jeśli nie istnieją
        os.makedirs(folder, exist_ok=True)

        # Zapisz historię do pliku JSON
        file_path = os.path.join(folder, f'{filename_prefix}_animal_faces_model.h5')
        self.model.save(file_path)  # Zapisanie modelu

    def save_history(self, filename_prefix="0", folder="history"):
        # Tworzenie brakujących folderów, jeśli nie istnieją
        os.makedirs(folder, exist_ok=True)

        # Zapisz historię do pliku JSON
        file_path = os.path.join(folder, f'{filename_prefix}_training_history.json')

        with open(file_path, 'w') as f:
            json.dump(self.history.history, f)

    def save_history2(self, filename_prefix="0", folder="history"):
        # Tworzenie brakujących folderów, jeśli nie istnieją
        os.makedirs(folder, exist_ok=True)

        # Zapisz historię do pliku JSON
        file_path = os.path.join(folder, f'{filename_prefix}_training_history.json')

        # Sprawdzamy, czy self.history to obiekt z atrybutem history
        if hasattr(self.history, 'history'):
            # Zapisz tylko słownik z wynikami
            with open(file_path, 'w') as f:
                json.dump(self.history.history, f)
        else:
            with open(file_path, 'w') as f:
                json.dump(self.history, f)
