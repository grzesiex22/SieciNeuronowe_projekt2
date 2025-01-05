from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def predict(model_path, img_path):
    # Załaduj zapisany model
    model = load_model(model_path)

    # Krok 1: Załaduj obraz
    img = image.load_img(img_path, target_size=(128, 128))  # Zmieniamy rozmiar obrazu do rozmiaru wejściowego modelu

    # Krok 2: Przekształć obraz do formatu tablicy NumPy
    img_array = image.img_to_array(img)

    # Normalizuj obraz (przekształć wartości pikseli do przedziału [0, 1] jeśli to konieczne)
    img_array = img_array / 255.0

    # Rozszerz wymiary, aby dodać wymiar partii (batch dimension), ponieważ model oczekuje wejścia w formie (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Krok 3: Predykcja
    np.set_printoptions(suppress=True, precision=4)
    predictions = model.predict(img_array)
    print(f'Predictions: {predictions}')

    # Wynik jest tablicą prawdopodobieństw, więc możemy zobaczyć, jaka klasa ma najwyższe prawdopodobieństwo
    predicted_class = np.argmax(predictions, axis=1)

    # Etykiety klas
    class_labels = ['cat', 'dog', 'wild']

    # Wyświetlenie obrazu i predykcji
    plt.imshow(img)
    plt.title(f'Predicted class: {class_labels[predicted_class[0]]}')
    plt.show()

    print(f'The predicted class is: {class_labels[predicted_class[0]]}')

predict(model_path='models/7_animal_faces_model.h5', img_path='test_images/R.jpg')