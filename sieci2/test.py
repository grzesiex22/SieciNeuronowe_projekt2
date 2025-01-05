# Załaduj zapisany model
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from Model import Model
from keras.src.saving import load_model

# Ścieżki do folderów 'train' i 'val'
test_dir = 'dataset/test'

# Załaduj dane testowe bez augmentacji
test_datagen = ImageDataGenerator(rescale=1. / 255)  # Tylko normalizacja

test_data = test_datagen.flow_from_directory(
    test_dir,  # Ścieżka do folderu z danymi testowymi
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
)

model = load_model('models/7_animal_faces_model.h5')

# Ocena modelu
results = model.evaluate(test_data)
test_loss = results[0]  # Pierwszy element to strata (loss)
test_acc = results[1]   # Drugi element to dokładność (accuracy)
test_mse = results[2]   # Trzeci element to błąd średniokwadratowy (MSE)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')
print(f'Test MSE: {test_mse}')