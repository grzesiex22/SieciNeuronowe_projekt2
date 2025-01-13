# Załaduj zapisany model
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from Model import Model
from keras.src.saving import load_model

# Ścieżki do folderów 'train' i 'val'
test_dir = 'dataset/test'


"""
BEZ AUGUMENTACJI
"""
print("Bez augumentacji")

# Załaduj dane testowe bez augmentacji
test_datagen = ImageDataGenerator(rescale=1. / 255)  # Tylko normalizacja

test_data = test_datagen.flow_from_directory(
    test_dir,  # Ścieżka do folderu z danymi testowymi
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
)

model = load_model('models/26_animal_faces_model.h5')

# Ocena modelu
results = model.evaluate(test_data)
test_loss = results[0]  # Pierwszy element to strata (loss)
test_acc = results[1]   # Drugi element to dokładność (accuracy)
test_mse = results[2]   # Trzeci element to błąd średniokwadratowy (MSE)
test_prec = results[3]  # Pierwszy element to strata (loss)
test_recall = results[4]   # Drugi element to dokładność (accuracy)
test_auc = results[5]   # Trzeci element to błąd średniokwadratowy (MSE)
test_f1 = 2 * (test_prec * test_recall) / (test_prec + test_recall) if (test_prec + test_recall) > 0 else 0

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')
print(f'Test MSE: {test_mse}')
print(f'Test Precision: {test_prec}')
print(f'Test Recall: {test_recall}')
print(f'Test F1-score: {test_f1}')
print(f'Test Roc_AUC: {test_auc}')

"""
Z AUGUMENTACJĄ
"""
print("Z augumentacja")

test_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalizacja pikseli
    rotation_range=40,  # Obrót
    width_shift_range=0.1,  # Przesunięcie poziome
    height_shift_range=0.1,  # Przesunięcie pionowe
    shear_range=0.2,  # Zniekształcenie
    zoom_range=0.2,  # Powiększenie
    horizontal_flip=True,  # Odbicie lustrzane
    fill_mode='nearest',  # Uzupełnianie
)

test_data = test_datagen.flow_from_directory(
    test_dir,  # Ścieżka do folderu z danymi testowymi
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
)

model = load_model('models/26_animal_faces_model.h5')

# Ocena modelu
results = model.evaluate(test_data)
test_loss = results[0]  # Pierwszy element to strata (loss)
test_acc = results[1]   # Drugi element to dokładność (accuracy)
test_mse = results[2]   # Trzeci element to błąd średniokwadratowy (MSE)
test_prec = results[3]  # Pierwszy element to strata (loss)
test_recall = results[4]   # Drugi element to dokładność (accuracy)
test_auc = results[5]   # Trzeci element to błąd średniokwadratowy (MSE)
test_f1 = 2 * (test_prec * test_recall) / (test_prec + test_recall) if (test_prec + test_recall) > 0 else 0

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')
print(f'Test MSE: {test_mse}')
print(f'Test Precision: {test_prec}')
print(f'Test Recall: {test_recall}')
print(f'Test F1-score: {test_f1}')
print(f'Test Roc_AUC: {test_auc}')