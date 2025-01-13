import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt

def fgsm_attack(image, epsilon, gradient):
    """
    Implementacja ataku FGSM.
    :param image: Obraz wejściowy (tensor).
    :param epsilon: Wartość zakłócenia (siła ataku).
    :param gradient: Gradient w stosunku do obrazu wejściowego.
    :return: Zmodyfikowany obraz z zakłóceniami.
    """
    signed_gradient = tf.sign(gradient)
    perturbed_image = image + epsilon * signed_gradient
    return tf.clip_by_value(perturbed_image, 0, 1)

def generate_adversarial_examples(model, dataset, epsilon, max_samples=10):
    """
    Generuje przykłady adversarialne dla podanego modelu i zbioru danych.
    :param model: Wytrenowany model.
    :param dataset: Generator danych.
    :param epsilon: Wartość zakłócenia.
    :param max_samples: Maksymalna liczba przykładów do wygenerowania.
    :return: Oryginalne obrazy, obrazy z zakłóceniami, prawdziwe etykiety, przewidywane etykiety.
    """

    original_images = []
    adversarial_images = []
    true_labels = []
    predicted_labels = []

    sample_count = 0

    for images, labels in dataset:
        images = tf.convert_to_tensor(images)  # Konwersja na tensor
        labels = tf.convert_to_tensor(labels)

        with tf.GradientTape() as tape:
            tape.watch(images)
            predictions = model(images, training=False)
            loss = tf.keras.losses.categorical_crossentropy(labels, predictions)

        gradient = tape.gradient(loss, images)

        for i in range(images.shape[0]):
            if sample_count >= max_samples:
                break

            original = images[i]
            perturbed = fgsm_attack(original, epsilon, gradient[i])

            original_images.append(original.numpy())
            adversarial_images.append(perturbed.numpy())
            true_labels.append(np.argmax(labels[i].numpy()))
            adversarial_pred = model.predict(perturbed[None, ...])
            predicted_labels.append(np.argmax(adversarial_pred))

            sample_count += 1

        if sample_count >= max_samples:
            break

    return np.array(original_images), np.array(adversarial_images), np.array(true_labels), np.array(predicted_labels)


def generate_and_plot_adversarial_examples(model, dataset, epsilons, max_samples=1):
    """
    Generuje zakłócone wersje obrazów przy różnych wartościach epsilon i wyświetla wyniki.
    :param model: Wytrenowany model.
    :param dataset: Generator danych (obrazy i etykiety).
    :param epsilons: Lista wartości epsilon dla zakłóceń.
    :param max_samples: Maksymalna liczba obrazów do wyświetlenia.
    """

    sample_count = 0

    for images, labels in dataset:
        images = tf.convert_to_tensor(images)  # Konwersja na tensor
        labels = tf.convert_to_tensor(labels)

        for i in range(images.shape[0]):
            if sample_count >= max_samples:
                break

            image = images[i]
            label = labels[i]

            adversarial_images = []
            predicted_labels = []

            for epsilon in epsilons:
                with tf.GradientTape() as tape:
                    tape.watch(image)
                    predictions = model(image[None, ...], training=False)
                    loss = tf.keras.losses.categorical_crossentropy(label[None, ...], predictions)

                gradient = tape.gradient(loss, image)
                perturbed_image = fgsm_attack(image, epsilon, gradient)

                adversarial_images.append(perturbed_image.numpy())
                adversarial_pred = model.predict(perturbed_image[None, ...])
                predicted_labels.append(np.argmax(adversarial_pred))

            # Wyświetlanie wyników dla jednego obrazu
            plt.figure(figsize=(15, 5))

            # Wyświetlenie oryginalnego obrazu z tłumaczeniem etykiety
            plt.subplot(1, len(epsilons) + 1, 1)
            plt.title(f"Oryginalny\n{label_to_name(np.argmax(label))}")
            plt.imshow(image.numpy())
            plt.axis('off')

            # Wyświetlanie obrazów zakłóconych z tłumaczeniem etykiety
            for j, epsilon in enumerate(epsilons):
                plt.subplot(1, len(epsilons) + 1, j + 2)
                plt.title(f"Epsilon: {epsilon}\nPred: {label_to_name(predicted_labels[j])}")
                plt.imshow(adversarial_images[j])
                plt.axis('off')

            plt.show()

            sample_count += 1

        if sample_count >= max_samples:
            break


# Funkcja do tłumaczenia etykiet na nazwy
def label_to_name(label):
    if label == 0:
        return "Kot"
    elif label == 1:
        return "Pies"
    elif label == 2:
        return "Dzikie zwierzę"
    else:
        return "Nieznane"

def evaluate_model_on_epsilons(model, dataset, epsilons, max_samples=1000):
    """
    Ocena modelu dla różnych wartości epsilon.
    :param model: Wytrenowany model.
    :param dataset: Generator danych.
    :param epsilons: Lista wartości epsilon.
    :param max_samples: Maksymalna liczba przykładów dla każdej wartości epsilon.
    :return: Lista skuteczności dla każdej wartości epsilon.
    """
    accuracies = []

    for epsilon in epsilons:
        print(f"Przetwarzanie dla epsilon = {epsilon}...")
        correct_predictions = 0
        total_samples = 0

        # Generuj przykłady z zakłóceniami
        original, adversarial, true_labels, predicted_labels = generate_adversarial_examples(
            model, dataset, epsilon, max_samples=max_samples
        )

        # Liczba poprawnych przewidywań
        correct_predictions = np.sum(true_labels == predicted_labels)
        total_samples = len(true_labels)

        # Oblicz dokładność
        accuracy = correct_predictions / total_samples
        accuracies.append(accuracy)

        print(f"Epsilon: {epsilon}, Skuteczność: {accuracy:.2%}")

    return accuracies



if __name__ == "__main__":
    import Model

    # Wczytaj model
    train_dir = r'C:\Users\patry\Downloads\dataset\train'
    val_dir = r'C:\Users\patry\Downloads\dataset\val'
    test_dir = r'C:\Users\patry\Downloads\dataset\test'

    model_obj = Model.Model(train_path=train_dir, val_path=val_dir, test_path=test_dir)
    model_obj.load_data()
    model_obj.create_model()
    model_obj.model.load_weights("models/26_animal_faces_model.h5")  # Podaj ścieżkę do wytrenowanego modelu

    test_dataset = model_obj.test_data

    """
        Wykres skuteczności od różnych epsilonów
    """

    # Lista epsilonów do przetestowania
    epsilons = [0.0, 0.005, 0.008, 0.01, 0.015, 0.025, 0.05, 0.06]

    # Ocena modelu
    accuracies = evaluate_model_on_epsilons(model_obj.model, test_dataset, epsilons, max_samples=1000)

    # Wykres skuteczności
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies, marker='o', label="Skuteczność modelu")
    plt.title("Skuteczność modelu w zależności od wartości epsilon")
    plt.xlabel("Wartość epsilon")
    plt.ylabel("Skuteczność")
    plt.grid(True)
    plt.legend()
    plt.show()

    """
         Oryginalny vs. zaatakowany i czy rozpoznaje
    """
    # Wygeneruj przykłady adversarialne
    epsilon = 0.01  # Siła ataku

    original, adversarial, true_labels, predicted_labels = generate_adversarial_examples(
        model_obj.model, test_dataset, epsilon, max_samples=10)

    # Wyświetl wyniki
    num_samples = 10  # Liczba przykładów do wizualizacji
    for i in range(num_samples):
        plt.figure(figsize=(8, 4))

        # Oryginalny obraz
        original_pred = model_obj.model.predict(original[i][None, ...])
        predicted_label_original = np.argmax(original_pred)
        recognition_status = "Poprawne" if true_labels[i] == predicted_label_original else "Niepoprawne"
        plt.subplot(1, 2, 1)
        plt.title(f"Oryginał: {label_to_name(true_labels[i])} ({recognition_status})")
        plt.imshow(original[i])
        plt.axis('off')

        # Obraz z zakłóceniami
        recognition_status_adv = "Poprawne" if true_labels[i] == predicted_labels[i] else "Niepoprawne"
        plt.subplot(1, 2, 2)
        plt.title(f"Zakłócony (ε={epsilon}): {label_to_name(predicted_labels[i])} ({recognition_status_adv})")
        plt.imshow(adversarial[i])
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    """
    Oryginalny vs. 3 różne epsilony na jednym 
    """

    # random_image, label = random.choice(test_dataset)

    # random_image = tf.convert_to_tensor(random_image)  # Konwersja na tensor
    #
    # random_image = random_image[0]
    #
    # plt.imshow(random_image)
    # plt.show()

    epsilons2 = [0.01, 0.1, 0.25]
    generate_and_plot_adversarial_examples(model_obj.model, test_dataset, epsilons2)