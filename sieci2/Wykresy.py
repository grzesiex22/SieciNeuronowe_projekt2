import numpy as np
import os
import matplotlib.pyplot as plt


def plot_training_history(history, title="Model Training History", filename_prefix="", folder="plots"):
    """
    Rysowanie wykresów strat, dokładności i MSE oraz zapisywanie ich jako pliki w określonym folderze.
    """
    # Sprawdzamy, czy folder istnieje, jeśli nie to go tworzymy
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Sprawdzamy, czy self.history to obiekt z atrybutem history
    if hasattr(history, 'history'):
        history = history.history

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']

    # Dodanie MSE (jeśli jest dostępne w historii treningu)
    mse = history.get('mse', None)
    val_mse = history.get('val_mse', None)

    epochs = range(1, len(acc) + 1)

    # Wykres dokładności
    plt.figure()
    plt.plot(epochs, acc, label='Training Accuracy', marker='o', color='blue')
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    accuracy_plot_path = os.path.join(folder, f'{filename_prefix}_accuracy.png')
    plt.savefig(accuracy_plot_path)  # Zapis wykresu jako plik PNG w folderze
    # plt.show()

    # Wykres strat (loss)
    plt.figure()
    plt.plot(epochs, loss, label='Training Loss', marker='o', color='blue')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    loss_plot_path = os.path.join(folder, f'{filename_prefix}_loss.png')
    plt.savefig(loss_plot_path)  # Zapis wykresu jako plik PNG w folderze
    # plt.show()

    # Wykres MSE (jeśli istnieje)
    if mse is not None:
        plt.figure()
        plt.plot(epochs, mse, label='Training MSE', marker='o', color='blue')
        plt.plot(epochs, val_mse, label='Validation MSE', marker='o', color='red')
        plt.title('Training and Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        mse_plot_path = os.path.join(folder, f'{filename_prefix}_mse.png')
        plt.savefig(mse_plot_path)  # Zapis wykresu jako plik PNG w folderze
        # plt.show()

    print(f"All plots saved in '{folder}' folder.")


def plot_additional_training_history(history, title="Model Training History", filename_prefix="", folder="plots"):
    """
    Rysowanie wykresów strat, dokładności i MSE oraz zapisywanie ich jako pliki w określonym folderze.
    """
    # Sprawdzamy, czy folder istnieje, jeśli nie to go tworzymy
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Sprawdzamy, czy self.history to obiekt z atrybutem history
    if hasattr(history, 'history'):
        history = history.history

    f1 = history['f1_score']
    val_f1 = history['val_f1_score']
    recall = history['recall']
    val_recall = history['val_recall']
    precision = history['precision']
    val_precision = history['val_precision']
    auc_roc = history['auc']
    val_auc_roc = history['val_auc']
    lr = history['lr']
    batch_size = history['batch_size']

    epochs = range(1, len(f1) + 1)

    # Wykres f1-score
    plt.figure()
    plt.plot(epochs, f1, label='Training F1-score', marker='o', color='blue')
    plt.plot(epochs, val_f1, label='Validation F1-score', marker='o', color='red')
    plt.title('Training and Validation F1-score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    accuracy_plot_path = os.path.join(folder, f'{filename_prefix}_f1.png')
    plt.savefig(accuracy_plot_path)  # Zapis wykresu jako plik PNG w folderze
    # plt.show()

    # Wykres recall
    plt.figure()
    plt.plot(epochs, recall, label='Training Recall', marker='o', color='blue')
    plt.plot(epochs, val_recall, label='Validation Recall', marker='o', color='red')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    accuracy_plot_path = os.path.join(folder, f'{filename_prefix}_recall.png')
    plt.savefig(accuracy_plot_path)  # Zapis wykresu jako plik PNG w folderze
    # plt.show()

    # Wykres precision
    plt.figure()
    plt.plot(epochs, precision, label='Training Precision', marker='o', color='blue')
    plt.plot(epochs, val_precision, label='Validation Precision', marker='o', color='red')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    loss_plot_path = os.path.join(folder, f'{filename_prefix}_precision.png')
    plt.savefig(loss_plot_path)  # Zapis wykresu jako plik PNG w folderze
    # plt.show()

    # Wykres auc_roc
    plt.figure()
    plt.plot(epochs, auc_roc, label='Training AUC-ROC', marker='o', color='blue')
    plt.plot(epochs, val_auc_roc, label='Validation AUC-ROC', marker='o', color='red')
    plt.title('Training and Validation AUC-ROC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC-ROC')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    mse_plot_path = os.path.join(folder, f'{filename_prefix}_auc_roc.png')
    plt.savefig(mse_plot_path)  # Zapis wykresu jako plik PNG w folderze
    # plt.show()

    # Wykres lr
    plt.figure()
    plt.plot(epochs, lr, marker='o', color='blue')
    plt.title('Learning rate')
    plt.xlabel('Epochs')
    plt.ylabel('lr')
    plt.grid(True, linestyle='--', alpha=0.7)
    mse_plot_path = os.path.join(folder, f'{filename_prefix}_lr.png')
    plt.savefig(mse_plot_path)  # Zapis wykresu jako plik PNG w folderze
    # plt.show()

    # Wykres batch_size
    plt.figure()
    plt.plot(epochs, batch_size, marker='o', color='blue')
    plt.title('Batch size')
    plt.xlabel('Epochs')
    plt.ylabel('Batch size')
    plt.grid(True, linestyle='--', alpha=0.7)
    mse_plot_path = os.path.join(folder, f'{filename_prefix}_batch_size.png')
    plt.savefig(mse_plot_path)  # Zapis wykresu jako plik PNG w folderze
    # plt.show()
    print(f"All plots saved in '{folder}' folder.")


def plot_weights_average(weights_history, layer_names, title="Model Training History", filename_prefix="", folder="plots"):
    """
    Rysuje średnie wartości wag dla podanych warstw jako słupki pionowe wraz z odchyleniem standardowym.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    for name in layer_names:
        weights = weights_history[name]  # Lista wag dla każdej epoki
        mean_weights = [np.mean(w.flatten()) for w in weights]  # Średnia wagi po spłaszczeniu
        std_weights = [np.std(w.flatten()) for w in weights]  # Odchylenie standardowe po spłaszczeniu

        plt.figure()
        epochs = range(1, len(weights) + 1)  # Numer epoki (1, 2, ..., n)
        plt.bar(epochs, mean_weights, yerr=std_weights, capsize=5, alpha=0.7, color='blue')
        plt.xlabel('Epoki')
        plt.ylabel('Średnia wartość wag (+ odchylenie standardowe)')
        plt.title(f'Zmiana wag w warstwie: {name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plot_path = os.path.join(folder, f'{filename_prefix}_{name}_weights.png')
        plt.savefig(plot_path)  # Zapis wykresu jako plik PNG w folderze

    plt.show()

def save_all_metrics(y_true, y_pred, y_pred_prob, folder="metrics", filename="all_metrics.txt"):
    # Sprawdzamy, czy folder istnieje, jeśli nie to go tworzymy
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Tworzymy pełną ścieżkę do pliku
    file_path = os.path.join(folder, filename)

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    loss = log_loss(y_true, y_pred_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    mse = mean_squared_error(y_true, y_pred)

    # Wyświetlanie wyników na ekranie
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    print(f"ROC-AUC: {auc_roc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Log Loss: {loss}")
    print(f"MCC: {mcc}")
    print(f"Precision-Recall AUC: {pr_auc}")
    print(f"Mean Squared Error (MSE): {mse}")

    # Zapis do pliku
    with open(file_path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"ROC-AUC: {auc_roc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Log Loss: {loss}\n")
        f.write(f"MCC: {mcc}\n")
        f.write(f"Precision-Recall AUC: {pr_auc}\n")
        f.write(f"Mean Squared Error (MSE): {mse}\n")

    print(f"All metrics saved to {filename}")


