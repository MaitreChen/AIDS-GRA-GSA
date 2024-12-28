from datetime import datetime
import os

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

from model.gcn import GCNet
from model.gat import GATNet
from model.granet import GRANet


def load_aids_dataset(root='data/TUDataset', batch_size=32, split_ratio=(0.8, 0.1, 0.1)):
    dataset = TUDataset(root=root, name='AIDS')

    # Get infos
    num_graphs = len(dataset)
    num_features = dataset.num_features
    num_classes = dataset.num_classes

    print(f"Number of graphs: {num_graphs}")
    print(f"Number of features per graph: {num_features}")
    print(f"Number of classes: {num_classes}")

    # Split Dataset
    train_ratio, val_ratio, test_ratio = split_ratio
    train_dataset, test_dataset = train_test_split(dataset, test_size=1 - train_ratio, random_state=42)
    val_dataset, test_dataset = train_test_split(test_dataset, test_size=test_ratio / (test_ratio + val_ratio),
                                                 random_state=42)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    print(f"Number of training graphs: {len(train_dataset)}")
    print(f"Number of validation graphs: {len(val_dataset)}")
    print(f"Number of test graphs: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, num_features, num_classes


def train(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in loader:
        data = data.cuda()
        optimizer.zero_grad()
        output, _ = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        _, predicted = output.max(dim=1)
        correct += (predicted == data.y).sum().item()
        total += data.y.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy


def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for data in loader:
            data = data.cuda()
            output, _ = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()

            _, predicted = output.max(dim=1)
            all_labels.extend(data.y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    conf_matrix = confusion_matrix(all_labels, all_preds)

    return total_loss / len(loader), accuracy, precision, recall, f1, conf_matrix


def train_and_evaluate(model, train_loader, val_loader, optimizer, epochs, timestamp_dir):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    conf_matrices = []

    best_val_accuracy = 0.0
    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=0)

    for epoch in range(epochs):
        train_loss, train_accuracy = train(model, train_loader)
        val_loss, val_accuracy, val_precision, val_recall, val_f1, conf_matrix = evaluate(model, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1)
        conf_matrices.append(conf_matrix)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        print(f'Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
        print('=' * 50)

        scheduler.step()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            model_save_path = os.path.join(timestamp_dir, f'{val_accuracy:.4f}_best_model.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f'Saving model at epoch {epoch + 1} with validation accuracy: {val_accuracy * 100:.4f}%')

    return (train_losses, val_losses, train_accuracies, val_accuracies,
            val_precisions, val_recalls, val_f1_scores, conf_matrices)


def plot_confusion_matrix(cm, labels, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(save_path, 'cm.png'), bbox_inches='tight', dpi=400)
    plt.show()


if __name__ == '__main__':
    lr = 3e-4
    bs = 32
    epochs = 500

    checkpoint_dir = 'checkpoints'
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    timestamp_dir = os.path.join(checkpoint_dir, timestamp)
    os.makedirs(timestamp_dir, exist_ok=True)

    # Get dataloader
    train_loader, val_loader, test_loader, num_features, num_classes = load_aids_dataset(
        root='data/TUDataset', batch_size=bs, split_ratio=(0.8, 0.1, 0.1)
    )

    # Get Model
    model = GATNet(in_channels=num_features, out_channels=num_classes).cuda()
    # model = GCNet(in_channels=num_features, out_channels=num_classes).cuda()
    # model = GRANet(in_channels=num_features, out_channels=num_classes, hidden_channels=128,
    #                num_heads=8).cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train and Evaluate
    train_losses, val_losses, train_accuracies, val_accuracies, val_precisions, val_recalls, val_f1_scores, conf_matrices = train_and_evaluate(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs,
        timestamp_dir)

    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix = evaluate(model,
                                                                                                test_loader)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')
    print(f'Test Confusion Matrix:\n{test_conf_matrix}')

    plot_confusion_matrix(test_conf_matrix,
                          labels=[str(i) for i in range(num_classes)],
                          save_path=timestamp_dir)

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_losses, label='Train Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(timestamp_dir, 'loss.png'), bbox_inches='tight', dpi=400)
    plt.show()

    # Plot acc curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), train_accuracies, label='Train Accuracy')
    plt.plot(range(epochs), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(timestamp_dir, 'acc.png'), bbox_inches='tight', dpi=400)
    plt.show()

    # Plot f1-score curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), val_f1_scores, label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(os.path.join(timestamp_dir, 'f1.png'), bbox_inches='tight', dpi=400)
    plt.show()
