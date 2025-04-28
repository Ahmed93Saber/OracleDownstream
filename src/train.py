import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import logging
import copy
from sklearn.metrics import accuracy_score, roc_auc_score

from src.dataset import ClinicalDataset
from src.utils import calculate_sensitivity_specificity
from src.models import SimpleNN

def run_epoch(model, loader, criterion, optimizer, device, is_training: bool):
    """
    Runs a single training or evaluation epoch.
    """
    model.train() if is_training else model.eval()
    total_loss = 0
    y_true, y_pred = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(torch.sigmoid(outputs).detach().cpu().numpy())

    pred_labels = (np.array(y_pred) > 0.5).astype(int)
    sensitivity, specificity = calculate_sensitivity_specificity(y_true, pred_labels)
    metrics = {
        "accuracy": accuracy_score(y_true, pred_labels),
        "auc": roc_auc_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity
    }
    predictions_and_labels = {"labels": np.array(y_true), "predictions": np.array(y_pred)}
    return total_loss / len(loader), metrics, predictions_and_labels

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Trains the model for one fold and returns the best model state based on validation AUC.
    """
    best_val_auc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss, train_metrics, _ = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        val_loss, val_metrics, val_ys = run_epoch(model, val_loader, criterion, optimizer, device, is_training=False)

        if epoch % 5 == 0:
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, AUC: {train_metrics['auc']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}"
            )

        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_model_state = copy.deepcopy(model.state_dict())

    return best_model_state, best_val_auc

def train_and_evaluate_model(
    dataloaders, feature_columns, test_df, exclude_columns,
    num_epochs=30, hidden_size=64, num_layers=3, learning_rate=0.001, batch_size=32
):
    """
    Trains and evaluates the model using cross-validation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN(input_size=len(feature_columns), hidden_size=hidden_size, num_layer=num_layers).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    test_metrics = {"loss": [], "accuracy": [], "auc": [], "sensitivity": [], "specificity": []}
    outputs_and_predictions = {"labels": [], "predictions": []}

    val_auc_list = []

    for fold, loaders in dataloaders.items():
        logging.info(f"Training Fold {fold + 1}")
        train_loader, val_loader = loaders['train'], loaders['val']

        best_model_state, best_val_auc = train_one_fold(
            model, train_loader, val_loader, criterion, optimizer, device, num_epochs
        )

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_metrics, test_ys = evaluate_on_test_set(
            model, test_df, exclude_columns, criterion, device, batch_size, test_metrics
        )

        outputs_and_predictions["labels"].append(test_ys["labels"])
        outputs_and_predictions["predictions"].append(test_ys["predictions"])

        val_auc_list.append(best_val_auc)

    np.save("predictions/outputs_and_predictions.npy", outputs_and_predictions)

    mean_test_metrics = {metric: np.mean(values) for metric, values in test_metrics.items()}
    std_test_metrics = {metric: np.std(values) for metric, values in test_metrics.items()}
    logging.info(
        f"Mean Test Metrics Across All Folds: "
        f"Loss: {mean_test_metrics['loss']:.4f} "
        f"Acc: {mean_test_metrics['accuracy']:.4f}±{std_test_metrics['accuracy']:.4f}, "
        f"AUC: {mean_test_metrics['auc']:.4f}±{std_test_metrics['auc']:.4f}, "
        f"Sensitivity: {mean_test_metrics['sensitivity']:.4f}±{std_test_metrics['sensitivity']:.4f}, "
        f"Specificity: {mean_test_metrics['specificity']:.4f}±{std_test_metrics['specificity']:.4f}"
    )


def evaluate_on_test_set(model, test_df, exclude_columns, criterion, device, batch_size, test_metrics):
    """
    Evaluates the model on the test set and updates metrics.
    """
    test_loader = DataLoader(
        ClinicalDataset(test_df, columns_to_drop=exclude_columns),
        batch_size=batch_size, shuffle=False
    )
    model.eval()
    test_loss, test_metrics_fold, test_ys = run_epoch(
        model, test_loader, criterion, None, device, is_training=False
    )


    test_metrics["loss"].append(test_loss)
    test_metrics["accuracy"].append(test_metrics_fold["accuracy"])
    test_metrics["auc"].append(test_metrics_fold["auc"])
    test_metrics["sensitivity"].append(test_metrics_fold["sensitivity"])
    test_metrics["specificity"].append(test_metrics_fold["specificity"])


    return test_metrics, test_ys

