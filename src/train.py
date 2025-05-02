import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import logging
import copy
from sklearn.metrics import accuracy_score, roc_auc_score

from src.dataset import ClinicalDataset
from src.utils import calculate_sensitivity_specificity, log_optuna_metrics
from src.models import SimpleNN, SimpleNNWithBatchNorm

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
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(y_true, pred_labels),
        "auc": roc_auc_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity
    }
    predictions_and_labels = {"labels": np.array(y_true), "predictions": np.array(y_pred)}
    return metrics["loss"], metrics, predictions_and_labels

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Trains the model for one fold and returns the best model state based on validation AUC.
    """
    best_val_auc = 0
    best_model_state = None
    best_val_metrics = {}

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
            best_val_metrics = val_metrics
            best_model_state = copy.deepcopy(model.state_dict())

    return best_model_state, best_val_metrics

def train_and_evaluate_model(
    trial, dataloaders, feature_columns, test_df, exclude_columns,
    num_epochs=30, hidden_size=64, num_layers=3, learning_rate=0.001, batch_size=32,
    model_cls=None, model_kwargs=None,
    dataset_cls=None, dataset_kwargs=None
):
    """
    Trains and evaluates the model using cross-validation.
    model_cls: class of the model to instantiate.
    model_kwargs: dict of kwargs to pass to the model constructor.
    dataset_cls: class of the dataset to instantiate for test set.
    dataset_kwargs: dict of kwargs to pass to the dataset constructor for test set.
    """
    if model_cls is None:
        from src.models import SimpleNN
        model_cls = SimpleNN
    if model_kwargs is None:
        model_kwargs = {"input_size": len(feature_columns), "hidden_size": hidden_size, "num_layer": num_layers}
    if dataset_cls is None:
        from src.dataset import ClinicalDataset
        dataset_cls = ClinicalDataset
    if dataset_kwargs is None:
        dataset_kwargs = {"columns_to_drop": exclude_columns}

    weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()

    test_metrics = {"loss": [], "accuracy": [], "auc": [], "sensitivity": [], "specificity": []}
    val_metrics_folds = {"loss": [], "accuracy": [], "auc": []}
    outputs_and_predictions = {"labels": [], "predictions": []}

    for fold, loaders in dataloaders.items():
        logging.info(f"Training Fold {fold + 1}")
        train_loader, val_loader = loaders['train'], loaders['val']

        # Instantiate a new model and optimizer for each fold (reset parameters)
        model = model_cls(**model_kwargs).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_model_state, val_metrics = train_one_fold(
            model, train_loader, val_loader, criterion, optimizer, device, num_epochs
        )

        val_metrics_folds["loss"].append(val_metrics["loss"])
        val_metrics_folds["accuracy"].append(val_metrics["accuracy"])
        val_metrics_folds["auc"].append(val_metrics["auc"])

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_metrics, test_ys = evaluate_on_test_set(
            model, test_df, exclude_columns, criterion, device, batch_size, test_metrics,
            dataset_cls=dataset_cls, dataset_kwargs=dataset_kwargs
        )

        outputs_and_predictions["labels"].append(test_ys["labels"])
        outputs_and_predictions["predictions"].append(test_ys["predictions"])

    np.save("predictions/outputs_and_predictions.npy", outputs_and_predictions)

    mean_test_metrics = {metric: np.mean(values) for metric, values in test_metrics.items()}
    std_test_metrics = {metric: np.std(values) for metric, values in test_metrics.items()}
    mean_val_metrics = {metric: np.mean(values) for metric, values in val_metrics_folds.items()}

    # log to optuna
    log_optuna_metrics(trial, mean_val_metrics)
    log_optuna_metrics(trial, mean_test_metrics, is_test=True)

    logging.info(
        f"Mean Test Metrics Across All Folds: "
        f"Loss: {mean_test_metrics['loss']:.4f} "
        f"Acc: {mean_test_metrics['accuracy']:.4f}±{std_test_metrics['accuracy']:.4f}, "
        f"AUC: {mean_test_metrics['auc']:.4f}±{std_test_metrics['auc']:.4f}, "
        f"Sensitivity: {mean_test_metrics['sensitivity']:.4f}±{std_test_metrics['sensitivity']:.4f}, "
        f"Specificity: {mean_test_metrics['specificity']:.4f}±{std_test_metrics['specificity']:.4f}"
    )

    return mean_val_metrics

def evaluate_on_test_set(
    model, test_df, exclude_columns, criterion, device, batch_size, test_metrics,
    dataset_cls, dataset_kwargs
):
    """
    Evaluates the model on the test set and updates metrics.
    """
    test_loader = DataLoader(
        dataset_cls(test_df, **dataset_kwargs),
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
