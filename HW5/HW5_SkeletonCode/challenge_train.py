'''
Challenge - Train
    Trains a neural network to classify images
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_challenge.py
'''

import utils
from challenge_data import get_train_val_test_loaders
from challenge_model import Challenge
from utils import *
from challenge_cnn import Cnn_2_2
# import torch_directml
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, recall_score
import torch.nn.functional as F


def _train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch of data from `data_loader`
    Use `optimizer` to optimize the specified `criterion`
    """
    # TODO: complete the training step, see train_cnn.py
    for i, (X, y) in enumerate(data_loader):
        # clear parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()


def _evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats):
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in tr_loader:
            output = model(X)
            predicted = torch.argmax(output, dim=1)  # Get class predictions
            y_true.append(y.cpu().numpy())
            y_pred.append(torch.softmax(output, dim=1).cpu().numpy())
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())

        train_loss = np.mean(running_loss)
        train_acc = correct / total
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Metrics
        train_f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average='weighted')  # Weighted average for multi-class
        train_auroc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted')  # 'ovr' for one-vs-rest
        train_auprc = average_precision_score(y_true, y_pred, average='weighted')
        train_recall = recall_score(y_true, np.argmax(y_pred, axis=1), average='weighted')

    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in val_loader:
            output = model(X)
            predicted = torch.argmax(output, dim=1)
            y_true.append(y.cpu().numpy())
            y_pred.append(torch.softmax(output, dim=1).cpu().numpy())
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())

        val_loss = np.mean(running_loss)
        val_acc = correct / total
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        # Metrics
        val_f1 = f1_score(y_true, np.argmax(y_pred, axis=1), average='weighted')  # Weighted average for multi-class
        val_auroc = roc_auc_score(y_true, y_pred, multi_class='ovr', average='weighted')  # 'ovr' for one-vs-rest
        val_auprc = average_precision_score(y_true, y_pred, average='weighted')
        val_recall = recall_score(y_true, np.argmax(y_pred, axis=1), average='weighted')

    # Log the metrics
    stats.append(
        [train_loss, val_loss, train_acc, val_acc,
         train_f1, val_f1, train_auroc, val_auroc,
         train_auprc, val_auprc, train_recall, val_recall]
    )
    # stats.append(
    #     [val_acc, val_loss, train_acc, train_loss,
    #      train_f1, train_auroc, train_auprc, val_f1,
    #      val_auroc, val_auprc])

    utils.log_cnn_training_c(epoch, stats)
    utils.update_cnn_training_plot_c(axes, epoch, stats)


def __evaluate_epoch(axes, tr_loader, val_loader, model, criterion, epoch, stats):
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in tr_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        train_loss = np.mean(running_loss)
        train_acc = correct / total
    with torch.no_grad():
        y_true, y_pred = [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in val_loader:
            output = model(X)
            predicted = predictions(output.data)
            y_true.append(y)
            y_pred.append(predicted)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            running_loss.append(criterion(output, y).item())
        val_loss = np.mean(running_loss)
        val_acc = correct / total
    stats.append([val_acc, val_loss, train_acc, train_loss])
    utils.log_cnn_training(epoch, stats)
    utils.update_cnn_training_plot(axes, epoch, stats)

def main():
    device = utils.get_device()
    # device = torch_directml.device(torch_directml.default_device())

    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        num_classes=config('challenge.num_classes'))
    # tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
    #     num_classes=10)

    # TODO: define model, loss function, and optimizer
    model = Challenge().to(device)
    # model = Cnn_2_2().to(device)

    criterion = torch.nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=5*config('challenge.learning_rate'),
    #     weight_decay=1e-4 # L2
    # )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5*config('challenge.learning_rate'),
        weight_decay=1e-4 # L2 regularization
    ) # you may use config('challenge.learning_rate')

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    #

    # Attempts to restore the latest checkpoint if exists
    print('Loading challenge...')
    model, start_epoch, stats = restore_checkpoint(model,
        config('challenge.checkpoint'))

    axes = utils.make_cnn_training_plot_c()

    # Evaluate model
    _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, start_epoch, stats)

    # Loop over the entire dataset multiple times
    for epoch in range(start_epoch, config('challenge.num_epochs')):
        # Train model
        _train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        _evaluate_epoch(axes, tr_loader, va_loader, model, criterion, epoch+1, stats)

        # Save model parameters
        save_checkpoint(model, epoch+1, config('challenge.checkpoint'), stats)

        # lr scheduler
        scheduler.step()

    print('Finished Training')

    # Keep plot open
    utils.hold_training_plot()

if __name__ == '__main__':
    utils.make_checkpoint_dir('./checkpoints/challenge/')
    main()
