"""
Utility functions
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import torch

def config(attr):
    """
    Retrieves the queried attribute value from the config file. Loads the
    config file on first call.
    """
    if not hasattr(config, 'config'):
        with open('config.json') as f:
            config.config = eval(f.read())
    node = config.config
    for part in attr.split('.'):
        node = node[part]
    return node

def get_device():
    """Gets the available device for PyTorch."""
    # return torch.device("cpu")
    def _get_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU with CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using GPU with MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device
    if not hasattr(get_device, 'device'):
        get_device.device = _get_device()
    return get_device.device

def denormalize_image(image):
    """ Rescale the image's color space from (min, max) to (0, 1) """
    ptp = np.max(image, axis=(0,1)) - np.min(image, axis=(0,1))
    return (image - np.min(image, axis=(0,1))) / ptp

def hold_training_plot():
    """
    Keep the program alive to display the training plot
    """
    plt.ioff()
    plt.show()

def log_cnn_training(epoch, stats):
    """
    Logs the validation accuracy and loss to the terminal
    """
    valid_acc, valid_loss, train_acc, train_loss = stats[-1]
    print('Epoch {}'.format(epoch))
    print('\tValidation Loss: {}'.format(valid_loss))
    print('\tValidation Accuracy: {}'.format(valid_acc))
    print('\tTrain Loss: {}'.format(train_loss))
    print('\tTrain Accuracy: {}'.format(train_acc))

def log_cnn_training_c(epoch, stats):
    """
    Logs the validation accuracy and loss to the terminal
    """

    (train_loss, val_loss, train_acc, val_acc,
     train_f1, val_f1, train_auroc, val_auroc,
     train_auprc, val_auprc, train_recall, val_recall) = \
        stats[-1]
    # val_acc, val_loss, train_acc, train_loss, train_f1, train_auroc, train_auprc, val_f1, val_auroc, val_auprc = \
    # stats[-1]

    print(f"Epoch {epoch}:")
    print(f"\tTrain Loss: {train_loss:.4f}")
    print(f"\tTrain Accuracy: {train_acc:.4f}")
    print(f"\tTrain F1: {train_f1:.4f}")
    print(f"\tTrain AUROC: {train_auroc:.4f}")
    print(f"\tTrain AUPRC: {train_auprc:.4f}")
    print(f"\tTrain Recall: {train_recall:.4f}")

    print(f"\tValidation Loss: {val_loss:.4f}")
    print(f"\tValidation Accuracy: {val_acc:.4f}")
    print(f"\tValidation F1: {val_f1:.4f}")
    print(f"\tValidation AUROC: {val_auroc:.4f}")
    print(f"\tValidation AUPRC: {val_auprc:.4f}")
    print(f"\tValidation Recall: {val_recall:.4f}")

    print("---"*8)

def make_cnn_training_plot():
    """
    Runs the setup for an interactive matplotlib graph that logs the loss and
    accuracy
    """
    plt.ion()
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    plt.suptitle('CNN Training')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')

    return axes

def make_cnn_training_plot_c():
    """
    Runs the setup for an interactive matplotlib graph that logs the loss and
    accuracy along with other metrics like F1, AUROC, and AUPRC.
    """
    plt.ion()
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots
    plt.suptitle('CNN Training')

    # Accuracy Plot
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')

    # Loss Plot
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')

    # F1 Score Plot
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')

    # AUROC Plot
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUROC')

    # AUPRC Plot
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('AUPRC')

    # AUPRC Plot
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Recall')

    return axes

def update_cnn_training_plot(axes, epoch, stats):
    """
    Updates the training plot with a new data point for loss and accuracy
    """
    valid_acc = [s[0] for s in stats]
    valid_loss = [s[1] for s in stats]
    train_acc = [s[2] for s in stats]
    train_loss = [s[3] for s in stats]
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), valid_acc,
        linestyle='--', marker='o', color='b')
    axes[0].plot(range(epoch - len(stats) + 1, epoch + 1), train_acc,
        linestyle='--', marker='o', color='r')
    axes[0].legend(['Validation', 'Train'])
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), valid_loss,
        linestyle='--', marker='o', color='b')
    axes[1].plot(range(epoch - len(stats) + 1, epoch + 1), train_loss,
        linestyle='--', marker='o', color='r')
    axes[1].legend(['Validation', 'Train'])
    plt.pause(0.00001)

def update_cnn_training_plot_c(axes, epoch, stats):
    """
    Updates the training plot with a new data point for loss, accuracy, F1 score,
    AUROC, and AUPRC.
    """

    train_loss = [s[0] for s in stats]
    val_loss = [s[1] for s in stats]
    train_acc = [s[2] for s in stats]
    val_acc = [s[3] for s in stats]
    train_f1 = [s[4] for s in stats]
    val_f1 = [s[5] for s in stats]
    train_auroc = [s[6] for s in stats]
    val_auroc = [s[7] for s in stats]
    train_auprc = [s[8] for s in stats]
    val_auprc = [s[9] for s in stats]
    train_recall = [s[10] for s in stats]
    val_recall = [s[11] for s in stats]

    # Accuracy Plot
    axes[0, 0].plot(range(epoch - len(stats) + 1, epoch + 1), val_acc,
        linestyle='--', marker='o', color='b')
    axes[0, 0].plot(range(epoch - len(stats) + 1, epoch + 1), train_acc,
        linestyle='--', marker='o', color='r')
    axes[0, 0].legend(['Validation', 'Train'])

    # Loss Plot
    axes[0, 1].plot(range(epoch - len(stats) + 1, epoch + 1), val_loss,
        linestyle='--', marker='o', color='b')
    axes[0, 1].plot(range(epoch - len(stats) + 1, epoch + 1), train_loss,
        linestyle='--', marker='o', color='r')
    axes[0, 1].legend(['Validation', 'Train'])

    # F1 Score Plot
    axes[1, 0].plot(range(epoch - len(stats) + 1, epoch + 1), val_f1,
        linestyle='--', marker='o', color='b')
    axes[1, 0].plot(range(epoch - len(stats) + 1, epoch + 1), train_f1,
        linestyle='--', marker='o', color='r')
    axes[1, 0].legend(['Validation', 'Train'])

    # AUROC Plot
    axes[1, 1].plot(range(epoch - len(stats) + 1, epoch + 1), val_auroc,
        linestyle='--', marker='o', color='b')
    axes[1, 1].plot(range(epoch - len(stats) + 1, epoch + 1), train_auroc,
        linestyle='--', marker='o', color='r')
    axes[1, 1].legend(['Validation', 'Train'])

    # AUPRC Plot
    axes[2, 0].plot(range(epoch - len(stats) + 1, epoch + 1), val_auprc,
        linestyle='--', marker='o', color='b')
    axes[2, 0].plot(range(epoch - len(stats) + 1, epoch + 1), train_auprc,
        linestyle='--', marker='o', color='r')
    axes[2, 0].legend(['Validation', 'Train'])

    # Recall Plot
    axes[2, 1].plot(range(epoch - len(stats) + 1, epoch + 1), val_recall,
        linestyle='--', marker='o', color='b')
    axes[2, 1].plot(range(epoch - len(stats) + 1, epoch + 1), train_recall,
        linestyle='--', marker='o', color='r')
    axes[2, 1].legend(['Validation', 'Train'])

    plt.pause(0.00001)

def save_cnn_training_plot():
    """
    Saves the training plot to a file
    """
    plt.savefig('cnn_training_plot.png', dpi=300)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_checkpoint_dir(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

def save_checkpoint(model, epoch, checkpoint_dir, stats):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'stats': stats,
    }

    filename = os.path.join(checkpoint_dir,
        'epoch={}.checkpoint.pth.tar'.format(epoch))
    torch.save(state, filename)

def restore_checkpoint(model, checkpoint_dir, cuda=False, force=False,
    pretrain=False):
    """
    If a checkpoint exists, restores the PyTorch model from the checkpoint.
    Returns the model and the current epoch.
    """
    cp_files = [file_ for file_ in os.listdir(checkpoint_dir)
        if file_.startswith('epoch=') and file_.endswith('.checkpoint.pth.tar')]

    if not cp_files:
        print('No saved model parameters found')
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []
    
    # Find latest epoch
    for i in itertools.count(1):
        if 'epoch={}.checkpoint.pth.tar'.format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print("Which epoch to load from? Choose in range [0, {}]."
            .format(epoch), "Enter 0 to train from scratch.")
        print(">> ", end='')
        inp_epoch = int(input())
        if inp_epoch not in range(epoch+1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch+1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(checkpoint_dir,
        'epoch={}.checkpoint.pth.tar'.format(inp_epoch))

    print("Loading from checkpoint {}?".format(filename))
    
    if cuda:
        checkpoint = torch.load(filename, weights_only=False)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, weights_only=False,
            map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint['epoch']
        stats = checkpoint['stats']
        if pretrain:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint['state_dict'])
        print("=> Successfully restored checkpoint (trained for {} epochs)"
            .format(checkpoint['epoch']))
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats

def clear_checkpoint(checkpoint_dir):
    filelist = [ f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar") ]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")

def predictions(logits):
    """
    Given the network output, determines the predicted class index

    Returns:
        the predicted class output as a PyTorch Tensor
    """
    ## SOLUTION
    _, pred = torch.max(logits, 1)
    return pred
    ##
