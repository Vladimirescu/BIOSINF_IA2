import torch
import torch.nn as nn
from tqdm import tqdm
import os


def iterate(model, 
            dataloader, 
            optimizer, 
            loss_fn, 
            is_training=True, 
            device='cuda'):
    """
    Single-iteration function, either for training or for testing.
    Iterates over dataloader and computes avg_loss and accuracy, in a classification setting.

    :param model: nn.Module, neural network
    :param dataloader: torch.utils.data.DataLoader object to iterate over
    :param optimizer: torch.optim object, used to update the model if is_training=True
    :param loss_fn: cost function
    :param is_training: bool, whether this iteration should compute gradients and update the model
    """
    # Set the model to training mode if it's a training step, otherwise to evaluation mode
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for i, batch in tqdm(enumerate(dataloader)):
        sentences, labels, lengths = batch
        sentences, labels, lengths = sentences.to(device), labels.to(device), lengths.to(device)

        # Forward pass
        outputs = model(sentences, lengths)

        # Calculate loss if it's a training step
        loss = loss_fn(outputs, labels) 

        # Backward pass and optimization if it's a training step
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Metrics calculation (accuracy for simplicity, adjust as needed)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Cumulative loss for reporting
        total_loss += loss.item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / (i + 1)
    accuracy = correct_predictions / total_samples if total_samples > 0 else None

    return avg_loss, accuracy


def train_loop(model, 
               train_loader, 
               optimizer, 
               loss, 
               epochs, 
               test_loader=None, 
               device="cpu", 
               folder_path=None, 
               file_name=None, 
               print_frequency=1):
    """
    Train loop functionality, for iterating, saving and (optional) loading pretrained model.
    """
    train_losses = []
    test_losses = []
    
    best_loss = torch.inf
    model = model.to(device)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join(folder_path, file_name)
                )["state_dict"]
            )
        except:
            print("Couldn't load model")
    
    for e in range(1, epochs + 1):
        train_loss, train_acc = iterate(model, train_loader, optimizer, loss, device=device)
        train_losses.append(train_loss)
        
        if test_loader is not None:
            test_loss, test_acc = iterate(model, test_loader, optimizer, loss, is_training=False, device=device)
            test_losses.append(test_loss)
        else:
            test_loss, test_acc = None, None
        
        if test_loss < best_loss:
            best_loss = test_loss
    
            checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(folder_path, file_name))
        
        if e % print_frequency == 0:
            print(f"Epoch {e}/{epochs}: train_loss={train_loss} train_acc={train_acc} test_loss={test_loss} test_acc={test_acc}")

    return train_losses, test_losses


def iterate_forecast(model, 
                    dataloader, 
                    optimizer, 
                    loss_fn, 
                    teacher_forcing_ratio,
                    is_training=True, 
                    device='cuda'):
    """
    Single-iteration function, either for training or for testing.
    Iterates over dataloader and computes avg_loss, in a forecasting setting: we have a source and a target

    :param model: nn.Module, neural network
    :param dataloader: torch.utils.data.DataLoader object to iterate over
    :param optimizer: torch.optim object, used to update the model if is_training=True
    :param loss_fn: cost function
    :param is_training: bool, whether this iteration should compute gradients and update the model
    """
    # Set the model to training mode if it's a training step, otherwise to evaluation mode
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_samples = 0

    for i, batch in tqdm(enumerate(dataloader)):
        source, target = batch
        source, target = source.to(device), target.to(device)

        # Forward pass
        outputs = model(source, target, teacher_forcing_ratio)

        # Calculate loss if it's a training step
        loss = loss_fn(outputs, target) 

        # Backward pass and optimization if it's a training step
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Cumulative loss for reporting
        total_loss += loss.item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / (i + 1)

    return avg_loss


def train_loop_forecast(model, 
                       train_loader, 
                       optimizer, 
                       loss, 
                       epochs, 
                       teacher_forcing_ratio,
                       test_loader=None, 
                       device="cpu", 
                       folder_path=None, 
                       file_name=None, 
                       print_frequency=1):
    """
    Train loop functionality, for iterating, saving and (optional) loading pretrained model.
    """
    train_losses = []
    test_losses = []
    
    best_loss = torch.inf
    model = model.to(device)
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        try:
            model.load_state_dict(
                torch.load(
                    os.path.join(folder_path, file_name)
                )["state_dict"]
            )
        except:
            print("Couldn't load model")
    
    for e in range(1, epochs + 1):
        train_loss = iterate_forecast(model, train_loader, optimizer, loss, teacher_forcing_ratio, device=device)
        train_losses.append(train_loss)
        
        if test_loader is not None:
            """Teacher forcing ratio is 0 for evaluation - we are only predicting given the previous prediction"""
            test_loss = iterate_forecast(model, test_loader, optimizer, loss, 0, is_training=False, device=device)
            test_losses.append(test_loss)
        else:
            test_loss = None
        
        if test_loss < best_loss:
            best_loss = test_loss
    
            checkpoint = {'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(folder_path, file_name))
        
        if e % print_frequency == 0:
            print(f"Epoch {e}/{epochs}: train_loss={train_loss} test_loss={test_loss}")

    return train_losses, test_losses


def get_prediction_targets(model, dataloader, device="cpu"):
    model.eval()
    
    true_targets = torch.empty(len(dataloader.dataset), dtype=torch.long, device=device)
    predicted = torch.empty(len(dataloader.dataset), dtype=torch.long, device=device)

    last_index = 0
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

        true_targets[last_index: last_index + inputs.shape[0]] = labels
        predicted[last_index: last_index + inputs.shape[0]] = preds

        last_index += inputs.shape[0]
        
    return true_targets.cpu().numpy(), predicted.cpu().numpy()