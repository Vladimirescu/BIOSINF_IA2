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

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # Calculate loss if it's a training step
        loss = loss_fn(outputs, labels) 

        # Backward pass and optimization if it's a training step
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_samples += labels.size(0)
        total_loss += loss.item()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(dataloader)

    return avg_loss


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
        train_loss = iterate(model, train_loader, optimizer, loss, device=device)
        train_losses.append(train_loss)
        
        if test_loader:
            with torch.no_grad():
                test_loss = iterate(model, test_loader, optimizer, loss, is_training=False, device=device)
            test_losses.append(test_loss)
        else:
            test_loss, test_acc = None, None
        
        if train_loss < best_loss:
            print(f"Train loss improved from {best_loss} to {train_loss}. Overwriting...")
            best_loss = train_loss
    
            checkpoint = {'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(folder_path, file_name))
        
        if e % print_frequency == 0:
            print(f"Epoch {e}/{epochs}: train_loss={train_loss} test_loss={test_loss}")

    return train_losses, test_losses
