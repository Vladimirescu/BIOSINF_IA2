import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import Audio, display, HTML


def create_xor_dataset(num_samples_per_class=100, noise=0.1):
    """
    Create a larger dataset simulating XOR relationship with two classes and two input variables.

    Parameters:
    - num_samples_per_class (int): Number of samples per class in the dataset.
    - noise (float): Amount of noise to add to the dataset.

    Returns:
    - X (numpy array): Input features.
    - y (numpy array): Output labels.
    """
    np.random.seed(42)

    # Define the four centroids
    centroids = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    labels = [0, 1, 1, 0]
    
    # Generate samples around each centroid
    X = np.zeros((num_samples_per_class * len(centroids), 2))
    y = np.zeros((num_samples_per_class * len(centroids),))

    for i, centroid in enumerate(centroids):
        # Generate samples around the centroid with some noise
        X[i*num_samples_per_class:(i+1)*num_samples_per_class, :] = centroid + noise * np.random.randn(num_samples_per_class, 2)
        y[i*num_samples_per_class:(i+1)*num_samples_per_class] = labels[i]

    # Shuffle the dataset
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    return X, y


def plot_decision_boundary(model, dataloader, h=0.02):
    """
    Plot the decision boundary of a trained model using a DataLoader.

    Parameters:
    - model: Trained machine learning model with a predict function.
    - dataloader: DataLoader containing input features and true labels.
    - h: Step size for the mesh grid.

    Returns:
    - None (displays the plot).
    """
    # Assume unique class labels are integers
    unique_classes = np.unique([labels.item() for _, labels in dataloader.dataset])

    cmap_classes = ListedColormap(['#FFAAAA', '#AAAAFF', '#AAFFAA'])  # Background colors for classes

    # Extract points from the DataLoader
    all_points = []
    for inputs, labels in dataloader:
        all_points.append(torch.cat((inputs, labels.view(-1, 1)), dim=1).numpy())

    all_points = np.concatenate(all_points, axis=0)

    # Create a mesh grid
    x_min, x_max = all_points[:, 0].min() - 0.1, all_points[:, 0].max() + 0.1
    y_min, y_max = all_points[:, 1].min() - 0.1, all_points[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the class for each point in the mesh grid
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).detach().numpy()
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)

    # Plot decision boundary with original class colors
    plt.contourf(xx, yy, Z, cmap=cmap_classes, alpha=0.3)

    # Plot points with original class colors
    plt.scatter(all_points[:, 0], all_points[:, 1], c=all_points[:, 2], cmap=cmap_classes,
                edgecolors='k', marker='o', s=30)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary and Class Regions')
    plt.show()


def plot_random_images(image_dataset, rows=1, columns=5):
    """
    Plot random images from an image dataset.
    
    Parameters:
    - image_dataset: torch Dataset object.
    - rows: Number of rows in the plot grid.
    - columns: Number of columns in the plot grid.

    Returns:
    - None (displays the plot).
    """
    # Calculate total number of images to plot
    num_images = rows * columns

    # Get random indices for images
    random_indices = np.random.choice(len(image_dataset), num_images, replace=False)

    # Plot images
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(random_indices):
        image, label = image_dataset[idx]
        image = image.numpy().transpose((1, 2, 0))  # Transpose to (H, W, C) for matplotlib
        plt.subplot(rows, columns, i + 1)
        plt.imshow((image + 1) / 2)  # Rescale to [0, 1]
        plt.title(f"{image_dataset.classes[label]}")
        plt.axis('off')

    plt.show()


def plot_and_listen_random(dataset, classes, class_indices, sr=16000):    
    html_content = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">'
    
    fig, axs = plt.subplots(1, len(classes), figsize=(len(classes)*4, 5))    
    for i, c in enumerate(classes):
        selected_sample = random.sample(class_indices[c], 1)[0]
        
        waveform = dataset[selected_sample][0]  
        axs[i].plot(waveform)
        axs[i].set_title(f'Class: {c}')
        
        audio_player = Audio(waveform, rate=sr)
        html_content += f'<div style="text-align: center;"><strong>{c}</strong><br>{audio_player._repr_html_()}</div>'
        
    html_content += '</div>'
    display(HTML(html_content))
    
    plt.tight_layout()
    plt.show()




