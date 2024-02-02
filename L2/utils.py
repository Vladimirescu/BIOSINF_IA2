import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_toy_signal(batch_size, frequencies, amplitudes, phases, signal_length=256):

    data_1d = torch.zeros(batch_size, len(frequencies), signal_length)

    for i in range(len(frequencies)):
        t = torch.linspace(0, 2 * 3.1415, signal_length)
        channel_signal = amplitudes[i] * torch.sin(2 * 3.1415 * frequencies[i] * t + phases[i]) + torch.randn(signal_length) * amplitudes[i] / 5
        data_1d[0, i, :] = channel_signal

    return data_1d


def plot_channels(signal):
    """
    Plot each channel of the signal in a single plot with different y-axis offsets.
    
    Args:
    - signal (torch.Tensor): Input signal tensor of shape (1, num_channels, time).
    """
    # Get the number of channels and time steps
    num_channels, time = signal.shape[1], signal.shape[2]

    offsets = []
    for i in range(num_channels):
        offset = i * 2.0
        offsets.append(offset)

        plt.plot(range(time), signal[0, i, :] + offset)
    
    plt.xlabel('Samples')
    plt.yticks(offsets, labels=[f"Channel {i}" for i in range(len(offsets))])


def plot_random_images(image_dataset, idx_to_class, rows=1, columns=5):
    """
    Plot random images from an image dataset.
    
    Parameters:
    - image_dataset: torch Dataset object.
    - idx_to_string: dictionary, keys=class index, values=class string
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
        plt.imshow(image)  # Rescale to [0, 1]
        plt.title(f"{idx_to_class[label]}", fontdict={"fontsize": 10})
        plt.axis('off')

    plt.show()


def plot_image_with_heatmap(original_image, heatmap):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot original image
    ax[0].imshow(original_image.permute(1, 2, 0))
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    # Plot heatmap overlay
    ax[1].imshow(heatmap.cpu(), cmap='viridis', alpha=0.6)
    ax[1].imshow(original_image.permute(1, 2, 0), alpha=0.4)
    ax[1].axis('off')
    ax[1].set_title('Heatmap Overlay')

    plt.show()


def plot_image_with_heatmap_overlay(rgb_image, heatmap, alpha=0.6):
    """
    Plot an RGB image with a heatmap overlay.

    Parameters:
    - rgb_image (torch.Tensor): RGB image tensor of shape (C, H, W).
    - heatmap (torch.Tensor): Single-channel heatmap tensor of shape (H, W).
    - alpha (float): Opacity of the heatmap overlay (default is 0.6).
    """
    # Convert the RGB image and heatmap to NumPy arrays
    rgb_image = rgb_image.permute(1, 2, 0).cpu().numpy()

    # Plot the RGB image
    plt.imshow(rgb_image)

    # Plot the heatmap overlay
    plt.imshow(heatmap, cmap='plasma', alpha=alpha)

    # Remove axis ticks and labels
    plt.xticks([])
    plt.yticks([])