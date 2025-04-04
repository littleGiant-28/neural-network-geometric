import os
import glob

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set random seed for reproducibility
SEED = 42931856
torch.manual_seed(SEED)
np.random.seed(SEED)

weight_viz_save_dir = "weight_visualization"
train_viz_save_dir = "train_visualization"
os.makedirs(weight_viz_save_dir, exist_ok=True)
os.makedirs(train_viz_save_dir, exist_ok=True)

def create_dataset(number_samples, noise, random_state):
    # noise controls the standard deviation of the dataset
    # more noise -> more spread out the data is
    # Generate moon dataset with 2 classes
    X, y = make_moons(
        n_samples=number_samples, 
        noise=noise,
        random_state=random_state
    )
    return X, y

def visualize_dataset(X, y):
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.scatter(
        X[:, 0], X[:, 1], c=y, 
        cmap=ListedColormap(['blue', 'green']), marker='o', edgecolor='k'
    )
    plt.savefig("dataset.png")
    
def preprocess_dataset(X, y):
    # Standardize the dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    return X, y
    
def create_dataloader(X, y, batch_size):
    # Preprocess the dataset
    X, y = preprocess_dataset(X, y)
    # Convert X and y to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    # Create a TensorDataset
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    # Create a DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader

def create_model(input_size, hidden_size, output_size):
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size)
    )
    return model

def train_model(model, dataloader, num_epochs, learning_rate, full_data):
    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X, y = full_data
    # Train the model
    for epoch in tqdm(range(num_epochs)):
        for X_batch, y_batch in dataloader:
            # Forward pass
            y_pred = model(X_batch)
            # import pdb;pdb.set_trace()
            loss = criterion(y_pred, y_batch)
            # import pdb;pdb.set_trace()
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
        # add description to tqdm
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
        visualize_boundary(model, X, y, save_path=f"{train_viz_save_dir}/epoch_{epoch + 1}.png")
    return model

@torch.no_grad()
def visualize_boundary(model, X, y, save_path):
    model.eval()
    X, y = preprocess_dataset(X, y)
    
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    input_mesh = np.c_[xx.ravel(), yy.ravel()]
    input_mesh_tensor = torch.from_numpy(input_mesh).float()
    
    plt.figure(figsize=(10, 6))
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    model_output = model(input_mesh_tensor).numpy()
    heatmap_values_0 = model_output[:, 0].reshape(xx.shape)
    heatmap_values_1 = model_output[:, 1].reshape(xx.shape)

    contourf_0 = plt.contourf(xx, yy, heatmap_values_0, cmap='Reds', alpha=0.5, levels=100)
    contourf_1 = plt.contourf(xx, yy, heatmap_values_1, cmap='Blues', alpha=0.5, levels=100)

    # Draw the decision boundary line
    decision_boundary = model_output[:, 0] - model_output[:, 1]
    decision_boundary = decision_boundary.reshape(xx.shape)
    plt.contour(xx, yy, decision_boundary, levels=[0], colors='purple', linewidths=2)

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['blue', 'green']), marker='o', edgecolor='k')

    plt.title('Weights Visualization')
    plt.colorbar(contourf_0, label='Output of Neuron 1')
    plt.colorbar(contourf_1, label='Output of Neuron 2')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

@torch.no_grad()
def visualize_weights(model, X, y):
    model.eval()
    X, y = preprocess_dataset(X, y)
    
    # First layer visualization
    
    first_layer_weights = model[0].weight.data
    first_layer_biases = model[0].bias.data
    
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    input_mesh = np.c_[xx.ravel(), yy.ravel()]
    input_mesh_tensor = torch.from_numpy(input_mesh).float()
    
    first_layer_output = input_mesh_tensor @ first_layer_weights.T + first_layer_biases
    first_layer_output_activation = F.relu(first_layer_output)
    
    num_neurons = first_layer_output_activation.shape[1]
    
    def draw_weight_activation_heatmap(ax, weight_vector, activation_output, title, save_path=''):
        contour = ax.contourf(xx, yy, activation_output, cmap='coolwarm', alpha=0.5, levels=50)
        plt.colorbar(contour, ax=ax, label='Activation Output')
        
        weight_vector = weight_vector / np.linalg.norm(weight_vector)
        ax.arrow(0, 0, weight_vector[0], weight_vector[1], color='red', width=0.02, head_width=0.1)
        ax.set_title(title)
        
    fig1, axes1 = plt.subplots(4, 2, figsize=(12, 16))  # For outputs without activation
    fig2, axes2 = plt.subplots(4, 2, figsize=(12, 16))  # For outputs with activation
    axes1 = axes1.flatten()
    axes2 = axes2.flatten()
        
    for i in range(num_neurons):
        heatmap_values = first_layer_output[:, i].numpy().reshape(xx.shape)
        draw_weight_activation_heatmap(
            axes1[i], first_layer_weights[i].numpy(), heatmap_values, 
            title=f"Weight-{i} boundary without activation",
            save_path=f"{weight_viz_save_dir}/weight_{i}_boundary_without_activation.png"
        )
        
        heatmap_values = first_layer_output_activation[:, i].numpy().reshape(xx.shape)
        draw_weight_activation_heatmap(
            axes2[i], first_layer_weights[i].numpy(), heatmap_values, 
            title=f"Weight-{i} boundary with activation",
            save_path=f"{weight_viz_save_dir}/weight_{i}_boundary_with_activation.png"
        )
        
    fig1.suptitle("First Layer Weight Boundaries Without Activation")
    fig2.suptitle("First Layer Weight Boundaries With Activation")
    plt.tight_layout()
    fig1.savefig(f"{weight_viz_save_dir}/first_layer_without_activation.png")
    fig2.savefig(f"{weight_viz_save_dir}/first_layer_with_activation.png")
    plt.close(fig1)
    plt.close(fig2)
    
    # Second layer visualization
    visualize_boundary(model, X, y, save_path=f"{weight_viz_save_dir}/second_layer_output.png")
    
def create_gif(train_viz_dir):
    images = glob.glob(f"{train_viz_dir}/*.png")
    output_gif_path = f"{train_viz_dir}/training_visualization.gif"
    print(f"Saving training visualization gif to {output_gif_path}")
    fps = 10
    # sort the images based on epoch number
    # epoch_10.png -> 10.png -> 10
    images = sorted(images, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    frames = [Image.open(img).resize((600, 400)) for img in images]
    frames[0].save(
        output_gif_path, save_all=True, append_images=frames[1:], 
        duration=int(1000 / fps), loop=0, optimize=True
    )
    
def main():
    # Create a dataset
    X, y = create_dataset(number_samples=1000, noise=0.2, random_state=SEED)
    visualize_dataset(X, y)
    # Create a DataLoader
    dataloader = create_dataloader(X.copy(), y.copy(), batch_size=BATCH_SIZE)
    # Create a model
    input_size = 2
    hidden_size = 8
    output_size = 2
    model = create_model(input_size, hidden_size, output_size)
    # Train the model
    num_epochs = 100
    learning_rate = 0.01
    model = train_model(model, dataloader, num_epochs, learning_rate, (X, y))
    visualize_weights(model, X, y)
    create_gif(train_viz_save_dir)
    # Save the model
    # torch.save(model.state_dict(), "model.pth")
    

if __name__ =='__main__':
    main()