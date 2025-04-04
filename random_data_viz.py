'''
Contains functions to randomly generate & visualize linearly separable and 
non-linearly separable clusters
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

SEED = 920481945
np.random.seed(SEED)

def draw_linearly_separable_cluster(line_parameters, num_points_to_plot, margin):
    slope, intercept = line_parameters
    
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
    )
    mesh = np.c_[xx.ravel(), yy.ravel()]
    region_set = mesh.copy()
    # y = mx + c -> mx - y + c = 0
    # Points who outputs value > 0 lies above this line and vice versa for points below line
    # Instead of comparing with 0, using margin so that points are visibly separated
    points_wrt_line = region_set[:, 0] * slope + intercept - region_set[:, 1]
    # Finding index of points above and below the line
    points_above_line_index = np.where(points_wrt_line > margin)[0]
    points_below_line_index = np.where(points_wrt_line < -margin)[0]

    # Sampling randomly few points index to plot
    points_to_plot_above_index = np.random.choice(
        points_above_line_index, size=num_points_to_plot, replace=False
    )
    points_to_plot_below_index = np.random.choice(
        points_below_line_index, size=num_points_to_plot, replace=False
    )

    # Getting actual point coordinate from region set
    points_to_plot_above = region_set[points_to_plot_above_index]
    points_to_plot_below = region_set[points_to_plot_below_index]
    points_to_plot = np.concatenate(
        (points_to_plot_above, points_to_plot_below), axis=0
    )

    # Creating class value for the points
    class_value = np.array([0 for _ in range(num_points_to_plot)] + \
        [1 for _ in range(num_points_to_plot)], dtype=np.int64)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Plot 1 -> Showing only clusters
    axs[0].scatter(
        points_to_plot[:, 0], points_to_plot[:, 1], c=class_value, 
        cmap=ListedColormap(['blue', 'green']), edgecolor='k', marker='o'
    )
    axs[0].set_title("Linearly Separable Cluster")
    axs[0].grid(False)
    axs[0].set_xticks([])  # Remove x-axis ticks
    axs[0].set_yticks([])  # Remove y-axis ticks
    
    # Plot 2 -> Showing clusters with decision boundary
    axs[1].scatter(
        points_to_plot[:, 0], points_to_plot[:, 1], c=class_value, 
        cmap=ListedColormap(['blue', 'green']), edgecolor='k', marker='o'
    )
    # Get xmin such that plot remains within the same bound y = mx+c -> x = (y-c)m
    line_x_min = (y_min - intercept) / slope
    line_x_max = (y_max - intercept) / slope
    line_xx = np.linspace(line_x_min, line_x_max, 500)
    # Draw line
    axs[1].plot(line_xx, slope * line_xx + intercept, color='red')
    axs[1].set_title("Linearly Separable Cluster with Decision Boundary")
    axs[1].grid(False)
    axs[1].set_xticks([])  # Remove x-axis ticks
    axs[1].set_yticks([])  # Remove y-axis ticks
    
    plt.tight_layout()
    plt.show()
    
def draw_non_linearly_separable_cluster(
        parabola_params, num_points_to_plot, margin
    ):
    a, b, c = parabola_params
    
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    mesh = np.c_[xx.ravel(), yy.ravel()]
    region_set = mesh.copy()
    
    # y = ax^2 + bx + c
    # Points who outputs value > 0 lies above this curve and vice versa for points below curve
    points_wrt_line = a * region_set[:, 0]**2 + b * region_set[:, 0] + c - region_set[:, 1]
    # Finding index of points above and below the curve
    points_above_line_index = np.where(points_wrt_line > margin)[0]
    points_below_line_index = np.where((points_wrt_line < -margin))[0]
    
    # Sampling randomly few points index to plot
    points_to_plot_above_index = np.random.choice(
        points_above_line_index, size=num_points_to_plot, replace=False
    )
    points_to_plot_below_index = np.random.choice(
        points_below_line_index, size=num_points_to_plot, replace=False
    )
    
    # Getting actual point coordinate from region set
    points_to_plot_above = region_set[points_to_plot_above_index]
    points_to_plot_below = region_set[points_to_plot_below_index]
    points_to_plot = np.concatenate(
        (points_to_plot_above, points_to_plot_below), axis=0
    )
    
    class_value = np.array([0 for _ in range(num_points_to_plot)] + \
        [1 for _ in range(num_points_to_plot)], dtype=np.int64)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Plot 1 -> Showing only clusters
    axs[0].scatter(
        points_to_plot[:, 0], points_to_plot[:, 1], c=class_value, 
        cmap=ListedColormap(['blue', 'green']), edgecolor='k', marker='o'
    )
    axs[0].set_title("Non-linearly Separable Cluster")
    axs[0].grid(False)
    axs[0].set_xticks([])  # Remove x-axis ticks
    axs[0].set_yticks([])  # Remove y-axis ticks
    
    # Plot 2 -> Showing clusters with decision boundary
    axs[1].scatter(
        points_to_plot[:, 0], points_to_plot[:, 1], c=class_value, 
        cmap=ListedColormap(['blue', 'green']), edgecolor='k', marker='o'
    )
    # Get xmin such that plot remains within the same bound y = ax^2 + bx + c -> x = (-b + sqrt(b^2 - 4ac)) / 2a
    # We will use y_max both times coz parabola has lowest point in middle and highest on both sides
    line_x_min = (-b - np.sqrt( b**2 - 4*a*(c-y_max))) / (2*a)
    line_x_max = (-b + np.sqrt( b**2 - 4*a*(c-y_max))) / (2*a)
    # import pdb;pdb.set_trace()
    line_xx = np.linspace(line_x_min, line_x_max, 500)
    # Draw line
    axs[1].plot(line_xx, a * line_xx**2 + b * line_xx + c, color='red')
    axs[1].set_title("Non-linearly Separable Cluster with Decision Boundary")
    axs[1].grid(False)
    axs[1].set_xticks([])  # Remove x-axis ticks
    axs[1].set_yticks([])  # Remove y-axis ticks
    
    plt.tight_layout()
    plt.show()
    
def main():
    random_line = [3, 2]    # slope, intercept
    margin = 4.0
    num_points_to_plot = 100
    draw_linearly_separable_cluster(random_line, num_points_to_plot, margin)
    
    random_parabola = [0.5, 1.5, -5.0]  # a, b, c
    margin = 3.5
    num_points_to_plot = 100
    draw_non_linearly_separable_cluster(random_parabola, num_points_to_plot, margin)
    
if __name__ == '__main__':
    main()


