import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import textwrap

def display_analysis(consistency_df, significant_changes):
    print("\nConsistency Analysis with Short Labels and Most Common Response:")
    print(consistency_df[['Short Question', 'Consistency', 'Most Common Response']])
    
    sd_questions = consistency_df[consistency_df['Most Common Response'] == 'SD']
    sa_questions = consistency_df[consistency_df['Most Common Response'] == 'SA']

    print("\nQuestions with 'Strongly Disagree' (SD) as the most common response:")
    print(sd_questions[['Short Question', 'Question', 'Consistency']])

    print("\nQuestions with 'Strongly Agree' (SA) as the most common response:")
    print(sa_questions[['Short Question', 'Question', 'Consistency']])

    perfect_consistency_df = consistency_df[consistency_df['Consistency'] == 1]
    print("\nQuestions with Perfect Consistency (Consistency = 1):")
    print(perfect_consistency_df[['Short Question', 'Question', 'Most Common Response']])

    average_consistency = consistency_df['Consistency'].mean()
    print(f"\nAverage Consistency: {average_consistency:.4f}")

    consistency_df['Compared to Average'] = consistency_df['Consistency'] - average_consistency
    print("\nConsistency Analysis with Comparison to Average:")
    print(consistency_df[['Short Question', 'Consistency', 'Most Common Response', 'Compared to Average']])

    print("\nQuestions with Above-Average Consistency:")
    print(consistency_df[consistency_df['Compared to Average'] > 0][['Short Question', 'Question', 'Consistency']])

    print("\nQuestions with Below-Average Consistency:")
    print(consistency_df[consistency_df['Compared to Average'] < 0][['Short Question', 'Question', 'Consistency']])

    consistency_df_sorted = consistency_df.sort_values('Consistency')
    print("\n5 Questions with the Worst Consistency:")
    worst_consistency = consistency_df_sorted.head(5)
    print(worst_consistency[['Short Question', 'Question', 'Consistency', 'Most Common Response']])

    worst_consistency_value = consistency_df_sorted['Consistency'].min()
    print(f"\nWorst Consistency Value: {worst_consistency_value:.4f}")

    significant_changes_df = pd.DataFrame(significant_changes)
    print("\nSignificant Changes in Responses:")
    print(significant_changes_df)

def plot_consistency(consistency_df):
    cmap = plt.colormaps['viridis']
    norm = mcolors.Normalize(vmin=consistency_df['Consistency'].min(), vmax=consistency_df['Consistency'].max())

    fig, ax = plt.subplots(figsize=(14, 12))
    bars = ax.barh(consistency_df['Short Question'], consistency_df['Consistency'], color=cmap(norm(consistency_df['Consistency'])))

    ax.set_title('Bar Plot of Consistency Values with Color Gradient', fontsize=16)
    ax.set_xlabel('Consistency', fontsize=12)
    ax.set_ylabel('Questions', fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Consistency Level', fontsize=12)

    mean_consistency = consistency_df['Consistency'].mean()
    worst_consistency_value = consistency_df['Consistency'].min()

    ax.axvline(x=mean_consistency, color='b', linestyle='--', linewidth=2, label=f'Average: {mean_consistency:.4f}')
    ax.axvline(x=worst_consistency_value, color='r', linestyle=':', linewidth=2, label=f'Worst: {worst_consistency_value:.4f}')

    ax.invert_yaxis()
    ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()

    for i, v in enumerate(consistency_df['Consistency']):
        ax.text(v, i, f' {v:.3f}', va='center', fontsize=8)

    plt.show()

def plot_political_compass(data):
    df = pd.DataFrame(data)
    mean_x = df['x'].mean()
    mean_y = df['y'].mean()

    plt.figure(figsize=(10, 10))
    plt.scatter(df['x'], df['y'], color='blue', label='Data Points')
    plt.scatter(mean_x, mean_y, color='red', label='Mean Point', zorder=5)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.axvline(0, color='grey', linewidth=0.5)
    plt.xlabel('Economic Left/Right')
    plt.ylabel('Social Libertarian/Authoritarian')
    plt.title('Political Compass Coordinates')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_combined(data):
    df = pd.DataFrame(data)
    mean_x = df['x'].mean()
    mean_y = df['y'].mean()
    
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['x', 'y']])
    
    colors = np.arange(len(data))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: K-Means Clustering
    axs[0].axhline(y=0, color='black', linewidth=2)
    axs[0].axvline(x=0, color='black', linewidth=2)
    for cluster in range(num_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        axs[0].scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster + 1}')
    axs[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', marker='X', label='Cluster Centers')
    axs[0].set_title('K-Means Clustering of PoliticalCompass Coordinates')
    axs[0].set_xlabel('Left-Right')
    axs[0].set_ylabel('Authoritarian-Libertarian')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(-10, 10)
    axs[0].set_ylim(-10, 10)
    axs[0].set_aspect('equal', adjustable='box')

    # Plot 2: Scatter Plot with Colormap
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    scatter = axs[1].scatter(df['x'], df['y'], c=colors, cmap='viridis', marker='o', label='PoliticalCompass Coordinates')
    axs[1].set_title('Scatter Plot of PoliticalCompass Coordinates')
    axs[1].set_xlabel('Left-Right')
    axs[1].set_ylabel('Authoritarian-Libertarian')
    axs[1].axhline(y=0, color='black', linewidth=2)
    axs[1].axvline(x=0, color='black', linewidth=2)
    fig.colorbar(scatter, cax=cax, label='Run')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlim(-10, 10)
    axs[1].set_ylim(-10, 10)
    axs[1].set_aspect('equal', adjustable='box')

    # Plot 3: Scatter plot with mean point
    axs[2].scatter(df['x'], df['y'], color='blue', label='Data Points')
    axs[2].scatter(mean_x, mean_y, color='red', label='Mean Point', zorder=5)
    axs[2].axhline(0, color='grey', linewidth=0.5)
    axs[2].axvline(0, color='grey', linewidth=0.5)
    axs[2].set_xlabel('Economic Left/Right')
    axs[2].set_ylabel('Social Libertarian/Authoritarian')
    axs[2].set_title('Political Compass Coordinates with Mean Point')
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_xlim(-10, 10)
    axs[2].set_ylim(-10, 10)
    axs[2].set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)
    plt.show()

# Analyze 32 runs for bar graph
def plot_32_runs(data):
    # Calculate mean and standard deviation for X and Y
    x_values = [d['x'] for d in data]
    y_values = [d['y'] for d in data]
    mean_x = np.mean(x_values)
    mean_y = np.mean(y_values)
    std_x = np.std(x_values)
    std_y = np.std(y_values)

    # Create a figure with subplots for each dataset
    fig, axs = plt.subplots(nrows=6, ncols=6, figsize=(20, 18), sharex=True, sharey=True)
    fig.suptitle("Variability in Standard Deviation for X and Y Across 32 Runs", fontsize=16)

    # Plot the standard deviations for each dataset
    for i, (ax, dataset) in enumerate(zip(axs.flat, data)):
        if i < 32:  # Ensure we only process 32 datasets
            std_x = dataset["x"] - mean_x
            std_y = dataset["y"] - mean_y
            ax.bar(["X-axis", "Y-axis"], [abs(std_x), abs(std_y)], color=["blue", "orange"])
            ax.set_title(f"Run {dataset['run']}", fontsize=10)
            ax.set_ylim(0, 2)  # Set y-axis range from 0 to 2
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            # Annotate the bars with values
            for j, std_value in enumerate([abs(std_x), abs(std_y)]):
                ax.annotate(f"{std_value:.2f}", xy=(j, std_value), xytext=(j, std_value + 0.1),
                            ha="center", va="bottom", fontsize=8, color="black")

            # Draw a red line to indicate the difference
            ax.plot([0, 1], [abs(std_x), abs(std_y)], color="red", linestyle="--")
            ax.annotate("Difference", xy=(0.5, (abs(std_x) + abs(std_y)) / 2), ha="center", va="bottom", fontsize=8, color="red")
        else:
            ax.axis('off')  # Turn off the unused subplots

    # Remove x-axis labels from inner subplots
    for ax in axs.flat:
        ax.label_outer()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Function to handle specific character replacement
def clean_text(text):
    return text.replace("’", "'")

# Function to create a plot with wrapped text for x-axis labels
def plot_significant_changes_wrapped_labels(significant_changes):
    plt.figure(figsize=(14, 10))
    if len(significant_changes) > 0:
        change_counts = pd.DataFrame(significant_changes)['Question'].value_counts().sort_values(ascending=False)
        unique_counts = sorted(change_counts.unique(), reverse=True)
        color_map = dict(zip(unique_counts, cm.viridis(np.linspace(0, 1, len(unique_counts)))))
        colors = [color_map[count] for count in change_counts]

        bars = change_counts.plot(kind='bar', color=colors)
        plt.title('Number of Significant Changes per Question', fontsize=16)
        plt.xlabel('Question', fontsize=14)
        plt.ylabel('Number of Significant Changes', fontsize=14)
        
        # Wrap text for x-axis labels and handle specific character replacement
        labels = [textwrap.fill(clean_text(label), 25) for label in change_counts.index]
        plt.xticks(range(len(change_counts)), labels, rotation=45, fontsize=8, ha='right')
        plt.yticks(fontsize=12)
        
        for i, count in enumerate(change_counts):
            plt.text(i, count + 0.2, str(count), ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.show()
    else:
        print("No significant changes found.")


# Function to create a plot with wrapped text for x-axis labels
def plot_significant_changes(significant_changes_df):
    cmap = plt.cm.viridis
    change_counts = significant_changes_df['Question'].value_counts().sort_values(ascending=False)
    norm = mcolors.Normalize(vmin=change_counts.min(), vmax=change_counts.max())

    fig, ax = plt.subplots(figsize=(14, 12))
    bars = ax.barh(change_counts.index, change_counts.values, color=cmap(norm(change_counts.values)))

    ax.set_title('Number of Significant Changes per Question', fontsize=16)
    ax.set_xlabel('Number of Significant Changes', fontsize=12)
    ax.set_ylabel('Questions', fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Number of Significant Changes', fontsize=12)

    mean_changes = change_counts.mean()
    worst_changes_value = change_counts.min()

    ax.axvline(x=mean_changes, color='b', linestyle='--', linewidth=2, label=f'Average: {mean_changes:.4f}')
    ax.axvline(x=worst_changes_value, color='r', linestyle=':', linewidth=2, label=f'Worst: {worst_changes_value:.4f}')

    ax.invert_yaxis()
    ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()

    for i, v in enumerate(change_counts):
        ax.text(v, i, f' {v}', va='center', fontsize=8)

    plt.show()

