import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
import textwrap


# Data for political compass plotting
political_compass_data = [
    {'run': 1, 'x': -5.63, 'y': -5.95},
    {'run': 2, 'x': -5.25, 'y': -5.23},
    {'run': 3, 'x': -7.13, 'y': -5.90},
    {'run': 4, 'x': -7.75, 'y': -6.56},
    {'run': 5, 'x': -5.25, 'y': -6.00},
    {'run': 6, 'x': -6.38, 'y': -5.85},
    {'run': 7, 'x': -5.75, 'y': -6.15},
    {'run': 8, 'x': -5.88, 'y': -5.59},
    {'run': 9, 'x': -5.38, 'y': -6.10},
    {'run': 10, 'x': -6.38, 'y': -5.79},
    {'run': 11, 'x': -6.38, 'y': -6.10},
    {'run': 12, 'x': -6.25, 'y': -5.18},
    {'run': 13, 'x': -5.88, 'y': -5.90},
    {'run': 14, 'x': -6.88, 'y': -6.00},
    {'run': 15, 'x': -7.38, 'y': -5.90},
    {'run': 16, 'x': -8.00, 'y': -6.31},
    {'run': 17, 'x': -6.38, 'y': -6.26},
    {'run': 18, 'x': -6.38, 'y': -5.90},
    {'run': 19, 'x': -5.50, 'y': -6.38},
    {'run': 20, 'x': -7.13, 'y': -5.90},
    {'run': 21, 'x': -5.75, 'y': -5.90},
    {'run': 22, 'x': -6.75, 'y': -6.05},
    {'run': 23, 'x': -5.88, 'y': -6.15},
    {'run': 24, 'x': -7.13, 'y': -6.26},
    {'run': 25, 'x': -7.25, 'y': -6.46},
    {'run': 26, 'x': -5.38, 'y': -6.10},
    {'run': 27, 'x': -5.75, 'y': -5.79},
    {'run': 28, 'x': -6.38, 'y': -5.74},
    {'run': 29, 'x': -5.25, 'y': -5.79},
    {'run': 30, 'x': -6.38, 'y': -6.05},
    {'run': 31, 'x': -7.50, 'y': -6.00},
    {'run': 32, 'x': -6.38, 'y': -6.10},
    {'run': 33, 'x': -5.63, 'y': -6.26},
    {'run': 34, 'x': -7.13, 'y': -6.21}
]

# Function to read a file with a specified encoding
def read_file_with_encoding(file_path, encoding='utf-8'):
    try:
        return pd.read_csv(file_path, encoding=encoding, usecols=['Q', 'SD', 'D', 'A', 'SA'])
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='ISO-8859-1', usecols=['Q', 'SD', 'D', 'A', 'SA'])

# Load data from dataset directory
def load_data(dataset_dir):
    all_files = glob.glob(dataset_dir + "*.csv")
    dfs = [read_file_with_encoding(file) for file in all_files]
    return pd.concat(dfs, ignore_index=True)

# Analyze responses for each question
def analyze_responses(combined_df):
    unique_questions = combined_df['Q'].unique()
    question_responses = {question: {'responses': [], 'most_common': None} for question in unique_questions}

    for question in unique_questions:
        responses = combined_df[combined_df['Q'] == question][['SD', 'D', 'A', 'SA']].dropna(how='all')
        if not responses.empty:
            selected_responses = responses.idxmax(axis=1)
            question_responses[question]['responses'] = selected_responses.tolist()
            question_responses[question]['most_common'] = selected_responses.mode().iloc[0]
    
    return question_responses

# Calculate consistency for each question
def calculate_consistency(question_responses):
    consistency = {}
    most_common_responses = {}
    significant_changes = []

    for question, data in question_responses.items():
        responses = data['responses']
        most_common = data['most_common']
        if responses:
            consistency[question] = responses.count(most_common) / len(responses)
            most_common_responses[question] = most_common
            
            # Check for significant changes
            for i in range(1, len(responses)):
                prev_response = responses[i-1]
                current_response = responses[i]
                if (prev_response == 'D' and current_response == 'SA') or (prev_response == 'A' and current_response == 'SD') or (prev_response == 'SD' and current_response == 'A') or (prev_response == 'SA' and current_response == 'D'):
                    significant_changes.append({'Question': question, 'Run': i, 'Previous': prev_response, 'Current': current_response})
        else:
            consistency[question] = 0
            most_common_responses[question] = None
    
    return consistency, most_common_responses, significant_changes

# Prepare a DataFrame for consistency analysis
def prepare_consistency_df(consistency, most_common_responses):
    consistency_df = pd.DataFrame({
        'Question': list(consistency.keys()),
        'Consistency': list(consistency.values()),
        'Most Common Response': [most_common_responses[q] for q in consistency.keys()]
    }).dropna(subset=['Question'])
    
    consistency_df['Short Question'] = ['Q' + str(i+1) for i in range(len(consistency_df))]
    return consistency_df

# Display detailed analysis
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

# Plot consistency analysis
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

# Plot political compass data
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
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: K-Means Clustering
    axs[0].axhline(y=0, color='black', linewidth=2)
    axs[0].axvline(x=0, color='black', linewidth=2)
    for cluster in range(num_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        axs[0].scatter(cluster_data['x'], cluster_data['y'], label=f'Cluster {cluster + 1}')
    axs[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', marker='X', label='Cluster Centers')
    axs[0].set_title('K-Means Clustering of Political Compass Coordinates')
    axs[0].set_xlabel('Left-Right')
    axs[0].set_ylabel('Authoritarian-Libertarian')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xlim(-10, 10)
    axs[0].set_ylim(-10, 10)
    axs[0].set_aspect('equal', adjustable='box')

    # Plot 2: Scatter plot with mean point
    axs[1].scatter(df['x'], df['y'], color='blue', label='Data Points')
    axs[1].scatter(mean_x, mean_y, color='red', label='Mean Point', zorder=5)

    # Add the lines to the correct subplot
    axs[1].axhline(y=0, color='black', linewidth=2)
    axs[1].axvline(x=0, color='black', linewidth=2)

    axs[1].set_xlabel('Economic Left/Right')
    axs[1].set_ylabel('Social Libertarian/Authoritarian')
    axs[1].set_title('Political Compass Coordinates with Mean Point')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xlim(-10, 10)
    axs[1].set_ylim(-10, 10)
    axs[1].set_aspect('equal', adjustable='box')
    plt.tight_layout()
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
    fig, axs = plt.subplots(nrows=2, ncols=17, figsize=(20, 18), sharex=True, sharey=True)
    fig.suptitle("Variability in Standard Deviation for X and Y Across 32 Runs", fontsize=16)

    # Plot the standard deviations for each dataset
    for i, (ax, dataset) in enumerate(zip(axs.flat, data)):
        if i < 40:  # Ensure we only process 32 datasets
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

def main():
    # Directory containing the dataset files
    dataset_dir = '/home/minjilee/Desktop/political_compass/dataset/'

    # Load and analyze data
    combined_df = load_data(dataset_dir)
    print("Combined Political Compass Dataset:")
    print(combined_df.head())

    question_responses = analyze_responses(combined_df)
    consistency, most_common_responses, significant_changes = calculate_consistency(question_responses)
    consistency_df = prepare_consistency_df(consistency, most_common_responses)

    display_analysis(consistency_df, significant_changes)
    plot_consistency(consistency_df)

    # Plot political compass data
    plot_political_compass(political_compass_data)

    # Plot combined figure for second, third, and fourth plots
    plot_combined(political_compass_data)

    # Plot 32 runs for bar graph
    plot_32_runs(political_compass_data)

if __name__ == "__main__":
    main()




# Function to read a file with a specified encoding
def read_file_with_encoding(file_path, encoding='utf-8'):
    try:
        return pd.read_csv(file_path, encoding=encoding, usecols=['Q', 'SD', 'D', 'A', 'SA'])
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='ISO-8859-1', usecols=['Q', 'SD', 'D', 'A', 'SA'])

# Load data from dataset directory
def load_data(dataset_dir):
    all_files = glob.glob(dataset_dir + "*.csv")
    dfs = [read_file_with_encoding(file) for file in all_files]
    return dfs

# Analyze responses for each question
def analyze_responses(dfs):
    changes = []
    for i in range(1, len(dfs)):
        prev_df = dfs[i-1]
        curr_df = dfs[i]
        for question in prev_df['Q'].unique():
            prev_responses = prev_df[prev_df['Q'] == question][['SD', 'D', 'A', 'SA']].dropna(how='all')
            curr_responses = curr_df[curr_df['Q'] == question][['SD', 'D', 'A', 'SA']].dropna(how='all')
            if not prev_responses.empty and not curr_responses.empty:
                prev_max = prev_responses.idxmax(axis=1).iloc[0]
                curr_max = curr_responses.idxmax(axis=1).iloc[0]
                if (prev_max == 'SD' and curr_max in ['A', 'SA']) or (prev_max == 'D' and curr_max in ['A', 'SA']) or \
                   (prev_max == 'A' and curr_max in ['SD', 'D']) or (prev_max == 'SA' and curr_max in ['SD', 'D']):
                    changes.append({'Question': question, 'Run': i+1, 'Previous': prev_max, 'Current': curr_max})
    return pd.DataFrame(changes)

# Directory containing the dataset files
dataset_dir = '/home/minjilee/Desktop/political_compass/dataset/'

# Load and analyze data
dfs = load_data(dataset_dir)
significant_changes = analyze_responses(dfs)

# Display significant changes
print("Significant Changes in Responses:")
print(significant_changes)

# If there are significant changes, display the DataFrame
if not significant_changes.empty:
    print(significant_changes)













import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import textwrap
import unicodedata

def clean_text(text):
    # Normalize Unicode characters
    normalized_text = unicodedata.normalize('NFKD', text)
    
    # Replace various types of quotation marks with standard single quotes
    quotation_marks = [''', ''', '′', '`', '´', ''', ''']
    for mark in quotation_marks:
        normalized_text = normalized_text.replace(mark, "'")
    
    # Remove any remaining non-ASCII characters
    cleaned_text = ''.join(char for char in normalized_text if ord(char) < 128)
    
    return cleaned_text

def wrap_labels(labels, width=25):
    return [textwrap.fill(clean_text(label), width) for label in labels]

def plot_significant_changes_wrapped_labels(significant_changes):
    plt.figure(figsize=(14, 10))
    if not significant_changes.empty:
        change_counts = significant_changes['Question'].value_counts().sort_values(ascending=False)
        unique_counts = sorted(change_counts.unique(), reverse=True)
        color_map = dict(zip(unique_counts, cm.viridis_r(np.linspace(0, 1, len(unique_counts)))))
        colors = [color_map[count] for count in change_counts]
        
        bars = change_counts.plot(kind='bar', color=colors)
        plt.title('Number of Significant Changes per Question', fontsize=16)
        plt.xlabel('Question', fontsize=14)
        plt.ylabel('Number of Significant Changes', fontsize=14)
        
        # Wrap text for x-axis labels and handle specific character replacement
        labels = wrap_labels(change_counts.index)
        plt.xticks(range(len(change_counts)), labels, rotation=45, fontsize=8, ha='right')
        plt.yticks(fontsize=12)
        
        for i, count in enumerate(change_counts):
            plt.text(i, count + 0.2, str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No significant changes found.")

def plot_significant_changes(significant_changes_df):
    if not significant_changes_df.empty:
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
        ax.axvline(x=worst_changes_value, color='r', linestyle=':', linewidth=2, label=f'Minimum: {worst_changes_value:.4f}')
        
        ax.invert_yaxis()
        ax.legend(fontsize=10, loc='lower right')
        
        # Clean and wrap y-axis labels
        labels = wrap_labels(change_counts.index, width=40)
        ax.set_yticklabels(labels)
        
        plt.tight_layout()
        
        for i, v in enumerate(change_counts):
            ax.text(v, i, f' {v}', va='center', fontsize=8)
        
        plt.show()
    else:
        print("No significant changes found.")

# Assume significant_changes is your DataFrame
plot_significant_changes_wrapped_labels(significant_changes)
plot_significant_changes(significant_changes)