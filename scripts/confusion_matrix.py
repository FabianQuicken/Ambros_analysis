import numpy as np
import pandas as pd
import glob
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Paths to the prediction and ground truth CSV files
project_path = "E:/Fabi_Setup/In_Soundchamber/behaviors_urine_validation_deepethogram"
path_asoid = f"E:/Fabi_Setup/In_Soundchamber/behaviors_urine_validation_deepethogram/stiched_evaluation_video/*.csv"
path_gt = f"E:/Fabi_Setup/In_Soundchamber/behaviors_urine_validation_deepethogram/stiched_evaluation_video_labels/*.csv"

# Collect file paths
file_list_asoid = glob.glob(path_asoid)
file_list_gt = glob.glob(path_gt)

# Initialize lists to collect all predictions and ground truth values
all_predictions_right = []
all_ground_truth_right = []
all_predictions_left = []
all_ground_truth_left = []
all_predictions_drinking = []
all_ground_truth_drinking = []
all_predictions_rear = []
all_ground_truth_rear = []
all_predictions_groom = []
all_ground_truth_groom = []

# Read and collect data from each pair of prediction and ground truth files
for asoid_file, gt_file in zip(file_list_asoid, file_list_gt):
    # Read the CSV files
    df_asoid = pd.read_csv(asoid_file)
    df_gt = pd.read_csv(gt_file)

    # Extract the arrays
    predictions_right = np.nan_to_num(df_asoid['"stimulusinvestigation'].to_numpy())
    ground_truth_right = df_gt['"stimulusinvestigation'].to_numpy()
    predictions_left = np.nan_to_num(df_asoid["foodinteraction"].to_numpy())
    ground_truth_left = df_gt["foodinteraction"].to_numpy()
    predictions_drink = np.nan_to_num(df_asoid["drinking"].to_numpy())
    ground_truth_drink = df_gt["drinking"].to_numpy()
    predictions_rear = np.nan_to_num(df_asoid['rearing"'].to_numpy())
    ground_truth_rear = df_gt['rearing"'].to_numpy()
    predictions_groom = np.nan_to_num(df_asoid['grooming'].to_numpy())
    ground_truth_groom = df_gt['grooming'].to_numpy()

    # Append the data to the lists
    all_predictions_right.extend(predictions_right)
    all_ground_truth_right.extend(ground_truth_right)
    all_predictions_left.extend(predictions_left)
    all_ground_truth_left.extend(ground_truth_left)
    all_predictions_drinking.extend(predictions_drink)
    all_ground_truth_drinking.extend(ground_truth_drink)
    all_predictions_rear.extend(predictions_rear)
    all_ground_truth_rear.extend(ground_truth_rear)
    all_predictions_groom.extend(predictions_groom)
    all_ground_truth_groom.extend(ground_truth_groom)

# Convert lists to numpy arrays
all_predictions_right = np.array(all_predictions_right)
all_ground_truth_right = np.array(all_ground_truth_right)
all_predictions_left = np.array(all_predictions_left)
all_ground_truth_left = np.array(all_ground_truth_left)
all_predictions_drinking = np.array(all_predictions_drinking)
all_ground_truth_drinking = np.array(all_ground_truth_drinking)
all_predictions_rear = np.array(all_predictions_rear)
all_ground_truth_rear = np.array(all_ground_truth_rear)
all_predictions_groom = np.array(all_predictions_groom)
all_ground_truth_groom = np.array(all_ground_truth_groom)



# Combine the behaviors into a single array for multi-class confusion matrix
all_predictions = 5 * all_predictions_right + 4 * all_predictions_left + 3 * all_predictions_drinking + 2 * all_predictions_rear + all_predictions_groom
all_ground_truth = 5 * all_ground_truth_right + 4 * all_ground_truth_left + 3 * all_ground_truth_drinking + 2 * all_ground_truth_rear + all_ground_truth_groom

# Compute the confusion matrix
conf_matrix = confusion_matrix(all_ground_truth, all_predictions)


# Normalize the confusion matrix by the total number of frames for each class
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]


# Define labels for multi-class confusion matrix
labels = ['background behavior', 'grooming', 'rearing', 'drinking', 'food interaction', 'stimulus investigation']

# Create a custom colormap from black to yellow
cmap = LinearSegmentedColormap.from_list('grey_to_yellow', ['black', 'gold'], N=256)

# Plot the normalized confusion matrix with raw values as annotations
plt.figure(figsize=(10, 8), facecolor='black')
ax = sns.heatmap(conf_matrix_normalized, annot=conf_matrix.astype(int), fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)

# Set the face color of the plot background
ax.set_facecolor('black')

# Set label colors
ax.set_xlabel('predicted', color='white')
ax.set_ylabel('true', color='white')
ax.set_title('A-Soid confusion matrix', color='white')

# Set tick colors
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Set tick label colors
plt.setp(ax.get_xticklabels(), color='white')
plt.setp(ax.get_yticklabels(), color='white')

# Customize the colorbar
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_majorticklabels(), color='white')
cbar.set_label(label="", color='white')

#plt.savefig(f"{project_path}/asoid_confusionmatrix.svg", format='svg', facecolor="black")
plt.show()