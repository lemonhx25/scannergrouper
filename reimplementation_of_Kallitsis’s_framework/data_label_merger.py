import pandas as pd
import glob
import os


def load_labels(label_directory):
    """

    Load and merge all label files.

    Parameters.
    - label_directory: The directory where the label files are located.

    Returns: label_directory: the directory where the label files are located.
    - labels_df: the merged label DataFrame.

    """
    all_files = glob.glob(f"{label_directory}/*.csv")
    list_labels = []

    for file in all_files:
        df = pd.read_csv(file)
        list_labels.append(df)

    labels_df = pd.concat(list_labels, ignore_index=True)

    # Check if there is one IP corresponding to multiple labels
    duplicate_labels = labels_df[labels_df.duplicated(subset=['IP'], keep=False)]
    if not duplicate_labels.empty:
        print("Warning: Some IPs have multiple labels associated with them.")
        grouped_duplicates = duplicate_labels.groupby('IP')['Label'].apply(list).reset_index()
        print(grouped_duplicates)

        # Choose a strategy for handling tags, such as selecting the first or merging tags into a single string
        labels_df = labels_df.drop_duplicates(subset=['IP'], keep='first')

    return labels_df


def load_and_merge_data(data_directory):
    """
    Loads and merges all data files.

    Parameters.
    - data_directory: Directory containing all data files.

    Returns: data_df: the merged data DataFrame.
    - data_df: the merged data DataFrame.
    """
    all_files = glob.glob(f"{data_directory}/*.csv")
    print('all_files',all_files)
    list_data = []

    for file in all_files:
        df = pd.read_csv(file)
        list_data.append(df)

    data_df = pd.concat(list_data, ignore_index=True)
    return data_df


def merge_data_with_labels(data_df, labels_df):
    """
    Merges labels with the dataset and removes data with empty labels.

    Parameters.
    - data_df: original dataset DataFrame.
    - labels_df: label DataFrame.

    Returns: merged_df
    - merged_df: the merged and cleaned DataFrame.
    """
    merged_df = pd.merge(data_df, labels_df, left_on='Scanner IP', right_on='IP', how='left')
    print('merged_df',merged_df)
    
    merged_df=merged_df.fillna('unknown')

    merged_df.drop(columns=['IP', 'Time'], inplace=True, errors='ignore')

    return merged_df


def save_merged_data(merged_df, output_directory):
    """
    Saves the merged dataset.

    Parameters.
    - merged_df: The merged DataFrame.
    - output_directory: The output directory to save the file.
    """
    os.makedirs(output_directory, exist_ok=True)
    merged_data_path = os.path.join(output_directory, 'selfdeploy25_merged_dataset_with_labels_all.csv')
    merged_df.to_csv(merged_data_path, index=False)
    print(f'Merged dataset saved to {merged_data_path}')


def print_label_distribution(merged_df):
    """
    Prints the number of rows of data corresponding to each label in the merged dataset.

    Parameters.
    - merged_df: The merged DataFrame.
    """
    label_distribution = merged_df['Label'].value_counts()
    print("\nLabel distribution in the merged dataset:")
    print(label_distribution)


def main():
    # Define paths for data and labels
    data_directory = 'dataset/processed_csv_selfdeploy25/all'
    label_directory = 'label/selfdeploy' 

    # Load and merge all data files
    data_df = load_and_merge_data(data_directory)
    print(f'Number of rows in the data_df: {data_df.shape[0]}')

    # Load all label files
    labels_df = load_labels(label_directory)
    print(f'Number of rows in the labels_df: {labels_df.shape[0]}')

    # Merge data and tags and clean data
    merged_df = merge_data_with_labels(data_df, labels_df)
    print(f'Number of rows in the merged_df: {merged_df.shape[0]}')

    # Prints the number of rows of data corresponding to each label in the merged data set
    print_label_distribution(merged_df)

    # Save merged data
    output_directory = 'output'
    save_merged_data(merged_df, output_directory)


if __name__ == '__main__':
    main()
