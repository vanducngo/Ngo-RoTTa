import pandas as pd

# File paths
# train_input_file = '/Users/admin/Working/Data/vinbigdata-chest-xray/validate.csv'
# train_output_file = '/Users/admin/Working/Data/vinbigdata-chest-xray/validate-5-class.csv'
train_input_file = '/Users/admin/Working/Data/MixData/PadChestPruning/padchest_labels.csv'
train_output_file = '/Users/admin/Working/Data/MixData/PadChestPruning/validate_filtered.csv'

def execute(input, output):
    # List of diseases to keep
    selected_diseases = [
        'Cardiomegaly', 
        'Consolidation', 
        'Pleural Effusion', 
        'Pneumothorax', 
        'Atelectasis'
    ]

    # Read the CSV file
    df = pd.read_csv(input)

    # Filter rows where at least one disease in selected_diseases has a value of 1.0
    df_filtered = df[df[selected_diseases].eq(1.0).any(axis=1)]

    # Keep only the 'image_id' column and the selected disease columns
    columns_to_keep = ['image_id'] + selected_diseases
    df_filtered = df_filtered[columns_to_keep]

    # Save to new CSV file
    df_filtered.to_csv(output, index=False)

    print(f"File has been filtered and saved to {output}")


if __name__ == "__main__":
    execute(train_input_file, train_output_file)
