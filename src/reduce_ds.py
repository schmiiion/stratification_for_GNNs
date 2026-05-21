import pandas as pd
import os

def reduce_csv_to_cora_ep1000(input_filename="/Users/jonas/Uni/SoSe26/Project/stratification_for_GNNs/src/logs/20260517_143150_StratifiedSamplingBaseline.csv"):
    try:
        # Load the CSV normally. Pandas will automatically use the first row
        # (Dataset, StratificationType, Fold_Seed, etc.) as the column headers.
        df = pd.read_csv(input_filename, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: {input_filename} not found.")
        return

    # Force the 'Epoch' column to be numeric so the == 1000 check works perfectly
    if 'Epoch' in df.columns:
        df['Epoch'] = pd.to_numeric(df['Epoch'], errors='coerce')
    else:
        print("Error: 'Epoch' column not found in the original file headers.")
        return

    # --- Filtering Logic ---
    # Keep only rows where 'Dataset' is 'Cora' AND 'Epoch' is 1000
    filtered_df = df[df['Epoch'] == 1000]

    if filtered_df.empty:
        print("No matching rows found for Cora at Epoch 1000. Check your data.")
        return

    # --- File Saving Logic ---
    base_name, extension = os.path.splitext(input_filename)
    output_filename = f"{base_name}_ep1000{extension}"

    # Save the filtered dataframe.
    # index=False prevents writing the row numbers.
    # Because we didn't override the headers, it will automatically write the original ones.
    filtered_df.to_csv(output_filename, index=False)

    print(f"Success! Filtered dataset saved to: {output_filename}")
    print(f"Total rows kept: {len(filtered_df)}")


if __name__ == "__main__":
    # Replace with the actual path to your CSV file
    reduce_csv_to_cora_ep1000()