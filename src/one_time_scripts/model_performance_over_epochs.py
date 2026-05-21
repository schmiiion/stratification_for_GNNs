import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_accuracy_with_snapshots(csv_filename="logs/20260517_143150_StratifiedSamplingBaseline.csv"):
    # 1. Load Data
    try:
        df = pd.read_csv(csv_filename, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found.")
        return

    if 'Dataset' not in df.columns:
        df.columns = ['Dataset', 'Stratification', 'Fold', 'Model', 'Seed', 'Epoch', 'Test_Accuracy']

    # Filter for Cora dataset
    target_dataset = 'chameleon'
    df = df[df['Dataset'] == target_dataset]

    if df.empty:
        print(f"No data found for the Dataset: {target_dataset}.")
        return

    # 2. Pre-calculate Mean and Std Dev for accurate text annotations
    df_stats = df.groupby(['Model', 'Epoch'])['Test_Accuracy'].agg(['mean', 'std']).reset_index()

    # Calculate Sample Sizes for the Heading
    total_samples = len(df)
    num_models = df['Model'].nunique()
    samples_per_model = total_samples // num_models if num_models > 0 else 0

    # 3. Setup Colors explicitly so lines and text labels match perfectly
    unique_models = df['Model'].unique()
    palette = sns.color_palette("tab10", len(unique_models))
    model_colors = dict(zip(unique_models, palette))

    # 4. Initialize Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 10))

    # Main lineplot with shaded standard deviation error bands
    sns.lineplot(
        data=df,
        x='Epoch',
        y='Test_Accuracy',
        hue='Model',
        style='Model',
        markers=True,
        dashes=False,
        errorbar='sd',
        palette=model_colors,
        linewidth=2.5,
        err_kws={'alpha': 0.12}
    )

    # 5. Snapshots Annotation Logic (200, 400, 600, 800, 1000)
    target_epochs = [200, 400, 600, 800, 1000]

    # Custom vertical adjustments to prevent overlapping text callouts
    y_offsets = {
        'MLP': 0.006,  # MLP sits safely isolated below, nudge slightly up
        'GCN': 0.006,  # Upper middle
        'GAT': 0.016,  # High top
        'SAGE': -0.008,  # Lower middle
        'MixHop': -0.018  # Bottom floor
    }

    for _, row in df_stats.iterrows():
        model = row['Model']
        epoch = row['Epoch']

        if epoch in target_epochs:
            mean_val = row['mean']
            std_val = row['std']

            # Fetch the dedicated offset for this model architecture
            offset = y_offsets.get(model, 0.0)

            # Create a string label: Mean top, ±Std Dev bottom
            label_text = f"{mean_val:.3f}\n±{std_val:.3f}"

            plt.text(
                x=epoch,
                y=mean_val + offset,
                s=label_text,
                color=model_colors[model],
                fontsize=8.5,
                weight='bold',
                ha='center',
                va='center',
                # Semi-transparent white box background ensures text readability over grid lines
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=1.5)
            )

    # Plot Titles and Configuration
    plt.title(
        f"{target_dataset} Dataset: Test Accuracy Evolution Over Epochs\n"
        f"(Total N={total_samples} | N={samples_per_model} samples per model)",
        fontsize=16,
        weight='bold',
        pad=15
    )
    plt.xlabel("Epochs", fontsize=13, weight='bold')
    plt.ylabel("Test Accuracy", fontsize=13, weight='bold')

    # Place ticks cleanly on target intervals
    plt.xticks(target_epochs)

    plt.legend(
        title="Model",
        title_fontsize='12',
        fontsize='11',
        bbox_to_anchor=(1.01, 1),
        loc='upper left'
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_accuracy_with_snapshots()