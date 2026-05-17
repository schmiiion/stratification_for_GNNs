import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_dataset_results(csv_filename="gnn_benchmark_results_full.csv"):
    # 1. Load Data
    try:
        df = pd.read_csv(csv_filename, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found.")
        return

    # --- Dataset Filtering Logic ---
    AVG_OVER_DSs = False
    all_datasets = ["Cora", "CiteSeer",] # "PubMed"]
    single_dataset = 'PubMed'

    if AVG_OVER_DSs:
        df = df[df['Dataset'].isin(all_datasets)]
        title_context = f"Multiple Datasets ({', '.join(all_datasets)})"
    else:
        df = df[df['Dataset'] == single_dataset]
        title_context = f"{single_dataset} Dataset"

    # --- Split Filtering Logic ---
    USE_ALL_SPLITS = False
    target_splits_list = [[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.1, 0.2, 0.3]]
    target_splits_list = [[0.6, 0.2, 0.2], [0.5, 0.3, 0.2]] #[0.1, 0.2, 0.3]]

    if not USE_ALL_SPLITS:
        formatted_splits = ["-".join(map(str, s)) for s in target_splits_list]
        df = df[df['Split_Ratio'].isin(formatted_splits)]
        split_context = f"Splits: {', '.join(formatted_splits)}"
    else:
        split_context = "All Available Splits"

    title_context += f" | {split_context}"

    if df.empty:
        print(f"No data found for the specified configuration.")
        return

    # 2. Calculate AGGREGATED Statistics
    stats_aggregated = df.groupby(['Split_Ratio'])['Test_Accuracy'].agg(
        Samples='count',
        Mean='mean',
        Std_Dev='std',
        Median='median',
        IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
    ).round(4).reset_index()

    # Calculate Sample Sizes for the Headings
    total_samples = len(df)
    num_models = df['Model'].nunique()
    samples_per_model = total_samples // num_models if num_models > 0 else 0

    # 3. Setup the Figure Layout (Now 3 Rows)
    sns.set_theme(style="whitegrid")
    # Increased height to 14 to comfortably fit the new row
    fig = plt.figure(figsize=(16, 14))

    # ADDED: Main dataset name prominently at the top
    fig.suptitle(f'Test Accuracy Analysis: {title_context}', fontsize=18, weight='bold', y=0.98)

    # 3 rows, 2 columns. The table row (index 2) gets a smaller height ratio
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 2.5, 1])

    ax_point_detailed = fig.add_subplot(gs[0, 0])
    ax_point_agg = fig.add_subplot(gs[0, 1])
    ax_box_detailed = fig.add_subplot(gs[1, 0])
    ax_box_agg = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[2, :])

    # --- ROW 1: Mean & Standard Deviation (Point Plots - No Dynamite Bars) ---
    # Detailed Point Plot
    sns.pointplot(
        data=df, x='Split_Ratio', y='Test_Accuracy', hue='Model',
        errorbar='sd', capsize=0.1, dodge=0.4, join=False, ax=ax_point_detailed
    )
    ax_point_detailed.set_title(f'Mean & Std Dev (By Model, N={samples_per_model})')
    ax_point_detailed.set_ylabel('Test Accuracy')
    ax_point_detailed.set_xlabel('')  # Hide x-label to reduce clutter between rows

    # Aggregated Point Plot
    sns.pointplot(
        data=df, x='Split_Ratio', y='Test_Accuracy', color='black',
        errorbar='sd', capsize=0.1, join=False, ax=ax_point_agg
    )
    ax_point_agg.set_title(f'Aggregated Mean & Std Dev (Total N={total_samples})')
    ax_point_agg.set_ylabel('')
    ax_point_agg.set_xlabel('')

    # --- ROW 2: Median & IQR (Boxplots - Means Removed) ---
    # Detailed Boxplot
    sns.boxplot(
        data=df, x='Split_Ratio', y='Test_Accuracy', hue='Model', ax=ax_box_detailed
    )
    # Ensure the legend doesn't duplicate if it's already in the top plot
    if ax_box_detailed.get_legend() is not None:
        ax_box_detailed.get_legend().remove()
    ax_box_detailed.set_title('Median & IQR Dispersion (By Model)')
    ax_box_detailed.set_ylabel('Test Accuracy')

    # Aggregated Boxplot
    sns.boxplot(
        data=df, x='Split_Ratio', y='Test_Accuracy', color='lightgray', ax=ax_box_agg
    )
    ax_box_agg.set_title('Aggregated Median & IQR Dispersion')
    ax_box_agg.set_ylabel('')

    # --- ROW 3: Aggregated Statistics Table ---
    ax_table.axis('off')

    table = ax_table.table(
        cellText=stats_aggregated.values,
        colLabels=stats_aggregated.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')

    ax_table.set_title("Aggregated Summary Statistics", pad=10, weight='bold', fontsize=14)

    # Adjust layout to make room for the suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    plot_dataset_results()