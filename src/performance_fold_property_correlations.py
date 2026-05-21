import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


def analyze_fold_property_correlations(
        accuracy_file="logs/20260517_143150_StratifiedSamplingBaseline_Cora_ep1000_plus_fold_id.csv",
        stats_file="logs/20260517_143150_FoldStatistics.csv"
):
    # ==========================================
    # 1. LOAD AND PREPARE ACCURACY DATA
    # ==========================================
    print("Loading and aggregating accuracy data...")
    df_acc = pd.read_csv(accuracy_file)

    # Aggregate Test_Accuracy over the 10 Init_Seeds using Median
    df_acc_agg = df_acc.groupby(['Dataset', 'StratificationType', 'Fold_Seed', 'Fold', 'Model'])[
        'Test_Accuracy'].median().reset_index()
    df_acc_agg.rename(columns={'Test_Accuracy': 'Median_Accuracy'}, inplace=True)

    # ==========================================
    # 2. LOAD AND PREPARE FOLD STATISTICS DATA
    # ==========================================
    print("Loading fold statistics data...")
    # Read normally - Pandas will use the 'StratificationType' header naturally!
    df_stats = pd.read_csv(stats_file)

    # Pivot the properties into columns
    df_stats_pivot = df_stats.pivot_table(
        index=['Dataset', 'StratificationType', 'Fold_Seed', 'Fold'],
        columns='Property',
        values='EMD'
    ).reset_index()

    # ==========================================
    # 3. MERGE DATASETS
    # ==========================================
    print("Merging datasets on Fold_Seed and Fold...")

    # Ensure keys are strictly integers before merging to prevent ValueError
    df_acc_agg['Fold_Seed'] = df_acc_agg['Fold_Seed'].astype(int)
    df_acc_agg['Fold'] = df_acc_agg['Fold'].astype(int)

    df_stats_pivot['Fold_Seed'] = df_stats_pivot['Fold_Seed'].astype(int)
    df_stats_pivot['Fold'] = df_stats_pivot['Fold'].astype(int)

    # Inner join ensures we only analyze data where we have both accuracy and stats
    df_merged = pd.merge(
        df_acc_agg,
        df_stats_pivot,
        on=['Dataset', 'StratificationType', 'Fold_Seed', 'Fold'],
        how='inner'
    )

    if df_merged.empty:
        print("Error: Merged dataframe is empty. Check if Stratification column strings match exactly.")
        return

    # Extract the list of dynamic properties (e.g., 'Degree', 'PageRank', etc.)
    properties = df_stats['Property'].unique()
    models = df_merged['Model'].unique()

    # ==========================================
    # 4. CALCULATE CORRELATIONS
    # ==========================================
    print("Calculating Spearman correlations...")
    correlation_results = []

    for model in models:
        df_model = df_merged[df_merged['Model'] == model]

        for prop in properties:
            # Drop NaNs just in case a property failed to calculate for a specific fold
            clean_data = df_model[['Median_Accuracy', prop]].dropna()

            if len(clean_data) > 2:
                # Calculate Spearman Rank Correlation and p-value
                rho, p_val = stats.spearmanr(clean_data[prop], clean_data['Median_Accuracy'])
                correlation_results.append({
                    'Model': model,
                    'Property': prop,
                    'Spearman_Rho': rho,
                    'P_Value': p_val
                })

    df_corr = pd.DataFrame(correlation_results)

    # Pivot for the heatmap: Rows = Properties, Columns = Models
    corr_matrix = df_corr.pivot(index='Property', columns='Model', values='Spearman_Rho')

    # ==========================================
    # 5. VISUALIZATIONS
    # ==========================================
    sns.set_theme(style="whitegrid")

    # --- PLOT A: Correlation Heatmap ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=.5,
        cbar_kws={'label': 'Spearman Correlation ($\\rho$)'}
    )
    plt.title("Correlation: Graph Property Distances (EMD) vs Median Test Accuracy", pad=20, fontsize=14, weight='bold')
    plt.ylabel("Graph Property")
    plt.xlabel("Model Architecture")
    plt.tight_layout()
    plt.show()

    # --- PLOT B: Detailed Scatter Plots ---
    # Create a grid of scatter plots with regression lines.
    # Rows = Properties, Cols = Models.
    df_melted = df_merged.melt(
        id_vars=['Fold_Seed', 'Fold', 'Model', 'Median_Accuracy'],
        value_vars=properties,
        var_name='Property',
        value_name='EMD_Distance'
    )

    g = sns.lmplot(
        data=df_melted,
        x='EMD_Distance',
        y='Median_Accuracy',
        col='Model',
        row='Property',
        hue='Model',
        sharex='row',  # Share X axis per property since ranges differ
        sharey=False,  # Do not share Y, allow each model to scale accurately
        height=3.5,
        aspect=1.2,
        scatter_kws={'alpha': 0.6, 'edgecolor': 'k'},
        line_kws={'linewidth': 2}
    )

    g.figure.suptitle("Detailed Impact of Fold Topology on Model Accuracy", y=1.02, fontsize=18, weight='bold')
    g.set_axis_labels("Distance (EMD)", "Median Accuracy")
    plt.tight_layout()
    plt.show()

    # Print numerical results to console
    print("\n--- STATISTICAL RESULTS SUMMARY ---")
    # Sort to show the strongest correlations (positive or negative) first
    df_corr['Abs_Rho'] = df_corr['Spearman_Rho'].abs()
    print(df_corr.sort_values(by=['Model', 'Abs_Rho'], ascending=[True, False]).drop(columns=['Abs_Rho']).to_string(
        index=False))


if __name__ == "__main__":
    analyze_fold_property_correlations()