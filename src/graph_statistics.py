import torch
from torch_geometric.utils import degree, to_undirected, homophily
import powerlaw
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from factories.dataset_factory import DatasetFactory


def analyze_graph_properties():
    results = []

    dataset_names = list(DatasetFactory._REGISTRY.keys())

    print(f"\n{'Dataset':<15} | {'Homophily (h)':<15} | {'Alpha (α)':<10} | {'KS Dist (D)':<10} | {'Status'}")
    print("-" * 75)

    for name in dataset_names:
        try:
            dataset, input_dim, output_dim, data = DatasetFactory.get_dataset(
                name=name, root_dir='data/', device='cpu'
            )

            # Calculate Edge Homophily
            h_ratio = homophily(data.edge_index, data.y, method='edge')

            # Ensure the graph is undirected for degree analysis
            edge_index = to_undirected(data.edge_index)

            # Calculate degrees
            deg = degree(edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
            deg = deg[deg > 0]  # Power law requires k > 0

            # Fit Power Law
            fit = powerlaw.Fit(deg, discrete=True, verbose=False)

            # Determine if it's "Scale-Free" (Typical range 2.0 < α < 3.0)
            alpha = fit.power_law.alpha
            ks_d = fit.power_law.D
            status = "Scale-Free" if 2.0 <= alpha <= 3.0 else "Non-Standard"

            results.append({
                'Dataset': name,
                'Homophily': h_ratio.item() if isinstance(h_ratio, torch.Tensor) else h_ratio,
                'Alpha': alpha,
                'KS_D': ks_d,
                'Status': status
            })

            print(f"{name:<15} | {h_ratio:<15.4f} | {alpha:<10.2f} | {ks_d:<10.3f} | {status}")
            print("-" * 35)

        except Exception as e:
            print(f"Error loading/processing {name}: {e}")

    # --- Visualization ---
    if results:
        df = pd.DataFrame(results)

        # Optionally save this table to a CSV for your records
        df.to_csv("Dataset_Structural_Properties.csv", index=False)

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Color palette: highlighting the scale-free "sweet spot"
        plot = sns.barplot(data=df, x='Alpha', y='Dataset', hue='Status', palette='magma')

        # Add theoretical bounds
        plt.axvline(x=2.0, color='green', linestyle='--', label='Lower Bound (α=2)')
        plt.axvline(x=3.0, color='red', linestyle='--', label='Upper Bound (α=3)')

        plt.title("Degree Distribution Analysis: Power Law Exponent (α)")
        plt.xlabel("Alpha (α) Value")
        plt.legend(title="Network Type", loc='lower right')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    analyze_graph_properties()