import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from pathlib import Path

sys.path.append('/media/data/facil-tc/src/')
from evaluator.micro_quality_evaluator import BiflowEvaluator

def compute_distances(real_histograms, synthetic_histograms, n_features):
    """
    Compute distances between real and synthetic histograms.
    
    Parameters:
        real_histograms (np.ndarray): Real data histograms ((n_features,) + (n_bins,) * ngram_size ).
        synthetic_histograms (np.ndarray): Synthetic data histograms ((n_features,) + (n_bins,) * ngram_size ).
        n_features (int): Number of features.
    
    Returns:
        dict: Distances (JSD, Hellinger, TVD) for each feature.
    """
    jsd = np.zeros(n_features)
    hd = np.zeros(n_features)
    tvd = np.zeros(n_features)
    
    for f in range(n_features):
        real_pdf = real_histograms[f, :]
        synth_pdf = synthetic_histograms[f, :]
        
        jsd[f] = BiflowEvaluator.compute_jsd(real_pdf, synth_pdf)
        hd[f] = BiflowEvaluator.compute_hd(real_pdf, synth_pdf)
        tvd[f] = BiflowEvaluator.compute_tvd(real_pdf, synth_pdf)

    return {
        "JSD": jsd,
        "Hellinger": hd,
        "TVD": tvd
    }

def load_parquet_as_array(file_path, n_pkts):
    """
    Load a parquet file and convert it into a 3D numpy array using only the PL column.
    
    Parameters:
        file_path (str): Path to the parquet file.
    
    Returns:
        tuple: 
            - np.ndarray: 2D array with shape (samples, time_steps).
            - np.ndarray: Labels associated with each sample.
    """
    df = pd.read_parquet(file_path)
    df = df[~df['IS_TRAIN']]
    labels = df["LABEL"].values
    
    # Ensure the PL column exists and process only PL
    if "PL" not in df.columns:
        raise ValueError(f"Column 'PL' not found in {file_path}")
    
    features = df["PL"].apply(lambda x: x[:n_pkts] if isinstance(x, list) else [0] * n_pkts).values
    data = np.array(features.tolist())
    
    return data, labels

def compute_histograms(real_data, synthetic_data, n_bins):
    """
    Compute histograms for real and synthetic data using only PL values.
    
    Parameters:
        real_data (np.ndarray): Real dataset (samples, time_steps).
        synthetic_data (np.ndarray): Synthetic dataset (samples, time_steps).
        n_bins (int): Number of bins for the histogram.
    
    Returns:
        tuple: Normalized histograms for real and synthetic datasets.
    """
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', subsample=1000, strategy='uniform')

    discretizer.fit(real_data)
    discr_real_data = discretizer.transform(real_data)
    discr_synthetic_data = discretizer.transform(synthetic_data)

    real_histograms = np.histogram(discr_real_data, bins=n_bins, range=(0, n_bins), density=True)[0]
    synthetic_histograms = np.histogram(discr_synthetic_data, bins=n_bins, range=(0, n_bins), density=True)[0]
    
    return real_histograms, synthetic_histograms

def compute_per_class_distances(real_data, real_labels, synthetic_data, synthetic_labels, n_bins):
    """
    Compute per-class distances for the real and synthetic datasets.
    
    Parameters:
        real_data (np.ndarray): Real dataset (samples, time_steps).
        real_labels (np.ndarray): Labels for the real dataset.
        synthetic_data (np.ndarray): Synthetic dataset (samples, time_steps).
        synthetic_labels (np.ndarray): Labels for the synthetic dataset.
        n_bins (int): Number of bins for histogram computation.
    
    Returns:
        dict: Per-class distances (JSD, Hellinger, TVD) for each class.
    """
    classes = np.unique(real_labels)
    per_class_distances = {}

    for cls in classes:
        # Filter real and synthetic data for the current class
        real_cls_data = real_data[real_labels == cls]
        synthetic_cls_data = synthetic_data[synthetic_labels == cls]

        # Compute histograms
        real_histograms, synthetic_histograms = compute_histograms(real_cls_data, synthetic_cls_data, n_bins)

        # Compute distances
        distances = compute_distances(real_histograms, synthetic_histograms, n_features=1)

        per_class_distances[cls] = distances

    return per_class_distances

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate similarity between real and synthetic datasets using histogram-based distances.")
    parser.add_argument("--real_path", type=str, required=True, help="Path to the real dataset (parquet file).")
    parser.add_argument("--synthetic_path", type=str, required=True, help="Path to the synthetic dataset (parquet file).")
    parser.add_argument("--n_pkts", type=int, default=10, help="Number of packets to consider (default: 10).")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of bins for histogram computation (default: 10).")
    args = parser.parse_args()

    # Load datasets
    real_data, real_labels = load_parquet_as_array(args.real_path, args.n_pkts)
    synthetic_data, synthetic_labels = load_parquet_as_array(args.synthetic_path, args.n_pkts)

    # Compute per-class distances
    per_class_distances = compute_per_class_distances(real_data, real_labels, synthetic_data, synthetic_labels, args.n_bins)

    # Print results
    Path(f"./evaluation_results").mkdir(parents=True, exist_ok=True)
    output_file = f"./evaluation_results/{Path(args.real_path).stem}_PL_{args.n_bins}.txt"
    with open(output_file, "w") as file:
        for cls, distances in per_class_distances.items():
            print(f"Class {cls}:")
            print("  Jensen-Shannon Divergence:", distances["JSD"])
            print("  Hellinger Distance:", distances["Hellinger"])
            print("  Total Variation Distance:", distances["TVD"])
            file.write(f"Class {cls}:\n")
            file.write(f"   Jensen-Shannon Divergence: {distances['JSD']}\n")
            file.write(f"   Hellinger Distance: {distances['Hellinger']}\n")
            file.write(f"   Total Variation Distance: {distances['TVD']}\n")
