"""
generate_dataset.py - Generates a synthetic training dataset for the road damage
priority classifier. Run this before train_model.py.

Usage:
    python generate_dataset.py
"""

import numpy as np
import csv
import os

RANDOM_SEED = 42
N_SAMPLES = 2000
OUTPUT_FILE = "road_damage_dataset.csv"


def priority_label(dmg_pct, regions, crack_den, roughness, dark_pct):
    """Rule-based labelling that mimics expert judgment."""
    score = 0
    score += dmg_pct * 1.5
    score += min(regions, 30) * 1.2
    score += crack_den * 400
    score += roughness * 50
    score += dark_pct * 0.8

    # Add realistic noise
    score += np.random.normal(0, 5)

    if score < 30:
        return "Low Priority"
    elif score < 60:
        return "Medium Priority"
    elif score < 90:
        return "High Priority"
    else:
        return "Critical Priority"


def generate():
    rng = np.random.default_rng(RANDOM_SEED)

    # Damage percentage: 0-100, skewed toward low values (most roads are fine)
    dmg_pcts = rng.beta(1.5, 5, N_SAMPLES) * 100

    # Number of damaged regions: 0-50
    regions = (rng.exponential(scale=5, size=N_SAMPLES)).clip(0, 50).astype(int)

    # Crack density: 0-0.5
    crack_dens = rng.beta(1.2, 6, N_SAMPLES) * 0.5

    # Texture roughness: 0-1
    roughness = rng.beta(2, 4, N_SAMPLES)

    # Dark surface percentage: 0-60
    dark_pcts = rng.beta(1.5, 5, N_SAMPLES) * 60

    rows = []
    for i in range(N_SAMPLES):
        label = priority_label(
            dmg_pcts[i], regions[i], crack_dens[i], roughness[i], dark_pcts[i]
        )
        rows.append([
            round(float(dmg_pcts[i]), 3),
            int(regions[i]),
            round(float(crack_dens[i]), 5),
            round(float(roughness[i]), 5),
            round(float(dark_pcts[i]), 3),
            label,
        ])

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "damage_percentage", "num_damaged_regions", "crack_density",
            "texture_roughness", "dark_surface_percentage", "priority"
        ])
        writer.writerows(rows)

    # Print distribution
    labels = [r[5] for r in rows]
    for lbl in ["Low Priority", "Medium Priority", "High Priority", "Critical Priority"]:
        count = labels.count(lbl)
        print(f"  {lbl}: {count} ({count / N_SAMPLES * 100:.1f}%)")

    print(f"\nDataset saved to {OUTPUT_FILE}  ({N_SAMPLES} rows)")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    print("Generating synthetic road damage dataset...")
    generate()