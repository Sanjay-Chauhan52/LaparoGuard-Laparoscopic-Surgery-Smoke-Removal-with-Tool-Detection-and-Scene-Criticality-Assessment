"""
plot_criticality_trajectory.py — Surgical Criticality Risk Index Over Time
Plots the dynamic criticality curve, colored risk zones (Green/Yellow/Red),
and component score breakdowns across surgical video duration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = r"C:\Users\thesa\Desktop\LaparoscopicDetection"
CSV_PATH = os.path.join(BASE_DIR, "critical_output_improved", "critical_events.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "detection_output", "criticality_timeline.png")

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    df = pd.read_csv(CSV_PATH)
    time = df['timestamp'].values
    criticality = df['criticality_score'].values
    cutting = df['cutting_score'].values
    contact = df['contact_score'].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2.2, 1.2]})

    # --- Top Plot: Overall Criticality Score ---
    ax1.plot(time, criticality, color='#1E293B', linewidth=2.2, label='Criticality Risk Index (Smoothed)')

    # Shaded Risk Zones
    ax1.axhspan(0.0, 0.30, color='#22C55E', alpha=0.18, label='Normal / Safe Zone (< 0.30)')
    ax1.axhspan(0.30, 0.60, color='#EAB308', alpha=0.18, label='Moderate Risk / Contact Zone (0.30 - 0.60)')
    ax1.axhspan(0.60, 1.00, color='#EF4444', alpha=0.18, label='Critical Risk / Dissection Zone (> 0.60)')

    ax1.axhline(0.30, color='#16A34A', linestyle='--', linewidth=1.2)
    ax1.axhline(0.60, color='#DC2626', linestyle='--', linewidth=1.2)

    ax1.set_ylabel('Criticality Index (0.0 to 1.0)', fontsize=11, weight='bold')
    ax1.set_title('LaparoGuard: Real-Time Surgical Criticality & Risk Trajectory (cutting.mp4)', fontsize=13, weight='bold', pad=10)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)

    # --- Bottom Plot: Key Contributing Sub-Scores ---
    ax2.plot(time, cutting, color='#DC2626', linewidth=1.8, label='Hook Cutting Score (+0.50)')
    ax2.plot(time, contact, color='#2563EB', linewidth=1.5, linestyle='-', label='Tissue Contact Score (x0.35)')

    ax2.set_xlabel('Elapsed Surgical Time (Seconds)', fontsize=11, weight='bold')
    ax2.set_ylabel('Sub-Score', fontsize=10, weight='bold')
    ax2.set_ylim(-0.05, 0.60)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()
    print(f"Successfully generated surgical timeline plot: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
