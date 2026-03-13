from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\Claude code\EVRL260207")
RAW_DIR = ROOT / "data" / "SIP" / "demand_original"
OUT = ROOT / "results" / "marl_supplementary_20260308-212225"


def q_files():
    files = []
    for p in RAW_DIR.glob("q_*.csv"):
        tail = p.stem.split("_")[-1]
        if tail.isdigit():
            files.append(p)
    return sorted(files, key=lambda p: int(p.stem.split("_")[-1]))


def load_raw_stats(z=50, t=26):
    files = q_files()
    file_time = []
    file_origin = []
    file_dest = []
    od_sum = np.zeros((z, z), dtype=float)
    od_nonzero = np.zeros((z, z), dtype=float)

    for fp in files:
        df = pd.read_csv(fp, header=None, names=["dataset", "O", "D", "T", "V"])
        df = df[(df["O"] >= 0) & (df["O"] < z) & (df["D"] >= 0) & (df["D"] < z) & (df["T"] >= 0) & (df["T"] < t)]
        time_vec = np.zeros(t, dtype=float)
        origin_vec = np.zeros(z, dtype=float)
        dest_vec = np.zeros(z, dtype=float)

        # Vectorized aggregations (much faster than row iteration)
        t_sum = df.groupby("T", as_index=True)["V"].sum()
        o_sum = df.groupby("O", as_index=True)["V"].sum()
        d_sum = df.groupby("D", as_index=True)["V"].sum()

        time_vec[t_sum.index.to_numpy(dtype=int)] = t_sum.to_numpy(dtype=float)
        origin_vec[o_sum.index.to_numpy(dtype=int)] = o_sum.to_numpy(dtype=float)
        dest_vec[d_sum.index.to_numpy(dtype=int)] = d_sum.to_numpy(dtype=float)

        od_mat = df.pivot_table(index="O", columns="D", values="V", aggfunc="sum", fill_value=0.0)
        o_idx = od_mat.index.to_numpy(dtype=int)
        d_idx = od_mat.columns.to_numpy(dtype=int)
        mat_vals = od_mat.to_numpy(dtype=float)
        od_sum[np.ix_(o_idx, d_idx)] += mat_vals
        od_nonzero[np.ix_(o_idx, d_idx)] += (mat_vals > 0).astype(float)

        file_time.append(time_vec)
        file_origin.append(origin_vec)
        file_dest.append(dest_vec)

    return (
        np.asarray(file_time, dtype=float),
        np.asarray(file_origin, dtype=float),
        np.asarray(file_dest, dtype=float),
        od_sum,
        od_nonzero,
    )


def fig_time_profile(file_time, out_file):
    mean_t = file_time.mean(axis=0)
    std_t = file_time.std(axis=0)
    x = np.arange(mean_t.size)
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.plot(x, mean_t, color="#2a9d8f", linewidth=2.2, label="mean demand")
    ax.fill_between(x, mean_t - std_t, mean_t + std_t, color="#2a9d8f", alpha=0.2, label="±1 std")
    ax.set_title("Raw Demand Time Profile from q-files")
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Demand")
    tick_idx = np.arange(0, mean_t.size, 2)
    tick_labels = []
    for i in tick_idx:
        total_min = 7 * 60 + int(i) * 30
        hh = total_min // 60
        mm = total_min % 60
        tick_labels.append(f"{hh:02d}:{mm:02d}")
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_zone_rank(file_origin, file_dest, out_file):
    o = file_origin.mean(axis=0)
    d = file_dest.mean(axis=0)
    top_o = np.argsort(o)[-15:][::-1]
    top_d = np.argsort(d)[-15:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    axes[0].bar(np.arange(15), o[top_o], color="#457b9d")
    axes[0].set_xticks(np.arange(15))
    axes[0].set_xticklabels(top_o)
    axes[0].set_title("Top-15 Origin Demand Zones (raw)")
    axes[0].set_xlabel("Zone ID")
    axes[0].set_ylabel("Avg demand")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(np.arange(15), d[top_d], color="#e76f51")
    axes[1].set_xticks(np.arange(15))
    axes[1].set_xticklabels(top_d)
    axes[1].set_title("Top-15 Destination Demand Zones (raw)")
    axes[1].set_xlabel("Zone ID")
    axes[1].set_ylabel("Avg demand")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_od_heatmap(od_sum, out_file):
    fig, ax = plt.subplots(figsize=(7.5, 6.1))
    im = ax.imshow(od_sum, cmap="viridis", aspect="auto")
    ax.set_title("Raw OD Demand Heatmap (sum over q-files)")
    ax.set_xlabel("Destination zone")
    ax.set_ylabel("Origin zone")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_od_nonzero_ratio(od_nonzero, n_files, out_file):
    ratio = od_nonzero / max(1, n_files)
    fig, ax = plt.subplots(figsize=(7.5, 6.1))
    im = ax.imshow(ratio, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_title("OD Activity Ratio Across q-files")
    ax.set_xlabel("Destination zone")
    ax.set_ylabel("Origin zone")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_file, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_notes(file_time, file_origin, file_dest, out_file):
    mean_t = file_time.mean(axis=0)
    lines = []
    lines.append("# Dataset Variability Notes (raw q-files)\n")
    lines.append(f"- q-files count: {file_time.shape[0]}\n")
    lines.append(f"- time demand peak index: {int(np.argmax(mean_t))}, value: {float(mean_t.max()):.2f}\n")
    lines.append(f"- time demand valley index: {int(np.argmin(mean_t))}, value: {float(mean_t.min()):.2f}\n")
    top_o = np.argsort(file_origin.mean(axis=0))[-5:][::-1].tolist()
    top_d = np.argsort(file_dest.mean(axis=0))[-5:][::-1].tolist()
    lines.append(f"- top origin zones: {top_o}\n")
    lines.append(f"- top destination zones: {top_d}\n")
    lines.append("\nThese plots show richer heterogeneity than preprocessed od_demand slices.\n")
    out_file.write_text("".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    file_time, file_origin, file_dest, od_sum, od_nonzero = load_raw_stats()
    n_files = file_time.shape[0]

    fig_time_profile(file_time, OUT / "figB9_raw_time_profile.png")
    fig_zone_rank(file_origin, file_dest, OUT / "figB10_raw_zone_rank.png")
    fig_od_heatmap(od_sum, OUT / "figB11_raw_od_heatmap.png")
    fig_od_nonzero_ratio(od_nonzero, n_files, OUT / "figB12_raw_od_activity_ratio.png")
    write_notes(file_time, file_origin, file_dest, OUT / "dataset_variability_notes.md")
    print(OUT)


if __name__ == "__main__":
    main()
