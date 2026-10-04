"""
HEATMAP TEMUAN 1 — Perubahan IE/EE20 & IE/EE40 (Two-Stage vs Single-Stage)
===========================================================================
Menghasilkan 2 heatmap (Stage 1 x Stage 2) yang menunjukkan seberapa besar
penambahan p_rentan (Stage 1) mengubah performa IE/EE dibanding baseline
single-stage dari masing-masing algoritma Stage 2.

Baris  = algoritma Stage 1 (LogReg, RF, ET, XGB)
Kolom  = algoritma Stage 2 (Ridge, RF, ET, XGB)
Nilai  = (IE/EE single-stage) - (IE/EE two-stage), dirata-rata per kabupaten
         -> POSITIF (hijau) = two-stage MENGURANGI error -> lebih baik
         -> NEGATIF (merah) = two-stage MENAMBAH error   -> lebih buruk
"""

import warnings
# mengimpor path untuk menangani alamat file dan folder
from pathlib import Path
# untuk operasi numerik dan array
import numpy as np
# untuk membaca dan mengolah data berbentuk tabel csv
import pandas as pd
# untuk membuat grafik heatmap
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

BASE_DIR = Path(".")  

# Urutan tampil di heatmap (baris = Stage 1, kolom = Stage 2)
STAGE1_ALGOS = ["logreg", "rf", "et", "xgb"]
STAGE2_ALGOS = ["ridge", "rf", "et", "xgb"]

LABELS = {
    "logreg": "LogReg",
    "ridge": "Ridge",
    "rf": "RF",
    "et": "ET",
    "xgb": "XGB",
}

#   Single-stage : output_single_stage_{stage2}_timedef/report_{stage2}_single_2025_timedef.csv
#   Two-stage    : output_two_stage_{stage1}_{stage2}_raw/report_{stage2}_two_stage_2025_timedef.csv
SINGLE_PATH_TEMPLATE = "output_single_stage_{stage2}_timedef/report_{stage2}_single_2025_timedef.csv"
TWO_PATH_TEMPLATE = "output_two_stage_{stage1}_{stage2}_raw/report_{stage2}_two_stage_2025_timedef.csv"

KAB_COL = "Kab"          # nama kolom kabupaten di report csv
IE20_COL, EE20_COL = "IE20", "EE20"
IE40_COL, EE40_COL = "IE40", "EE40"

# mencari kolom berdasarkan nama tanpa membedakan huruf besar-kecil atau spasi di awal akhir
def _find_col(df: pd.DataFrame, target: str) -> str:
    """Cari nama kolom tanpa peduli huruf besar/kecil/spasi."""
    norm_target = target.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == norm_target:
            return c
    raise KeyError(f"Kolom '{target}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")

# memastikan file ada, lalu membaca file csv menjadi dataframe
def load_report(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    kab_col = _find_col(df, KAB_COL)
    # buang baris ringkasan average, karena tidak mewakili kabupaten tertentu
    mask_avg = df[kab_col].astype(str).str.strip().str.lower() == "average"
    df_clean = df.loc[~mask_avg].copy()
    return df_clean

# mengambil kolom IE dan EE, mengubah nilainya menjadi angka, lalu merata-ratakan keduanya. 
def avg_ie_ee(df: pd.DataFrame, ie_col: str, ee_col: str) -> float:
    ie = df[_find_col(df, ie_col)].astype(float)
    ee = df[_find_col(df, ee_col)].astype(float)
    # IE dan EE identik di data ini, tapi tetap dirata-rata
    combined = pd.concat([ie, ee]).mean()
    return combined

# fungsi ini menghitung angka untuk setiap sel heatmap. 
def compute_delta_matrix(threshold: str) -> pd.DataFrame:
    """threshold: '20' atau '40' -> mengembalikan matrix Stage1 x Stage2 berisi delta IE/EE."""
    ie_col = f"IE{threshold}"
    ee_col = f"EE{threshold}"

    matrix = pd.DataFrame(index=STAGE1_ALGOS, columns=STAGE2_ALGOS, dtype=float)
    single_baseline = {}
    for s2 in STAGE2_ALGOS:
        path = BASE_DIR / SINGLE_PATH_TEMPLATE.format(stage2=s2)
        try:
            df_single = load_report(path)
            single_baseline[s2] = avg_ie_ee(df_single, ie_col, ee_col)
        except FileNotFoundError:
            warnings.warn(f"[SKIP] File single-stage tidak ditemukan: {path}")
            single_baseline[s2] = np.nan

    for s1 in STAGE1_ALGOS:
        for s2 in STAGE2_ALGOS:
            path = BASE_DIR / TWO_PATH_TEMPLATE.format(stage1=s1, stage2=s2)
            try:
                df_two = load_report(path)
                two_val = avg_ie_ee(df_two, ie_col, ee_col)
                base_val = single_baseline[s2]
                # delta positif = two-stage MENGURANGI error = perbaikan
                matrix.loc[s1, s2] = base_val - two_val
            except FileNotFoundError:
                warnings.warn(f"[SKIP] File two-stage tidak ditemukan: {path}")
                matrix.loc[s1, s2] = np.nan

    return matrix

# plot heatmap dengan skala warna merah-putih-hijau, center di 0, dan anotasi angka di tiap sel
def plot_heatmap(matrix: pd.DataFrame, threshold: str, out_path: Path):
    # 1. mengambil angka dari dataframe
    data = matrix.values.astype(float)

    # skala warna: merah (memburuk) - putih (netral) - hijau (membaik), center di 0
    finite_vals = data[~np.isnan(data)]
    if finite_vals.size == 0:
        raise ValueError("Semua nilai matrix kosong / NaN — cek path file report.")
    vmax = max(abs(finite_vals.min()), abs(finite_vals.max()))
    vmax = vmax if vmax > 0 else 1e-6
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(data, cmap="RdYlGn", norm=norm, aspect="auto")

    ax.set_xticks(range(len(STAGE2_ALGOS)))
    ax.set_xticklabels([LABELS[a] for a in STAGE2_ALGOS], fontsize=11)
    ax.set_yticks(range(len(STAGE1_ALGOS)))
    ax.set_yticklabels([LABELS[a] for a in STAGE1_ALGOS], fontsize=11)

    ax.set_xlabel("Stage 2 (Regressor)", fontsize=11, labelpad=10)
    ax.set_ylabel("Stage 1 (Classifier p_rentan)", fontsize=11, labelpad=10)
    ax.set_title(
        f"Perubahan IE/EE{threshold} — Two-Stage vs Single-Stage\n"
        f"(hijau = error berkurang / membaik, merah = error bertambah / memburuk)",
        fontsize=11.5, pad=14,
    )

    # anotasi angka di tiap sel
    for i in range(len(STAGE1_ALGOS)):
        for j in range(len(STAGE2_ALGOS)):
            val = data[i, j]
            if np.isnan(val):
                text = "n/a"
            else:
                text = f"{val:+.4f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                     color="black", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Δ IE/EE (single − two-stage)", fontsize=9)

    ax.set_xticks(np.arange(-0.5, len(STAGE2_ALGOS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(STAGE1_ALGOS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    out_dir = BASE_DIR / "heatmap_temuan1_output"
    out_dir.mkdir(exist_ok=True)
    # 1. menghitung dan menyimpan heatmap perubahan IE/EE20 
    # 2. menghitung dan menyimpan heatmap perubahan IE/EE40
    for threshold in ["20", "40"]:
        matrix = compute_delta_matrix(threshold)
        print(f"\n=== Delta IE/EE{threshold} (Stage1 x Stage2) ===")
        print(matrix.round(4))
        plot_heatmap(matrix, threshold, out_dir / f"heatmap_ie_ee{threshold}.png")

    print("\nSelesai. Cek folder 'heatmap_temuan1_output' untuk hasil PNG-nya.")