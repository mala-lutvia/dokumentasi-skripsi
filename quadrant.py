"""
QUADRANT CHART — Δ IE/EE vs Δ RMSE per Kabupaten (Single-Stage vs Two-Stage)
=============================================================================
Untuk satu pasangan (algoritma Stage 2, algoritma Stage 1), skrip ini
menghasilkan 2 chart:
    1. Delta IE/EE20 vs Delta RMSE20
    2. Delta IE/EE40 vs Delta RMSE40

Tiap titik = 1 kabupaten, diberi label nama kabupaten, dan diwarnai sesuai
kuadran tempat dia berada:

    Delta = (nilai Two-Stage) - (nilai Single-Stage)
    -> Delta IE/EE negatif (kiri)  = error berkurang = BAIK
    -> Delta RMSE  negatif (bawah) = RMSE berkurang  = BAIK

    Kuadran kiri-bawah  (IE- , RMSE-) -> HIJAU  -> "Nilai terbaik" (keduanya membaik)
    Kuadran kiri-atas   (IE- , RMSE+) -> BIRU   -> IE membaik, RMSE memburuk
    Kuadran kanan-bawah (IE+ , RMSE-) -> PINK   -> RMSE membaik, IE memburuk
    Kuadran kanan-atas  (IE+ , RMSE+) -> MERAH  -> "Terburuk" (keduanya memburuk)
"""
# membaca pilihan yang diberikan saat script dijalankan, seperti algoritma dan folder output 
import argparse
# memberi peringatan jika ada hal yang tidak sesuai
import warnings
# membentuk alamat file dan folder
from pathlib import Path
# mengolah tabel dan menggambar grafik
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# membuat area warna dan legenda
import matplotlib.patches as mpatches

try:
    # menyesuaikan posisi label agar tidak saling tumpang tindih
    from adjustText import adjust_text
except ImportError as e:
    raise ImportError(
        "Library 'adjustText' belum terinstall. "
        "Jalankan: pip install adjustText"
    ) from e

KAB_NAMES = {
    3301: "Cilacap", 3302: "Banyumas", 3303: "Purbalingga", 3304: "Banjarnegara",
    3305: "Kebumen", 3306: "Purworejo", 3307: "Wonosobo", 3308: "Magelang",
    3309: "Boyolali", 3310: "Klaten", 3311: "Sukoharjo", 3312: "Wonogiri",
    3313: "Karanganyar", 3314: "Sragen", 3315: "Grobogan", 3316: "Blora",
    3317: "Rembang", 3318: "Pati", 3319: "Kudus", 3320: "Jepara",
    3321: "Demak", 3322: "Semarang", 3323: "Temanggung", 3324: "Kendal",
    3325: "Batang", 3326: "Pekalongan", 3327: "Pemalang", 3328: "Tegal",
    3329: "Brebes", 3371: "Kota Magelang", 3372: "Kota Surakarta",
    3373: "Kota Salatiga", 3374: "Kota Semarang", 3375: "Kota Pekalongan",
    3376: "Kota Tegal", 3401: "Kulon Progo", 3402: "Bantul",
    3403: "Gunungkidul", 3404: "Sleman", 3471: "Kota Yogyakarta",
}

SINGLE_PATH_TEMPLATE = "output_single_stage_{algo}_timedef/report_{algo}_single_2025_timedef.csv"
TWO_PATH_TEMPLATE = "output_two_stage_{stage1}_{algo}_raw/report_{algo}_two_stage_2025_timedef.csv"

KAB_COL = "Kab"
LABEL_ALGO = {"logreg": "LogReg", "ridge": "Ridge", "rf": "RF", "et": "ET", "xgb": "XGB"}

QUADRANT_COLORS = {
    "best": "#2ecc71",     # hijau: IE- RMSE- (keduanya membaik)
    "ie_only": "#3498db",  # biru : IE- RMSE+ (hanya IE membaik)
    "rmse_only": "#e91e8c",# pink : IE+ RMSE- (hanya RMSE membaik)
    "worst": "#e74c3c",    # merah: IE+ RMSE+ (keduanya memburuk)
}
QUADRANT_BG = {
    "best": "#eafaf1",
    "ie_only": "#eaf3fb",
    "rmse_only": "#fdeaf4",
    "worst": "#fdecea",
}

# mencari kolom tanpa membedakan huruf besar-kecil atau spasi di awal akhir
def _find_col(df: pd.DataFrame, target: str) -> str:
    norm_target = target.strip().lower()
    for c in df.columns:
        if str(c).strip().lower() == norm_target:
            return c
    raise KeyError(f"Kolom '{target}' tidak ditemukan. Kolom tersedia: {list(df.columns)}")

# memastikan file, membaca CSV, lalu menghapus baris AVERAGE
def load_report(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    kab_col = _find_col(df, KAB_COL)
    mask_avg = df[kab_col].astype(str).str.strip().str.lower() == "average"
    return df.loc[~mask_avg].copy()

# memilih kolom IE dan EE, mengubah angka menjadi float lalu merata-ratakan keduanya
def get_metric(df: pd.DataFrame, threshold: str):
    """Mengembalikan Series index=Kab code, value = rata-rata IE & EE (identik di data ini)."""
    kab_col = _find_col(df, KAB_COL)
    ie_col = _find_col(df, f"IE{threshold}")
    ee_col = _find_col(df, f"EE{threshold}")
    ie_ee = df[[ie_col, ee_col]].astype(float).mean(axis=1)
    return pd.Series(ie_ee.values, index=df[kab_col].astype(int).values)

# mengambil RMSE20 atau RMSE40, lalu mengubah menjadi float
def get_rmse(df: pd.DataFrame, threshold: str):
    kab_col = _find_col(df, KAB_COL)
    rmse_col = _find_col(df, f"RMSE{threshold}")
    return pd.Series(df[rmse_col].astype(float).values, index=df[kab_col].astype(int).values)


def quadrant_key(delta_ie: float, delta_rmse: float) -> str:
    if delta_ie <= 0 and delta_rmse <= 0:
        return "best"
    if delta_ie <= 0 and delta_rmse > 0:
        return "ie_only"
    if delta_ie > 0 and delta_rmse <= 0:
        return "rmse_only"
    return "worst"
# fungsi ini membuat grafik untuk satu threshold dan satu pasangan algoritma stage 1 dan stage 2
def plot_quadrant(df_single: pd.DataFrame, df_two: pd.DataFrame, threshold: str,
                   algo: str, stage1: str, out_path: Path):
    ie_single = get_metric(df_single, threshold)
    ie_two = get_metric(df_two, threshold)
    rmse_single = get_rmse(df_single, threshold)
    rmse_two = get_rmse(df_two, threshold)

    common = ie_single.index.intersection(ie_two.index)
    common = common.intersection(rmse_single.index).intersection(rmse_two.index)
    if len(common) == 0:
        warnings.warn(f"Tidak ada kabupaten yang cocok antara single dan two-stage untuk threshold {threshold}.")
        return

    delta_ie = (ie_two.loc[common] - ie_single.loc[common])
    delta_rmse = (rmse_two.loc[common] - rmse_single.loc[common])

    fig, ax = plt.subplots(figsize=(11, 8.5))

    xmax = max(abs(delta_ie.min()), abs(delta_ie.max()), 1e-6) * 1.25
    ymax = max(abs(delta_rmse.min()), abs(delta_rmse.max()), 1e-6) * 1.25

    # background kuadran
    ax.add_patch(mpatches.Rectangle((-xmax, -ymax), xmax, ymax, color=QUADRANT_BG["best"], zorder=0))
    ax.add_patch(mpatches.Rectangle((-xmax, 0), xmax, ymax, color=QUADRANT_BG["ie_only"], zorder=0))
    ax.add_patch(mpatches.Rectangle((0, -ymax), xmax, ymax, color=QUADRANT_BG["rmse_only"], zorder=0))
    ax.add_patch(mpatches.Rectangle((0, 0), xmax, ymax, color=QUADRANT_BG["worst"], zorder=0))
    texts = []
    scatter_x, scatter_y = [], []

    for kab_code in common:
        x = delta_ie.loc[kab_code]
        y = delta_rmse.loc[kab_code]
        qk = quadrant_key(x, y)
        color = QUADRANT_COLORS[qk]
        name = KAB_NAMES.get(int(kab_code), str(kab_code))

        ax.scatter(x, y, color=color, s=70, edgecolor="white", linewidth=0.8, zorder=3)

        scatter_x.append(x)
        scatter_y.append(y)
        texts.append(
            ax.text(x, y, name, fontsize=7.5, color="#333333", zorder=4)
        )

    adjust_text(
        texts,
        x=scatter_x,
        y=scatter_y,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5, alpha=0.7),
        expand_points=(1.4, 1.6),
        expand_text=(1.1, 1.3),
        force_text=(0.4, 0.6),
        force_points=(0.3, 0.5),
    )

    ax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=2)
    ax.axvline(0, color="black", linewidth=1, linestyle="--", zorder=2)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-ymax, ymax)

    ax.set_xlabel(f"Δ IE/EE{threshold}  (Two-Stage − Single-Stage)   ←  membaik", fontsize=10)
    ax.set_ylabel(f"Δ RMSE{threshold}  (Two-Stage − Single-Stage)   ↓  membaik", fontsize=10)

    algo_label = LABEL_ALGO.get(algo, algo.upper())
    stage1_label = LABEL_ALGO.get(stage1, stage1.upper())
    ax.set_title(
        f"{algo_label} Single-Stage vs {stage1_label}+{algo_label} Two-Stage\n"
        f"Δ IE/EE{threshold} vs Δ RMSE{threshold} per Kabupaten",
        fontsize=13, pad=14,
    )

    legend_items = [
        mpatches.Patch(color=QUADRANT_COLORS["best"], label="IE/EE↓ RMSE↓  (terbaik)"),
        mpatches.Patch(color=QUADRANT_COLORS["ie_only"], label="IE/EE↓ RMSE↑"),
        mpatches.Patch(color=QUADRANT_COLORS["rmse_only"], label="IE/EE↑ RMSE↓"),
        mpatches.Patch(color=QUADRANT_COLORS["worst"], label="IE/EE↑ RMSE↑  (terburuk)"),
    ]
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=9, frameon=False, title="Kuadran")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")

    n_best = sum(quadrant_key(delta_ie[k], delta_rmse[k]) == "best" for k in common)
    print(f"   Kabupaten di kuadran 'terbaik' (IE- RMSE-): {n_best}/{len(common)}")

def main():
    parser = argparse.ArgumentParser(description="Quadrant chart Delta IE/EE vs Delta RMSE per kabupaten.")
    parser.add_argument("--base-dir", type=str, default=".", help="Folder induk semua output_*")
    parser.add_argument("--algo", type=str, required=True, choices=["ridge", "rf", "et", "xgb"],
                         help="Algoritma Stage 2 / single-stage yang dibandingkan")
    parser.add_argument("--stage1", type=str, required=True, choices=["logreg", "rf", "et", "xgb"],
                         help="Algoritma Stage 1 pada versi two-stage")
    parser.add_argument("--out-dir", type=str, default="quadrant_temuan_output")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    single_path = base_dir / SINGLE_PATH_TEMPLATE.format(algo=args.algo)
    two_path = base_dir / TWO_PATH_TEMPLATE.format(stage1=args.stage1, algo=args.algo)

    df_single = load_report(single_path)
    df_two = load_report(two_path)
    # memanggilplot_quadrant untuk threshold 20 dan 40, lalu menyimpan chart ke folder output
    for threshold in ["20", "40"]:
        out_file = out_dir / f"quadrant_{args.stage1}_{args.algo}_{threshold}.png"
        plot_quadrant(df_single, df_two, threshold, args.algo, args.stage1, out_file)

    print(f"\nSelesai. 2 chart tersimpan di: {out_dir}")


if __name__ == "__main__":
    main()