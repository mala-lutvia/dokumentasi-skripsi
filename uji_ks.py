from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import ks_2samp, gaussian_kde

warnings.filterwarnings("ignore")

# =========================================================
# 1. PENGATURAN FILE
# =========================================================
# Script uji_ks.py diasumsikan berada satu folder dengan:
# - gk_full_kab.xlsx
# - folder output/susenas_clean_2022.csv ... _2025.csv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "output"
GK_FILE = BASE_DIR / "gk_full_kab.xlsx"
OUTPUT_DIR = DATA_DIR / "hasil_ks_time_deflator"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    2022: DATA_DIR / "susenas_clean_2022.csv",
    2023: DATA_DIR / "susenas_clean_2023.csv",
    2024: DATA_DIR / "susenas_clean_2024.csv",
    2025: DATA_DIR / "susenas_clean_2025.csv",
}

KODE_COL = "kode_kab"
PCE_COL = "pcexp"
PROVINSI_DIPAKAI = [33, 34]
TIMEDEF_COLS = {
    2022: "timedef_22",
    2023: "timedef_23",
    2024: "timedef_24",
    2025: "timedef_25",
}
PLOT_KAB = 3401
TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR = 2025
COLORS = {
    "train": "#4C72B0",
    "test": "#DD8452",
}


def province_from_kode(kode):
    """Ambil kode provinsi dari kode gabungan provinsi-kabupaten, mis. 3401 -> 34."""
    return (kode // 100).astype("Int64")


def calculate_ks(source_data, target_data):
    s = pd.to_numeric(pd.Series(source_data), errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(pd.Series(target_data), errors="coerce").to_numpy(dtype=float)
    s = s[np.isfinite(s)]
    t = t[np.isfinite(t)]
    if len(s) == 0 or len(t) == 0:
        return np.nan, np.nan, len(s), len(t)
    result = ks_2samp(s, t, alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue), len(s), len(t)


def kde_curve(values, lo=0.0, hi=1.0, n_points=500, bw_adjust=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or np.unique(values).size < 2:
        return None, None
    kde = gaussian_kde(values)
    kde.set_bandwidth(kde.factor * bw_adjust)
    xs = np.linspace(lo, hi, n_points)
    return xs, kde(xs)


def style_axis(ax, lo, hi):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.3")
    ax.spines["bottom"].set_color("0.3")
    ax.grid(True, color="0.88", linewidth=0.7, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(lo, hi)


def plot_density_panel(ax, train_values, test_values, title, xlabel, bw_adjust=1.0):
    train_values = pd.Series(train_values).dropna().to_numpy(dtype=float)
    test_values = pd.Series(test_values).dropna().to_numpy(dtype=float)

    for vals in (train_values, test_values):
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2 or np.unique(vals).size < 2:
            raise ValueError(
                "KDE membutuhkan minimal dua observasi dengan nilai berbeda "
                "pada masing-masing kelompok."
            )

    lo = min(float(np.min(train_values)), float(np.min(test_values)))
    hi = max(float(np.max(train_values)), float(np.max(test_values)))
    pad = (hi - lo) * 0.04 if hi > lo else 0.01
    lo, hi = max(0.0, lo - pad), hi + pad

    xs_tr, ys_tr = kde_curve(train_values, lo, hi, bw_adjust=bw_adjust)
    xs_te, ys_te = kde_curve(test_values, lo, hi, bw_adjust=bw_adjust)

    ax.fill_between(xs_tr, ys_tr, color=COLORS["train"], alpha=0.30, linewidth=0)
    ax.fill_between(xs_te, ys_te, color=COLORS["test"], alpha=0.30, linewidth=0)
    ax.plot(xs_tr, ys_tr, color=COLORS["train"], linewidth=2.2, solid_capstyle="round")
    ax.plot(xs_te, ys_te, color=COLORS["test"], linewidth=2.2, solid_capstyle="round")

    ymax = max(float(np.max(ys_tr)), float(np.max(ys_te)))
    ax.set_ylim(0, ymax * 1.15)
    ax.set_title(title, fontsize=15, pad=10, color="0.15")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10, colors="0.25")
    style_axis(ax, lo, hi)
    ax.legend(handles=[
        Line2D([0], [0], color=COLORS["train"], lw=2.2,
               label=f"Train Data (n={len(train_values):,})"),
        Line2D([0], [0], color=COLORS["test"], lw=2.2,
               label=f"Test Data (n={len(test_values):,})"),
    ], loc="upper right", frameon=False, fontsize=10)


# =========================================================
# 2. BACA DATA SUSENAS, FILTER JATENG + DIY
# =========================================================
frames = []
for year, file_path in FILES.items():
    if not file_path.exists():
        raise FileNotFoundError(f"File Susenas tidak ditemukan: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)
    required = [KODE_COL, PCE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom {missing} tidak ditemukan di {file_path}. "
            f"Kolom tersedia: {df.columns.tolist()}"
        )

    df = df[required].copy()
    df[KODE_COL] = pd.to_numeric(df[KODE_COL], errors="coerce")
    df[PCE_COL] = pd.to_numeric(df[PCE_COL], errors="coerce")
    df = df.dropna(subset=required)
    df[KODE_COL] = df[KODE_COL].astype(int)

    # Kode gabungan: 33xx = Jawa Tengah, 34xx = DI Yogyakarta.
    df = df[df[KODE_COL].between(3300, 3499)].copy()
    df = df[df[KODE_COL].floordiv(100).isin(PROVINSI_DIPAKAI)].copy()
    df["tahun"] = int(year)

    print(f"Susenas {year}: {len(df):,} baris valid setelah filter Jateng + DIY")
    frames.append(df)

data = pd.concat(frames, ignore_index=True)
if data.empty:
    raise ValueError(
        "Tidak ada data setelah filter kode_kab 3300-3499. "
        "Pastikan kode_kab memakai format gabungan, misalnya 3401."
    )


# =========================================================
# 3. BACA TIME DEFLATOR DARI GK DAN TANGANI DUPLIKAT
# =========================================================
if not GK_FILE.exists():
    raise FileNotFoundError(
        f"File GK tidak ditemukan: {GK_FILE}\n"
        "Pastikan gk_full_kab.xlsx berada satu folder dengan uji_ks.py."
    )

gk = pd.read_excel(GK_FILE)
required_gk = [KODE_COL, *TIMEDEF_COLS.values()]
missing_gk = [c for c in required_gk if c not in gk.columns]
if missing_gk:
    raise ValueError(
        f"Kolom {missing_gk} tidak ditemukan di file GK. "
        f"Kolom tersedia: {gk.columns.tolist()}"
    )

gk = gk[required_gk].copy()
gk[KODE_COL] = pd.to_numeric(gk[KODE_COL], errors="coerce")
gk = gk.dropna(subset=[KODE_COL])
gk[KODE_COL] = gk[KODE_COL].astype(int)
gk = gk[gk[KODE_COL].between(3300, 3499)].copy()
gk = gk[gk[KODE_COL].floordiv(100).isin(PROVINSI_DIPAKAI)].copy()

# Bentuk lebar -> panjang, lalu validasi duplikat hanya untuk Jateng + DIY.
gk_long = gk.melt(
    id_vars=KODE_COL,
    value_vars=list(TIMEDEF_COLS.values()),
    var_name="td_col",
    value_name="timedef",
)
year_map = {col: year for year, col in TIMEDEF_COLS.items()}
gk_long["tahun"] = gk_long["td_col"].map(year_map)
gk_long["timedef"] = pd.to_numeric(gk_long["timedef"], errors="coerce")
gk_long = gk_long.dropna(subset=[KODE_COL, "tahun", "timedef"])
gk_long["tahun"] = gk_long["tahun"].astype(int)

duplicates = gk_long[gk_long.duplicated([KODE_COL, "tahun"], keep=False)]
if not duplicates.empty:
    dup_codes = duplicates[[KODE_COL, "tahun"]].drop_duplicates()
    raise ValueError(
        "Masih ada kode_kab-tahun duplikat di file GK untuk Jateng/DIY. "
        "Script berhenti agar nilai time deflator tidak digandakan.\n"
        f"{dup_codes.to_string(index=False)}\n"
        "Periksa baris duplikat pada gk_full_kab.xlsx."
    )

# Pastikan tidak ada kode Susenas yang tidak punya time deflator.
data = data.merge(
    gk_long[[KODE_COL, "tahun", "timedef"]],
    on=[KODE_COL, "tahun"],
    how="left",
    validate="many_to_one",
    indicator=True,
)
unmatched = data.loc[data["_merge"] == "left_only", [KODE_COL, "tahun"]].drop_duplicates()
if not unmatched.empty:
    raise ValueError(
        "Sebagian kode wilayah-tahun Susenas tidak memiliki time deflator di file GK. "
        "Contoh:\n" + unmatched.head(20).to_string(index=False)
    )
data = data.drop(columns="_merge")

if data["timedef"].isna().any() or (data["timedef"] <= 0).any():
    raise ValueError("Ditemukan time deflator kosong, nol, atau negatif.")

data["timepcexp"] = data[PCE_COL] * data["timedef"]
data.loc[data[PCE_COL] < 0, "timepcexp"] = np.nan
data.to_csv(OUTPUT_DIR / "data_time_deflator.csv", index=False)


# =========================================================
# 4. UJI KS: 2022/2023/2024 VS 2025, PER KAB/KOTA
# =========================================================
results = []
for kode_kab, df_kab in data.groupby(KODE_COL):
    target_2025 = df_kab[df_kab["tahun"] == TEST_YEAR]
    if target_2025.empty:
        continue

    for year in TRAIN_YEARS:
        historic = df_kab[df_kab["tahun"] == year]
        if historic.empty:
            continue

        d_before, p_before, n_hist, n_2025 = calculate_ks(
            historic[PCE_COL], target_2025[PCE_COL]
        )
        d_after, p_after, n_hist_adj, n_2025_adj = calculate_ks(
            historic["timepcexp"], target_2025["timepcexp"]
        )

        results.append({
            "kode_kab": int(kode_kab),
            "tahun_historis": year,
            "tahun_acuan": TEST_YEAR,
            "D_sebelum": d_before,
            "pvalue_sebelum": p_before,
            "D_sesudah": d_after,
            "pvalue_sesudah": p_after,
            "n_historis_sebelum": n_hist,
            "n_2025_sebelum": n_2025,
            "n_historis_sesudah": n_hist_adj,
            "n_2025_sesudah": n_2025_adj,
        })

results_df = pd.DataFrame(results)
if results_df.empty:
    raise ValueError("Tidak ada hasil uji KS. Pastikan data 2025 tersedia untuk kode wilayah.")

results_path = OUTPUT_DIR / "hasil_ks_time_deflator.csv"
results_df.to_csv(results_path, index=False)

summary = (
    results_df.groupby("tahun_historis")
    .agg(
        jumlah_kab=("kode_kab", "nunique"),
        rata_D_sebelum=("D_sebelum", "mean"),
        rata_D_sesudah=("D_sesudah", "mean"),
        median_D_sebelum=("D_sebelum", "median"),
        median_D_sesudah=("D_sesudah", "median"),
        rata_pvalue_sebelum=("pvalue_sebelum", "mean"),
        rata_pvalue_sesudah=("pvalue_sesudah", "mean"),
    )
    .reset_index()
)
summary_path = OUTPUT_DIR / "ringkasan_ks_time_deflator.csv"
summary.to_csv(summary_path, index=False)

print("\nHasil uji KS (baris awal):")
print(results_df.head(12).to_string(index=False))
print("\nRingkasan statistik KS:")
print(summary.to_string(index=False))


# =========================================================
# 5. VISUALISASI DENSITY KDE GAYA ZENN
#    Dua panel: sebelum dan sesudah deflator untuk PLOT_KAB
# =========================================================
plot_data = data[data[KODE_COL] == PLOT_KAB].copy()
if plot_data.empty:
    available = sorted(data[KODE_COL].unique().tolist())
    print(
        f"\n[PERINGATAN] PLOT_KAB={PLOT_KAB} tidak ditemukan. "
        f"Contoh kode yang tersedia: {available[:15]}"
    )
else:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="white")

    for ax, col, title in [
        (axes[0], PCE_COL, "(a) Sebelum time deflator"),
        (axes[1], "timepcexp", "(b) Sesudah time deflator"),
    ]:
        train_values = plot_data.loc[
            plot_data["tahun"].isin(TRAIN_YEARS), col
        ].dropna().to_numpy(dtype=float)
        test_values = plot_data.loc[
            plot_data["tahun"] == TEST_YEAR, col
        ].dropna().to_numpy(dtype=float)

        # Untuk plot pengeluaran yang bukan probabilitas, grid mengikuti data.
        plot_density_panel(
            ax,
            train_values,
            test_values,
            title=title,
            xlabel="Pengeluaran per kapita",
            bw_adjust=1.0,
        )

    fig.suptitle(
        f"Distribusi Pengeluaran per Kapita — Kode Kabupaten/Kota {PLOT_KAB}",
        fontsize=15,
        color="0.15",
    )
    fig.tight_layout()
    plot_path = OUTPUT_DIR / "distribusi_time_deflator_kde.png"
    fig.savefig(plot_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"\nGrafik KDE tersimpan: {plot_path}")

print("\n[SELESAI] Uji KS time deflator")
print(f"Data penyesuaian : {OUTPUT_DIR / 'data_time_deflator.csv'}")
print(f"Hasil uji KS     : {results_path}")
print(f"Ringkasan KS     : {summary_path}")
print(f"Folder output    : {OUTPUT_DIR}")
