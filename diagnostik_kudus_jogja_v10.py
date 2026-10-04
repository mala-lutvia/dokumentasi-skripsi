import os
import re
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

CONFIG = {
    "stage1_files": {
        2022: "output_stage1_logreg_raw/susenas_prentan_logreg_fixed_raw_2022.csv",
        2023: "output_stage1_logreg_raw/susenas_prentan_logreg_fixed_raw_2023.csv",
        2024: "output_stage1_logreg_raw/susenas_prentan_logreg_fixed_raw_2024.csv",
        2025: "output_stage1_logreg_raw/susenas_prentan_logreg_fixed_raw_2025.csv",
    },

    "kab_col_candidates": ["kode_kab_new", "kode_kab", "kab"],
    "prentan_col_candidates": ["p_rentan"],
    "group20_col_candidates": ["group20"],
    "group40_col_candidates": ["group40"],

    "train_years": [2022, 2023, 2024],
    "test_year": 2025,
    "train_label": "TRAIN_2022_2024",
    "test_label": "TEST_2025",

    "focus": {
        "Kudus": 3319,
        "Kota Tegal": 3376,
        "Kota Magelang": 3371,
    },

    "min_n": 5,  # ambang minimum observasi per kelompok untuk statistik & overlap

    # p_rentan adalah probabilitas (output logreg Stage 1) -> rentang alami [0,1].
    # Dipakai sebagai bin edges TETAP untuk SEMUA perbandingan overlap (B40
    # maupun B20), supaya OVL antar TRAIN vs TEST dan antar-wilayah benar-benar
    # bisa dibandingkan apple-to-apple (bukan bin yang menyesuaikan data tiap
    # perbandingan).
    "prentan_range": (0.0, 1.0),
    "hist_bins": 25,

    # bandwidth KDE untuk figure density gaya Zenn (make_covariate_shift_figures).
    # Dipakai SAMA untuk Train & Test pada setiap panel supaya kurva tetap
    # dibandingkan secara wajar (bukan salah satu dibuat lebih halus).
    "covshift_bw_adjust": 1.0,

    "output": "output/diagnostic_prentan_group40_group20",
}

CAUTION_LANGUAGE_NOTE = (
    "CATATAN BAHASA: seluruh interpretasi di script dan laporan ini memakai "
    "bahasa hati-hati (\"berkaitan dengan\", \"konsisten dengan\", \"memberikan "
    "indikasi\", \"dapat membantu menjelaskan\") dan TIDAK PERNAH memakai bahasa "
    "kausal (\"menyebabkan\", \"disebabkan oleh\", \"causal effect\"). Script ini "
    "adalah diagnostik deskriptif terhadap p_rentan hasil Stage 1, BUKAN "
    "evaluasi ulang performa model (RMSE/IE/EE tidak dihitung ulang di sini)."
)

DATA_NOTE = (
    "CATATAN DATA: p_rentan, group40, dan group20 diambil/dibentuk APA ADANYA "
    "dari output Stage 1 (tidak dihitung ulang, tidak dimodifikasi). Data "
    "bersifat CROSS-SECTIONAL per tahun -- observasi B40/B20 pada satu periode "
    "TIDAK diasumsikan sebagai rumah tangga yang sama dengan observasi B40/B20 "
    "pada periode lain. Ini berlaku baik untuk perbandingan lintas-wilayah "
    "MAUPUN perbandingan temporal (TRAIN vs TEST) -- keduanya tetap "
    "perbandingan antar himpunan rumah tangga yang berbeda, bukan pelacakan "
    "individu (household-level transition)."
)

DEFINITION_NOTE = (
    "CATATAN DEFINISI: kelompok B40 (group40=0) SELALU berisi sekitar 40% "
    "observasi dan kelompok B20 (group20=0) SELALU berisi sekitar 20% "
    "observasi, keduanya secara definisi (dibentuk dari decile rank "
    "pengeluaran, bukan dari p_rentan). B20 adalah subset yang lebih miskin "
    "di dalam B40 (decile 0-1 vs decile 0-3). Karena itu, yang relevan dan "
    "dianalisis di sini BUKAN 'proporsi B40/B20' -- itu sudah tetap oleh "
    "definisi -- melainkan POLA/SEBARAN NILAI p_rentan DI DALAM masing-masing "
    "kelompok tersebut: seberapa mirip atau berbeda sebaran itu antar wilayah "
    "fokus, dan antara TRAIN dan TEST."
)

# mencari nama kolom yang cocok dari daftar kandidar dan melempar error yang menyebutkan kolom apa saja yang tersedia jika tdk ditemukan.
def _resolve_col(df, candidates, file_label):
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    raise KeyError(
        f"\n\n[ERROR] File '{file_label}' tidak punya salah satu dari kolom {candidates}.\n"
        f"Kolom yang tersedia:\n  {list(df.columns)}"
    )

# mencoba beberapa ekstensi file untuk menemukan file yang ada, melempar error jika tidak ditemukan.
def _find_existing_path(base_path):
    candidates = [
        base_path,
        base_path + ".csv",
        base_path + ".parquet",
        base_path + ".pkl",
        base_path + ".pickle",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"[ERROR] Tidak ditemukan file untuk '{base_path}'. Dicoba: {candidates}"
    )

# memilih fungsi pembaca file yang sesuai berdasarkan ekstensi file, default ke CSV jika tidak dikenali.
def _read_any(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    return pd.read_csv(path)

# mengubah nama wilayah menjadi format nama file/kolom yang aman
def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_stage1(cfg):
    print("\n" + "=" * 78 + "\nLOADING OUTPUT STAGE 1 (logreg_raw) 2022-2025\n" + "=" * 78)
    frames = []
    for year, base_path in cfg["stage1_files"].items():
        path = _find_existing_path(base_path)
        d = _read_any(path)
        d.columns = [str(c).strip() for c in d.columns]

        kab_col = _resolve_col(d, cfg["kab_col_candidates"], path)
        prentan_col = _resolve_col(d, cfg["prentan_col_candidates"], path)

        d = d.rename(columns={
            kab_col: "kode_kab",
            prentan_col: "p_rentan",
        })

        d["kode_kab"] = pd.to_numeric(d["kode_kab"], errors="coerce")
        d["p_rentan"] = pd.to_numeric(d["p_rentan"], errors="coerce")

        # group40 & group20 TIDAK tersedia langsung pada output Stage 1.
        # Bentuk dari decile aktual yang sudah tersedia.
        if "decile" not in d.columns:
            raise KeyError(
                f"[ERROR] File '{path}' tidak memiliki kolom 'decile', "
                "sehingga group40/group20 tidak dapat dibentuk."
            )

        d["decile"] = pd.to_numeric(d["decile"], errors="coerce")

        # B40 aktual:
        # decile 0, 1, 2, 3 -> group40 = 0
        # decile 4, 5, 6, 7, 8, 9 -> group40 = 1
        d["group40"] = np.where(d["decile"] <= 3, 0, 1)

        # B20 aktual:
        # decile 0, 1 -> group20 = 0
        # decile 2, 3, 4, 5, 6, 7, 8, 9 -> group20 = 1
        d["group20"] = np.where(d["decile"] <= 1, 0, 1)
        d["year"] = int(year)

        print(f"  year={year}: rows={len(d):,} (file={path})")
        frames.append(d[["kode_kab", "year", "p_rentan", "group40", "group20"]])

    df = pd.concat(frames, ignore_index=True)
    df["period"] = np.where(
        df["year"].isin(cfg["train_years"]), cfg["train_label"],
        np.where(df["year"] == cfg["test_year"], cfg["test_label"], "OTHER")
    )
    df = df[df["period"] != "OTHER"].copy()
    print(f"  TOTAL rows (4 tahun digabung) = {len(df):,}")

    # Validasi rentang p_rentan terhadap asumsi [0,1] (probabilitas logreg).
    pr_min, pr_max = df["p_rentan"].min(), df["p_rentan"].max()
    lo_cfg, hi_cfg = cfg["prentan_range"]
    if pr_min < lo_cfg - 1e-6 or pr_max > hi_cfg + 1e-6:
        print(f"  [PERINGATAN] p_rentan observasi berada di luar rentang asumsi "
              f"{cfg['prentan_range']}: aktual min={pr_min:.6f}, max={pr_max:.6f}. "
              f"Overlap coefficient memakai bin tetap {cfg['prentan_range']} -- "
              f"observasi di luar rentang ini TIDAK ikut terhitung dalam OVL "
              f"(dilaporkan terpisah).")
    else:
        print(f"  p_rentan aktual: min={pr_min:.6f}, max={pr_max:.6f} "
              f"(konsisten dengan asumsi rentang probabilitas {cfg['prentan_range']}).")

    return df


def filter_focus(df, cfg):
    focus_kabs = list(cfg["focus"].values())
    focus_names = {v: k for k, v in cfg["focus"].items()}
    sub = df[df["kode_kab"].isin(focus_kabs)].copy()
    sub["nama"] = sub["kode_kab"].map(focus_names)

    print("\n  Jumlah observasi per kab x period x group40 (setelah filter fokus):")
    print(sub.groupby(["nama", "period", "group40"]).size().to_string())

    for name, kab in cfg["focus"].items():
        if kab not in sub["kode_kab"].unique():
            print(f"  [PERINGATAN] kode_kab={kab} ({name}) tidak ditemukan di data Stage 1.")

    return sub

# menghitung statistik deskripsi dari p_rentan itu sendiri
def describe_prentan(vals, min_n):
    vals = pd.Series(vals).dropna()
    n = len(vals)
    if n < min_n:
        return {
            "n": n, "mean": np.nan, "median": np.nan, "sd": np.nan,
            "P25": np.nan, "P75": np.nan, "IQR": np.nan,
            "note": f"n={n} < ambang minimum ({min_n}), statistik tidak dihitung penuh.",
        }
    p25, p75 = vals.quantile(0.25), vals.quantile(0.75)
    return {
        "n": n,
        "mean": vals.mean(),
        "median": vals.median(),
        "sd": vals.std(ddof=1) if n > 1 else np.nan,
        "P25": p25,
        "P75": p75,
        "IQR": p75 - p25,
        "note": "",
    }


def compute_group40_summary(df, cfg):
    """N + statistik deskriptif p_rentan untuk SETIAP kombinasi
    kab x period x group40 (mencakup group40=0 DAN group40=1, untuk konteks).
    Generik untuk berapa pun jumlah wilayah fokus."""
    rows = []
    for (name, period, g40), sub in df.groupby(["nama", "period", "group40"]):
        stats = describe_prentan(sub["p_rentan"], cfg["min_n"])
        rows.append({
            "nama": name, "kode_kab": sub["kode_kab"].iloc[0],
            "period": period, "group40": int(g40),
            **stats,
        })
    out = pd.DataFrame(rows).sort_values(["nama", "period", "group40"]).reset_index(drop=True)
    return out


def compute_group20_summary(df, cfg):
    """N + statistik deskriptif p_rentan untuk SETIAP kombinasi
    kab x period x group20 (mencakup group20=0 DAN group20=1, untuk konteks).
    Generik untuk berapa pun jumlah wilayah fokus."""
    rows = []
    for (name, period, g20), sub in df.groupby(["nama", "period", "group20"]):
        stats = describe_prentan(sub["p_rentan"], cfg["min_n"])
        rows.append({
            "nama": name, "kode_kab": sub["kode_kab"].iloc[0],
            "period": period, "group20": int(g20),
            **stats,
        })
    out = pd.DataFrame(rows).sort_values(["nama", "period", "group20"]).reset_index(drop=True)
    return out


def histogram_overlap_coefficient(vals_a, vals_b, bin_edges):
    """
    Overlapping Coefficient (OVL) berbasis histogram BERSAMA dengan bin_edges
    yang SUDAH DITENTUKAN DI LUAR fungsi ini (tetap/global untuk semua
    perbandingan) -- supaya OVL antar perbandingan (cross-region, temporal,
    B40 maupun B20) benar-benar bisa dibandingkan apple-to-apple.

    OVL = sum( min(density_a, density_b) * lebar_bin ).
    Berkisar 0 (distribusi sama sekali terpisah, dalam rentang bin yang
    dipakai) sampai ~1 (distribusi identik). Ukuran deskriptif sederhana,
    BUKAN uji hipotesis formal.
    """
    a = pd.Series(vals_a).dropna().values
    b = pd.Series(vals_b).dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan

    dens_a, _ = np.histogram(a, bins=bin_edges, density=True)
    dens_b, _ = np.histogram(b, bins=bin_edges, density=True)
    bin_width = bin_edges[1] - bin_edges[0]

    ovl = float(np.sum(np.minimum(dens_a, dens_b)) * bin_width)
    ovl = max(0.0, min(1.0, ovl))
    return ovl


def pairwise_comparison(vals_a, vals_b, bin_edges, min_n):
    """Satu perbandingan berpasangan: OVL (histogram global) + KS statistic
    (scipy.stats.ks_2samp, ukuran pemisahan independen dari OVL) + n masing2.
    Dipakai untuk B40 maupun B20 (grup mana yang dibandingkan ditentukan oleh
    caller lewat filter data sebelum vals_a/vals_b dikirim ke sini)."""
    a = pd.Series(vals_a).dropna()
    b = pd.Series(vals_b).dropna()
    n_a, n_b = len(a), len(b)

    if n_a < min_n or n_b < min_n:
        return {
            "n_a": n_a, "n_b": n_b,
            "overlap_coefficient": np.nan,
            "KS_stat": np.nan, "KS_pvalue": np.nan,
            "note": f"n tidak cukup (min={min_n}) untuk hitung overlap/KS.",
        }

    ovl = histogram_overlap_coefficient(a, b, bin_edges)
    ks_stat, ks_p = ks_2samp(a, b)
    return {
        "n_a": n_a, "n_b": n_b,
        "overlap_coefficient": ovl,
        "KS_stat": ks_stat, "KS_pvalue": ks_p,
        "note": "",
    }


def compute_all_group_comparisons(df, cfg, group_col, group_value, glabel):
    """
    Versi GENERIK dari perbandingan berpasangan (dipakai untuk B40 dan B20).
    group_col: 'group40' atau 'group20'.
    group_value: nilai yang menandai kelompok "termiskin" (0 untuk B40/B20
    keduanya, karena definisi group40=0 -> B40 dan group20=0 -> B20).
    glabel: label pendek untuk ditampilkan/dipakai di nama kolom, mis. 'B40'
    atau 'B20'.

    Membentuk SEMUA perbandingan berpasangan pada p_rentan, secara otomatis
    dari cfg['focus'] -- tidak perlu diubah manual ketika wilayah fokus
    ditambah/dikurangi:

      1. cross_region_<PERIOD>_<A>_vs_<B> : untuk setiap pasangan wilayah,
         dalam TRAIN dan dalam TEST secara terpisah (kombinasi C(n,2) x 2 periode).
      2. temporal_<REGION> : TRAIN vs TEST untuk masing-masing wilayah.

    Semua memakai bin_edges GLOBAL TETAP (dari cfg['prentan_range'] &
    cfg['hist_bins']) supaya OVL antar semua perbandingan ini (termasuk lintas
    B40 vs B20 bila dibandingkan manual dari kedua tabel) bisa dibandingkan
    langsung.
    """
    regions = cfg["focus"]  # nama -> kode_kab
    gslug = _slug(glabel)
    lo, hi = cfg["prentan_range"]
    bin_edges = np.linspace(lo, hi, cfg["hist_bins"] + 1)

    def grp_vals(kab, period):
        return df[(df["kode_kab"] == kab) & (df["period"] == period)
                  & (df[group_col] == group_value)]["p_rentan"]

    # Kumpulkan nilai untuk setiap (wilayah, periode) sekali saja.
    raw_vals = {}
    for name, kab in regions.items():
        raw_vals[(name, cfg["train_label"])] = grp_vals(kab, cfg["train_label"])
        raw_vals[(name, cfg["test_label"])] = grp_vals(kab, cfg["test_label"])

    rows = []

    # 1) Cross-region: semua pasangan wilayah, dalam TRAIN dan dalam TEST.
    #    region_a/region_b/period disimpan sebagai kolom terpisah (bukan
    #    hasil parsing string "label") supaya baris TRAIN dan TEST untuk
    #    pasangan wilayah yang sama bisa dicocokkan dengan andal saat
    #    dibuat pivot di build_report.
    for period, period_txt in [(cfg["train_label"], "TRAIN 2022-2024"),
                                (cfg["test_label"], "TEST 2025")]:
        for name_a, name_b in combinations(regions.keys(), 2):
            a_vals = raw_vals[(name_a, period)]
            b_vals = raw_vals[(name_b, period)]
            r = pairwise_comparison(a_vals, b_vals, bin_edges, cfg["min_n"])
            rows.append({
                "comparison": f"{gslug}_cross_region_{period}_{_slug(name_a)}_vs_{_slug(name_b)}",
                "label": f"{name_a} vs {name_b}, {period_txt}",
                "group": glabel,
                "type": "cross_region",
                "region_a": name_a, "region_b": name_b, "period": period,
                "kelompok_a": f"{name_a} ({period_txt})",
                "kelompok_b": f"{name_b} ({period_txt})",
                "n_a": r["n_a"], "n_b": r["n_b"],
                "overlap_coefficient": r["overlap_coefficient"],
                "KS_stat": r["KS_stat"], "KS_pvalue": r["KS_pvalue"],
                "note": r["note"],
            })

    # 2) Temporal: TRAIN vs TEST untuk masing-masing wilayah.
    for name in regions:
        a_vals = raw_vals[(name, cfg["train_label"])]
        b_vals = raw_vals[(name, cfg["test_label"])]
        r = pairwise_comparison(a_vals, b_vals, bin_edges, cfg["min_n"])
        rows.append({
            "comparison": f"{gslug}_temporal_{_slug(name)}",
            "label": f"{name}: TRAIN 2022-2024 vs TEST 2025",
            "group": glabel,
            "type": "temporal",
            "region_a": name, "region_b": name, "period": None,
            "kelompok_a": f"{name} (TRAIN)",
            "kelompok_b": f"{name} (TEST)",
            "n_a": r["n_a"], "n_b": r["n_b"],
            "overlap_coefficient": r["overlap_coefficient"],
            "KS_stat": r["KS_stat"], "KS_pvalue": r["KS_pvalue"],
            "note": r["note"],
        })

    comparisons_df = pd.DataFrame(rows)
    return comparisons_df, raw_vals, bin_edges


def make_group_figures(raw_vals, bin_edges, cfg, gslug, glabel):
    """Versi GENERIK pembuat figure (dipakai untuk B40 dan B20).
    gslug dipakai sebagai prefix nama file (mis. 'b40', 'b20') supaya
    figure B40 dan B20 tidak saling menimpa."""
    figdir = os.path.join(cfg["output"], "figures")
    os.makedirs(figdir, exist_ok=True)
    regions = cfg["focus"]

    def plot_pair(vals_a, vals_b, label_a, label_b, title, fname):
        a = pd.Series(vals_a).dropna()
        b = pd.Series(vals_b).dropna()
        if a.empty and b.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        if len(a):
            ax.hist(a, bins=bin_edges, alpha=0.5, density=True, label=f"{label_a} (n={len(a)})")
        if len(b):
            ax.hist(b, bins=bin_edges, alpha=0.5, density=True, label=f"{label_b} (n={len(b)})")
        ax.set_title(title)
        ax.set_xlabel("p_rentan")
        ax.set_ylabel("Densitas")
        ax.set_xlim(bin_edges[0], bin_edges[-1])
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, fname), dpi=150)
        plt.close(fig)
        print(f"  figure tersimpan: {fname}")

    # 1) Cross-region, dalam TRAIN dan dalam TEST, untuk setiap pasangan wilayah.
    for period, period_txt, tag in [(cfg["train_label"], "TRAIN 2022-2024", "train"),
                                     (cfg["test_label"], "TEST 2025", "test")]:
        for name_a, name_b in combinations(regions.keys(), 2):
            plot_pair(
                raw_vals[(name_a, period)], raw_vals[(name_b, period)],
                name_a, name_b,
                f"Distribusi p_rentan pada {glabel} -- {period_txt}\n{name_a} vs {name_b}",
                f"prentan_{gslug}_cross_region_{tag}_{_slug(name_a)}_vs_{_slug(name_b)}.png",
            )

    # 2) Temporal per wilayah.
    for name in regions:
        plot_pair(
            raw_vals[(name, cfg["train_label"])], raw_vals[(name, cfg["test_label"])],
            "TRAIN 2022-2024", "TEST 2025",
            f"{name} -- Distribusi p_rentan pada {glabel}\nTRAIN vs TEST",
            f"prentan_{gslug}_temporal_{_slug(name)}.png",
        )

    # 3) Ringkasan gabungan: semua wilayah x periode dalam satu gambar
    #    (step histogram, warna per wilayah, garis solid=TRAIN / putus-putus=TEST).
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = cm.get_cmap("tab10", max(len(regions), 1))
    for i, name in enumerate(regions):
        color = palette(i)
        for period, ls in [(cfg["train_label"], "-"), (cfg["test_label"], "--")]:
            vals = pd.Series(raw_vals[(name, period)]).dropna()
            if vals.empty:
                continue
            period_short = "TRAIN" if period == cfg["train_label"] else "TEST"
            ax.hist(vals, bins=bin_edges, density=True, histtype="step",
                    linewidth=2, linestyle=ls, color=color,
                    label=f"{name} {period_short} (n={len(vals)})")
    ax.set_title(f"Ringkasan gabungan: distribusi p_rentan pada {glabel}\n"
                  "Semua wilayah fokus, TRAIN vs TEST")
    ax.set_xlabel("p_rentan")
    ax.set_ylabel("Densitas")
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"prentan_{gslug}_ringkasan_gabungan.png"), dpi=150)
    plt.close(fig)
    print(f"  figure tersimpan: prentan_{gslug}_ringkasan_gabungan.png")


# ======================================================================
# VISUALISASI GAYA ZENN (KDE manual pakai scipy.stats.gaussian_kde)
# ----------------------------------------------------------------------
# Menggantikan versi lama yang memakai seaborn.kdeplot(fill=True).
# HANYA gaya visual yang berubah -- tidak ada perhitungan statistik, tidak
# ada resampling, dan tidak ada modifikasi data. KDE dihitung langsung dari
# p_rentan (kolom asli, apa adanya) untuk kelompok B40/B20 (=0) di setiap
# wilayah x periode, persis sama seperti versi seaborn sebelumnya.
# ======================================================================

COVSHIFT_COLOR_TRAIN = "#4C72B0"
COVSHIFT_COLOR_TEST = "#DD8452"


def _covshift_kde_curve(vals, lo=0.0, hi=1.0, n_points=500, bw_adjust=1.0):
    """KDE manual pakai scipy.stats.gaussian_kde, dievaluasi pada grid tetap
    [lo, hi]. bw_adjust mengalikan bandwidth Scott's rule bawaan scipy --
    dipakai SAMA untuk Train dan Test dalam satu panel supaya bentuk kurva
    tetap dibandingkan secara wajar (bukan salah satu dibuat lebih halus)."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    kde = gaussian_kde(vals)
    kde.set_bandwidth(kde.factor * bw_adjust)
    xs = np.linspace(lo, hi, n_points)
    ys = kde(xs)
    return xs, ys


def _covshift_style_axis(ax, lo, hi):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("0.3")
    ax.spines["bottom"].set_color("0.3")
    ax.grid(True, color="0.88", linewidth=0.7, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_xlim(lo, hi)
    if (lo, hi) == (0.0, 1.0):
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def _covshift_plot_panel(ax, vals_train, vals_test, title, lo, hi, bw_adjust):
    xs_tr, ys_tr = _covshift_kde_curve(vals_train, lo=lo, hi=hi, bw_adjust=bw_adjust)
    xs_te, ys_te = _covshift_kde_curve(vals_test, lo=lo, hi=hi, bw_adjust=bw_adjust)

    # Fill dulu (tanpa edge/border), baru garis kurva di atasnya, supaya
    # garis tetap terlihat jelas dan overlap antar area tetap tampak.
    ax.fill_between(xs_tr, ys_tr, color=COVSHIFT_COLOR_TRAIN, alpha=0.30, linewidth=0)
    ax.fill_between(xs_te, ys_te, color=COVSHIFT_COLOR_TEST, alpha=0.30, linewidth=0)
    ax.plot(xs_tr, ys_tr, color=COVSHIFT_COLOR_TRAIN, linewidth=2.2, solid_capstyle="round")
    ax.plot(xs_te, ys_te, color=COVSHIFT_COLOR_TEST, linewidth=2.2, solid_capstyle="round")

    ymax = max(ys_tr.max(), ys_te.max())
    ax.set_ylim(0, ymax * 1.15)

    ax.set_title(title, fontsize=16, pad=10, color="0.15")
    ax.set_xlabel("p_rentan", fontsize=12.5)
    ax.set_ylabel("Density", fontsize=12.5)
    ax.tick_params(labelsize=10.5, colors="0.25")
    _covshift_style_axis(ax, lo, hi)

    handles = [
        Line2D([0], [0], color=COVSHIFT_COLOR_TRAIN, lw=2.2,
               label=f"Train Data (n={len(vals_train)})"),
        Line2D([0], [0], color=COVSHIFT_COLOR_TEST, lw=2.2,
               label=f"Test Data (n={len(vals_test)})"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=10.5)


def make_covariate_shift_figures(df, cfg, bw_adjust=None):
    """
    Density plot gaya Zenn (KDE manual scipy.stats.gaussian_kde, filled,
    tanpa histogram/bin) seperti
    https://zenn.dev/yuto_mo/articles/1f6478e0d7ae1d.
    Satu figure per wilayah (sama seperti versi lama), plus SATU figure
    tambahan berisi 2 panel berdampingan (Kudus & Kota Tegal) untuk
    perbandingan langsung di skripsi.

    Warna: Train = #4C72B0 (biru), Test = #DD8452 (oranye) -- persis
    referensi Zenn. Style: fill alpha=0.30 tanpa edge, garis solid
    linewidth=2.2, background putih, spine atas & kanan dihilangkan, grid
    tipis abu-abu muda. Tidak ada anotasi KS/OVL/mean di dalam gambar --
    statistik itu tetap dijelaskan di teks skripsi, bukan di sini.

    bw_adjust: pengali bandwidth KDE (>1 = kurva lebih halus). Kalau None,
    dipakai cfg["covshift_bw_adjust"] (default 1.0). Dipakai SAMA untuk
    Train & Test pada setiap panel/wilayah.
    """
    if bw_adjust is None:
        bw_adjust = cfg.get("covshift_bw_adjust", 1.0)

    figdir = os.path.join(cfg["output"], "figures")
    os.makedirs(figdir, exist_ok=True)

    # (wilayah, kolom group, label) -- sesuai kebutuhan: Kudus & Yogya pakai
    # B40, Tegal//Blora pakai B20.
    targets = [
        ("Kudus", "group40", "B40"),
        ("Kota Tegal", "group40", "B40"),
        ("Kota Magelang", "group20", "B20"),
    ]

    lo, hi = cfg["prentan_range"]

    print("\n" + "=" * 78 + "\nMEMBUAT VISUALISASI GAYA ZENN (KDE manual, per wilayah)\n" + "=" * 78)

    vals_cache = {}  # (name, glabel) -> (vals_train, vals_test), dipakai ulang utk figure gabungan

    for name, group_col, glabel in targets:
        kab = cfg["focus"][name]
        vals_train = df[(df["kode_kab"] == kab) & (df["period"] == cfg["train_label"])
                         & (df[group_col] == 0)]["p_rentan"].dropna()
        vals_test = df[(df["kode_kab"] == kab) & (df["period"] == cfg["test_label"])
                        & (df[group_col] == 0)]["p_rentan"].dropna()

        if len(vals_train) < cfg["min_n"] or len(vals_test) < cfg["min_n"]:
            print(f"  [SKIP covshift] {name} ({glabel}): n tidak cukup "
                  f"(train={len(vals_train)}, test={len(vals_test)})")
            continue

        vals_cache[(name, glabel)] = (vals_train, vals_test)

        fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
        title = f"Density Plot of p_rentan -- {name} ({glabel})\nTRAIN 2022-2024 vs TEST 2025"
        _covshift_plot_panel(ax, vals_train.values, vals_test.values, title, lo, hi, bw_adjust)

        fig.tight_layout()
        fname = f"covshift_{_slug(glabel)}_{_slug(name)}.png"
        # facecolor="white": pastikan background PNG bersih putih (bukan
        # transparan) -- supaya tidak muncul bingkai gelap saat dibuka di
        # viewer/app bertema gelap. bbox_inches="tight": buang whitespace
        # berlebih di tepi.
        fig.savefig(os.path.join(figdir, fname), dpi=150, facecolor="white",
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  figure tersimpan: {fname}")

    # Figure tambahan: 2 panel berdampingan, Kudus vs Kota Tegal (B40),
    # untuk perbandingan langsung dalam satu gambar di skripsi.
    pair = [("Kudus", "(a) Kabupaten Kudus"), ("Kota Tegal", "(b) Kota Tegal")]
    if all((name, "B40") in vals_cache for name, _ in pair):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="white")
        for (name, title), ax in zip(pair, axes):
            vals_train, vals_test = vals_cache[(name, "B40")]
            _covshift_plot_panel(ax, vals_train.values, vals_test.values, title, lo, hi, bw_adjust)
        fig.tight_layout()
        fname = "covshift_zenn_panel_kudus_kota_tegal.png"
        fig.savefig(os.path.join(figdir, fname), dpi=200, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print(f"  figure tersimpan: {fname}")


def fmt(x, spec=",.4f"):
    return format(x, spec) if pd.notna(x) else "n/a"


def _interpret_ovl(ovl):
    if pd.isna(ovl):
        return "tidak dapat dievaluasi (data tidak cukup)."
    if ovl >= 0.7:
        return "SANGAT bertumpuk (overlap tinggi) -- kedua kelompok kurang terpisah pada nilai p_rentan-nya."
    if ovl >= 0.4:
        return "bertumpuk SEBAGIAN (overlap sedang) -- pemisahan moderat."
    return "relatif TERPISAH (overlap rendah) -- kedua kelompok cenderung berbeda pada nilai p_rentan-nya."


def _build_descriptive_section(summary_df, cfg, section_letter, glabel, group_col):
    lines = []
    lines.append(f"{section_letter}. STATISTIK DESKRIPTIF p_rentan (SEMUA {group_col}, per kab x periode)")
    lines.append("-" * 78)
    for _, r in summary_df.iterrows():
        g_val = int(r[group_col])
        g_desc = f"{glabel} ({group_col}=0)" if g_val == 0 else f"Non-{glabel} ({group_col}=1)"
        period_txt = "TRAIN 2022-2024" if r["period"] == cfg["train_label"] else "TEST 2025"
        lines.append(f"  {r['nama']} | {period_txt} | {g_desc}")
        lines.append(f"      n={int(r['n'])}, mean={fmt(r['mean'])}, median={fmt(r['median'])}, "
                      f"sd={fmt(r['sd'])}, P25={fmt(r['P25'])}, P75={fmt(r['P75'])}, IQR={fmt(r['IQR'])}")
        if r["note"]:
            lines.append(f"      [CATATAN] {r['note']}")
    lines.append("")
    return lines


def _build_comparison_section(comparisons_df, cfg, section_letter_b, section_letter_c, glabel):
    """Membangun bagian B (perbandingan berpasangan) dan C (ringkasan
    lintas-wilayah) untuk SATU kelompok (glabel='B40' atau 'B20'). Generik
    supaya bisa dipakai ulang untuk B40 dan B20 tanpa duplikasi logika."""
    lines = []

    lines.append(f"{section_letter_b}. PERBANDINGAN BERPASANGAN PADA {glabel}")
    lines.append("-" * 78)
    lines.append("  Semua OVL memakai bin histogram GLOBAL TETAP pada rentang "
                  f"{cfg['prentan_range']} ({cfg['hist_bins']} bin) -- sehingga seluruh "
                  "OVL di bawah ini bisa dibandingkan langsung satu sama lain, baik "
                  "cross-region maupun temporal.")
    lines.append("  OVL mendekati 1 -> distribusi SANGAT bertumpuk (kurang terpisah).")
    lines.append("  OVL mendekati 0 -> distribusi relatif TERPISAH.")
    lines.append("  KS statistic (0-1) adalah ukuran pemisahan independen kedua -- KS makin "
                  "besar berarti dua distribusi makin terpisah (arah berlawanan dari OVL).")
    lines.append("")
    for _, r in comparisons_df.iterrows():
        lines.append(f"  [{r['comparison']}] {r['label']}")
        lines.append(f"      n_a({r['kelompok_a']})={int(r['n_a'])}, "
                      f"n_b({r['kelompok_b']})={int(r['n_b'])}")
        lines.append(f"      overlap_coefficient={fmt(r['overlap_coefficient'])} "
                      f"-> {_interpret_ovl(r['overlap_coefficient'])}")
        lines.append(f"      KS_stat={fmt(r['KS_stat'])}, KS_pvalue={fmt(r['KS_pvalue'],'.4g')}")
        if r["note"]:
            lines.append(f"      [CATATAN] {r['note']}")
        lines.append("")

    lines.append(f"{section_letter_c}. RINGKASAN LINTAS-WILAYAH -- {glabel}")
    lines.append("-" * 78)
    lines.append(f"  Pertanyaan: apakah distribusi p_rentan pada {glabel} memiliki pola "
                  "overlap/pemisahan yang berbeda antar wilayah fokus, dan apakah pola "
                  "tersebut stabil dari TRAIN 2022-2024 ke TEST 2025 -- yang secara "
                  "deskriptif dapat membantu memahami mengapa efek Two-Stage terhadap "
                  "IE/EE/RMSE dapat berbeda antar wilayah?")
    lines.append("")

    valid = comparisons_df.dropna(subset=["overlap_coefficient"])
    cross = valid[valid["type"] == "cross_region"]
    temporal = valid[valid["type"] == "temporal"]

    if cross.empty and temporal.empty:
        lines.append("  Tidak dapat dijawab -- tidak ada perbandingan dengan data yang cukup "
                      "(lihat CATATAN pada bagian di atas).")
    else:
        # Ringkas stabilitas temporal per wilayah, urut dari yang paling TIDAK stabil.
        if not temporal.empty:
            temp_sorted = temporal.sort_values("overlap_coefficient")
            lines.append(f"  Stabilitas temporal p_rentan {glabel} per wilayah (TRAIN vs TEST), "
                          "diurutkan dari OVL terendah (paling tidak stabil) ke tertinggi:")
            for _, r in temp_sorted.iterrows():
                nama = r["region_a"]
                lines.append(f"      {nama}: OVL={fmt(r['overlap_coefficient'])} "
                              f"({_interpret_ovl(r['overlap_coefficient'])})")
            lines.append("")

        # Ringkas pasangan cross-region dengan perubahan OVL terbesar dari TRAIN ke TEST.
        # Dipivot langsung dari kolom region_a/region_b/period (bukan parsing string),
        # supaya baris TRAIN dan TEST untuk pasangan wilayah yang sama pasti tercocokkan.
        if not cross.empty:
            cross = cross.copy()
            cross["period_tag"] = np.where(cross["period"] == cfg["train_label"], "TRAIN", "TEST")
            pivot = cross.pivot_table(index=["region_a", "region_b"], columns="period_tag",
                                       values="overlap_coefficient", aggfunc="first")
            if "TRAIN" in pivot.columns and "TEST" in pivot.columns:
                pivot["delta"] = pivot["TEST"] - pivot["TRAIN"]
                pivot = pivot.dropna(subset=["delta"]).sort_values("delta", key=lambda s: s.abs(),
                                                                     ascending=False)
                lines.append("  Perubahan OVL cross-region dari TRAIN ke TEST per pasangan wilayah "
                              "(diurutkan dari perubahan absolut terbesar):")
                for (region_a, region_b), row in pivot.iterrows():
                    lines.append(f"      {region_a} vs {region_b}: OVL_TRAIN={fmt(row['TRAIN'])}, "
                                  f"OVL_TEST={fmt(row['TEST'])}, delta={row['delta']:+.4f}")
                lines.append("")

        lines.append(f"  Kesimpulan (deskriptif, {glabel}): pola di atas KONSISTEN DENGAN adanya "
                      f"perbedaan struktural pada sebaran p_rentan {glabel} antar wilayah yang "
                      "muncul atau berubah pada 2025 untuk wilayah/pasangan dengan delta OVL "
                      "besar atau OVL temporal rendah, dan SECARA DESKRIPTIF DAPAT MEMBANTU "
                      "MENJELASKAN mengapa penambahan p_rentan sebagai fitur (Two-Stage) "
                      "berpotensi berdampak berbeda terhadap IE/EE/RMSE antar wilayah tersebut: "
                      f"jika p_rentan di dalam {glabel} lebih terpisah/bervariasi di suatu "
                      "wilayah, fitur ini berpotensi menyumbang lebih banyak daya pembeda "
                      "tambahan bagi model di wilayah tersebut, dan sebaliknya jika lebih "
                      "bertumpuk. Wilayah/pasangan dengan delta OVL kecil dan OVL temporal "
                      "tinggi menunjukkan pola yang relatif stabil, sehingga faktor ini "
                      "kemungkinan BUKAN kandidat utama untuk menjelaskan perbedaan performa "
                      "di sana (perlu dicek faktor lain).")

    lines.append("")
    return lines


def build_report(summary40_df, comparisons40_df, summary20_df, comparisons20_df, cfg):
    regions = cfg["focus"]
    region_list_txt = ", ".join(f"{name} ({kab})" for name, kab in regions.items())

    lines = []
    lines.append("DIAGNOSTIK SEDERHANA (v5, multi-region, B40 & B20) -- p_rentan")
    lines.append(f"Wilayah fokus: {region_list_txt}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(CAUTION_LANGUAGE_NOTE)
    lines.append("")
    lines.append(DATA_NOTE)
    lines.append("")
    lines.append(DEFINITION_NOTE)
    lines.append("")

    lines.append("=" * 78)
    lines.append("BAGIAN B40 (group40=0)")
    lines.append("=" * 78)
    lines.append("")
    lines += _build_descriptive_section(summary40_df, cfg, "A1", "B40", "group40")
    lines += _build_comparison_section(comparisons40_df, cfg, "B1", "C1", "B40")

    lines.append("=" * 78)
    lines.append("BAGIAN B20 (group20=0)")
    lines.append("=" * 78)
    lines.append("")
    lines += _build_descriptive_section(summary20_df, cfg, "A2", "B20", "group20")
    lines += _build_comparison_section(comparisons20_df, cfg, "B2", "C2", "B20")

    lines.append("  PENTING -- BATASAN INTERPRETASI (berlaku untuk B40 MAUPUN B20 di atas):")
    lines.append("    - Temuan di atas TIDAK menyatakan bahwa overlap/pemisahan p_rentan "
                  "MENYEBABKAN perbedaan performa targeting (IE/EE/RMSE); hubungan yang "
                  "dinyatakan bersifat deskriptif/asosiatif (\"konsisten dengan\", \"dapat "
                  "membantu menjelaskan\"), bukan kausal.")
    lines.append("    - B40 (group40=0) dan B20 (group20=0) selalu berisi proporsi observasi "
                  "yang tetap secara definisi -- yang dibandingkan di sini adalah SEBARAN "
                  "NILAI p_rentan di dalam masing-masing kelompok, BUKAN perubahan proporsi.")
    lines.append("    - B20 adalah SUBSET dari B40 (decile 0-1 di dalam decile 0-3), sehingga "
                  "kedua bagian laporan ini TIDAK independen satu sama lain -- perbedaan/"
                  "kesamaan pola antara bagian B40 dan B20 untuk wilayah yang sama dapat "
                  "membantu melihat apakah pola B40 didorong terutama oleh subset termiskin "
                  "(B20) atau tersebar merata di seluruh B40.")
    lines.append("    - RMSE/IE/EE TIDAK dihitung ulang di script ini; kaitan dengan performa "
                  "Two-Stage harus dikonfirmasi dari analisis performa model yang TERPISAH "
                  "(mis. report_xgb_two_stage_*.csv).")
    lines.append("")

    return "\n".join(lines)


def main():
    cfg = CONFIG
    out = cfg["output"]
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)

    region_list_txt = ", ".join(f"{name} ({kab})" for name, kab in cfg["focus"].items())

    print("=" * 78)
    print("DIAGNOSTIK SEDERHANA (v5, multi-region, B40 & B20) -- p_rentan")
    print(f"Wilayah fokus: {region_list_txt}")
    print("=" * 78)
    print("\n" + CAUTION_LANGUAGE_NOTE + "\n")
    print(DATA_NOTE + "\n")
    print(DEFINITION_NOTE + "\n")

    df_all = load_stage1(cfg)
    df_focus = filter_focus(df_all, cfg)

    # ---------------- B40 ----------------
    print("\n" + "=" * 78 + "\nSTATISTIK DESKRIPTIF p_rentan PER KAB x PERIOD x GROUP40\n" + "=" * 78)
    summary40_df = compute_group40_summary(df_focus, cfg)
    summary40_df.to_csv(os.path.join(out, "prentan_group40_summary.csv"), index=False)
    print(summary40_df.to_string(index=False))
    print(f"\n  prentan_group40_summary.csv tersimpan, {len(summary40_df)} baris.")

    print("\n" + "=" * 78 + "\nPERBANDINGAN BERPASANGAN B40 (cross-region & temporal)\n" + "=" * 78)
    comparisons40_df, raw_vals40, bin_edges = compute_all_group_comparisons(
        df_focus, cfg, group_col="group40", group_value=0, glabel="B40"
    )
    comparisons40_df.to_csv(os.path.join(out, "prentan_b40_pairwise_comparisons.csv"), index=False)
    print(comparisons40_df.to_string(index=False))
    print(f"\n  prentan_b40_pairwise_comparisons.csv tersimpan, {len(comparisons40_df)} baris.")

    print("\n" + "=" * 78 + "\nMEMBUAT VISUALISASI B40 (histogram p_rentan)\n" + "=" * 78)
    make_group_figures(raw_vals40, bin_edges, cfg, gslug="b40", glabel="B40")

    # ---------------- B20 ----------------
    print("\n" + "=" * 78 + "\nSTATISTIK DESKRIPTIF p_rentan PER KAB x PERIOD x GROUP20\n" + "=" * 78)
    summary20_df = compute_group20_summary(df_focus, cfg)
    summary20_df.to_csv(os.path.join(out, "prentan_group20_summary.csv"), index=False)
    print(summary20_df.to_string(index=False))
    print(f"\n  prentan_group20_summary.csv tersimpan, {len(summary20_df)} baris.")

    print("\n" + "=" * 78 + "\nPERBANDINGAN BERPASANGAN B20 (cross-region & temporal)\n" + "=" * 78)
    comparisons20_df, raw_vals20, bin_edges20 = compute_all_group_comparisons(
        df_focus, cfg, group_col="group20", group_value=0, glabel="B20"
    )
    comparisons20_df.to_csv(os.path.join(out, "prentan_b20_pairwise_comparisons.csv"), index=False)
    print(comparisons20_df.to_string(index=False))
    print(f"\n  prentan_b20_pairwise_comparisons.csv tersimpan, {len(comparisons20_df)} baris.")

    print("\n" + "=" * 78 + "\nMEMBUAT VISUALISASI B20 (histogram p_rentan)\n" + "=" * 78)
    make_group_figures(raw_vals20, bin_edges20, cfg, gslug="b20", glabel="B20")

    # ---------------- Covariate shift style (KDE gaya Zenn, per wilayah) ----------------
    make_covariate_shift_figures(df_focus, cfg)

    # ---------------- Laporan gabungan ----------------
    print("\n" + "=" * 78 + "\nMENULIS LAPORAN\n" + "=" * 78)
    report_text = build_report(summary40_df, comparisons40_df, summary20_df, comparisons20_df, cfg)
    with open(os.path.join(out, "prentan_group40_group20_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\n" + "=" * 78)
    print(report_text)
    print("=" * 78)
    print(f"\nDONE\nOutput folder: {out}/")


if __name__ == "__main__":
    main()
