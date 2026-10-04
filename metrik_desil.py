"""
===============================================================================
ANALISIS TAMBAHAN TESIS PMT — KARAKTERISTIK RUMAH TANGGA B40 TERLEMPAR
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = {
    "xgb": {
        "input_file": "output_two_stage_logreg_xgb_raw/master_xgb_two_stage_2025_timedef.csv",
        "color": "#ED7D31",
        "label": "XGBoost",
    },
    "rf": {
        "input_file": "output_two_stage_logreg_rf_raw/master_rf_two_stage_2025_timedef.csv",  # <- ganti sesuai nama file RF Anda
        "color": "#4472C4",
        "label": "Random Forest",
    },
}

WILAYAH = {
    3319: "Kudus",
    3376: "Kota Tegal",
}
KAB_COL = "kode_kab_new"

# Warna bubble matrix sekarang PER WILAYAH (kab), bukan per algoritma.
# Kudus akan selalu bervarna sama untuk XGBoost maupun Random Forest,
# begitu juga Kota Magelang. Silakan ganti hex code sesuai selera.
WILAYAH_COLORS = {
    3319: "#ED7D31",   # Kudus
    3376: "#4472C4",   # Kota Tegal
}

# group40      : 0 = aktual B40      , 1 = aktual bukan B40
# group40_pred : 0 = diprediksi B40  , 1 = diprediksi bukan B40
GROUP40_COL = "group40"
GROUP40_PRED_COL = "group40_pred"

# Kolom decile SUDAH tersedia langsung di master file, dalam nilai INTERNAL
# 0-9 (bukan 1-10). TIDAK dihitung ulang dari pcexp/pcexp_pred, dan TIDAK
# memakai make_decile(). Semua analisis/crosstab/CSV tetap memakai nilai
# internal 0-9 ini apa adanya. Konversi ke label 1-10 HANYA dilakukan saat
# menggambar bubble figure (lihat plot_decile_bubble), bukan di dataframe.
TRUE_DECILE_COL = "decile"
PRED_DECILE_OUT_COL = "decile_pred"

HH_ID_COL = "hhid"          # <- ganti sesuai nama kolom ID rumah tangga
YEAR_LABEL = "2025"
OUTPUT_DIR = "analysis_error_targeting"

# Threshold Cohen untuk menyaring "relatif sama" (konvensi umum: small = 0.2)
EFFECT_SIZE_THRESHOLD = 0.2

# 97 BASE_PREDICTORS
BASE_PREDICTORS = [
    'h_hhcount','h_nmale','h_nfemale','h_nage04','h_nage519','h_nage2064','h_nage65up',
    'h_ngrad_sd','h_ngrad_smp','h_ngrad_sma','h_ngrad_d','h_ngrad_s','h_notgrad',
    'h_sec1_stat1','h_sec1_stat2','h_sec1_stat3','h_sec1_stat4','h_sec1_stat5','h_sec1_stat6',
    'h_sec2_stat1','h_sec2_stat2','h_sec2_stat3','h_sec2_stat4','h_sec2_stat5','h_sec2_stat6',
    'h_sec3_stat1','h_sec3_stat2','h_sec3_stat3','h_sec3_stat4','h_sec3_stat5','h_sec3_stat6',
    'h_neverschool','h_stillschool','h_notschool',
    'h_house1','h_house2','h_house3','h_house4',
    'h_floor1','h_floor2','h_floor3','h_floor4',
    'h_wall1','h_wall2','h_wall3','h_wall4',
    'h_roof1','h_roof2','h_roof3','h_roof4',
    'h_dwater1','h_dwater2','h_dwater3','h_dwater4','h_dwater5',
    'h_lighting1','h_lighting2','h_lighting3',
    'h_epower1','h_epower2','h_epower3',
    'h_cookingfuel1','h_cookingfuel2','h_cookingfuel3','h_cookingfuel4','h_cookingfuel5',
    'h_toiltype1','h_toiltype2','h_toiltype3','h_toiltype4','h_toiltype5','h_toiltype6',
    'h_septic1','h_septic2','h_septic3','h_septic4','h_septic5',
    'h_asset_lpg5kg','h_asset_fridge','h_asset_ac','h_asset_wheater',
    'h_asset_phone','h_asset_computer','h_asset_jewelry','h_asset_motorcycle',
    'h_asset_boat','h_asset_motorboat','h_asset_car','h_asset_tv',
    'h_asset_smartphone','h_asset_land',
    'h_lnluaslantai',
    'h_nfamily','h_notmarried','h_married','h_divorced','h_widowed'
]

# Variabel numerik: mencakup jumlah ART per ruta, proporsi pendidikan,
# dan ukuran kontinu. Variabel get_wide() lain di bawah ini adalah hitungan
# ART per ruta, bukan dummy 0/1.
NUMERIC_PREDICTORS = [
    'h_hhcount', 'h_nmale', 'h_nfemale', 'h_nage04', 'h_nage519', 'h_nage2064',
    'h_nage65up', 'h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma', 'h_ngrad_d', 'h_ngrad_s',
    'h_notgrad',
    'h_neverschool', 'h_stillschool', 'h_notschool',
    *[f'h_sec{s}_stat{st}' for s in [1, 2, 3] for st in range(1, 7)],
    'h_notmarried', 'h_married', 'h_divorced', 'h_widowed',
    'h_lnluaslantai', 'h_nfamily',
]

# Status sekolah dibandingkan sebagai proporsi per ruta (skala 0–100),
# memakai denominator yang sama dengan eda_kemiskinan.py.
SCHOOL_STATUS_COLS = ['h_neverschool', 'h_stillschool', 'h_notschool']

THEMA_MAP = {
    'Demografi': ['h_hhcount','h_nmale','h_nfemale','h_nage04','h_nage519','h_nage2064','h_nage65up'],
    'Pendidikan': ['h_ngrad_sd','h_ngrad_smp','h_ngrad_sma','h_ngrad_d','h_ngrad_s','h_notgrad',
                   'h_neverschool','h_stillschool','h_notschool'],
    'Ketenagakerjaan': ['h_sec1_stat1','h_sec1_stat2','h_sec1_stat3','h_sec1_stat4','h_sec1_stat5','h_sec1_stat6',
                        'h_sec2_stat1','h_sec2_stat2','h_sec2_stat3','h_sec2_stat4','h_sec2_stat5','h_sec2_stat6',
                        'h_sec3_stat1','h_sec3_stat2','h_sec3_stat3','h_sec3_stat4','h_sec3_stat5','h_sec3_stat6'],
    'Kondisi rumah': ['h_house1','h_house2','h_house3','h_house4','h_floor1','h_floor2','h_floor3','h_floor4',
                       'h_wall1','h_wall2','h_wall3','h_wall4','h_roof1','h_roof2','h_roof3','h_roof4',
                       'h_lnluaslantai'],
    'Fasilitas dasar': ['h_dwater1','h_dwater2','h_dwater3','h_dwater4','h_dwater5',
                         'h_lighting1','h_lighting2','h_lighting3','h_epower1','h_epower2','h_epower3',
                         'h_cookingfuel1','h_cookingfuel2','h_cookingfuel3','h_cookingfuel4','h_cookingfuel5',
                         'h_toiltype1','h_toiltype2','h_toiltype3','h_toiltype4','h_toiltype5','h_toiltype6',
                         'h_septic1','h_septic2','h_septic3','h_septic4','h_septic5'],
    'Kepemilikan aset': ['h_asset_lpg5kg','h_asset_fridge','h_asset_ac','h_asset_wheater','h_asset_phone',
                          'h_asset_computer','h_asset_jewelry','h_asset_motorcycle','h_asset_boat',
                          'h_asset_motorboat','h_asset_car','h_asset_tv','h_asset_smartphone','h_asset_land'],
    'Karakteristik keluarga': ['h_nfamily','h_notmarried','h_married','h_divorced','h_widowed'],
}
VAR_TO_THEME = {v: k for k, vs in THEMA_MAP.items() for v in vs}

# --- Pengaturan tampilan bubble matrix (ala Fig. 2) — SAMA untuk keempat figure ---
BUBBLE_SCALE = 70          # luas bubble ~ persentase; konstan lintas figure agar comparable
FIGSIZE = (7, 7)           # square, sama seperti contoh paper
AXIS_PAD = 0.6             # perluasan sumbu di luar 1-10 supaya bubble tepi tidak terpotong
BUBBLE_ALPHA = 0.75
BUBBLE_EDGECOLOR = "dimgray"
BUBBLE_EDGEWIDTH = 0.6
FONT_AXIS_LABEL = 10
FONT_TICK = 9
FONT_BUBBLE_TEXT = 7.5
FONT_TITLE = 10.5

def cohens_d(x1: pd.Series, x2: pd.Series) -> float:
    """Standardized Mean Difference (pooled SD) untuk variabel numerik."""
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    v1, v2 = x1.var(ddof=1), x2.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0
    return (x1.mean() - x2.mean()) / pooled_sd


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h untuk selisih dua proporsi (p dalam skala 0-1)."""
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def interpret_direction(effect_size: float, threshold: float = EFFECT_SIZE_THRESHOLD) -> str:
    if pd.isna(effect_size) or abs(effect_size) < threshold:
        return "relatif sama"
    return "lebih tinggi pada B40 terlempar" if effect_size > 0 else "lebih rendah pada B40 terlempar"


def load_and_prepare(model_key: str, cfg: dict) -> pd.DataFrame:
    df = pd.read_csv(cfg["input_file"])

    required = [KAB_COL, GROUP40_COL, GROUP40_PRED_COL, TRUE_DECILE_COL, PRED_DECILE_OUT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"[{model_key}] Kolom wajib tidak ditemukan: {missing}. "
                        f"Pastikan master file punya kolom '{TRUE_DECILE_COL}' dan "
                        f"'{PRED_DECILE_OUT_COL}' (nilai internal 0-9).")

    missing_school = [c for c in SCHOOL_STATUS_COLS if c not in df.columns]
    if missing_school:
        raise KeyError(f"[{model_key}] Kolom status sekolah tidak ditemukan: {missing_school}")

    # Ubah hitungan ART per kategori menjadi proporsi dalam setiap ruta.
    # Rute dengan total status sekolah 0 mendapat proporsi 0, sama seperti EDA.
    school_count = df[SCHOOL_STATUS_COLS].sum(axis=1)
    denominator = school_count.replace(0, np.nan)
    for col in SCHOOL_STATUS_COLS:
        df[col] = (df[col] / denominator * 100).fillna(0.0)

    for col in (TRUE_DECILE_COL, PRED_DECILE_OUT_COL):
        out_of_range = df[~df[col].isin(range(0, 10))]
        if len(out_of_range) > 0:
            bad_values = sorted(out_of_range[col].unique().tolist())
            raise ValueError(
                f"[{model_key}] Kolom '{col}' berisi nilai di luar rentang decile "
                f"internal 0-9: {bad_values}. Periksa master file '{cfg['input_file']}' "
                f"— skrip tidak akan mengubah nilai ini secara diam-diam."
            )

    print(f"[{model_key}] Menggunakan kolom decile internal 0-9 langsung: "
          f"'{TRUE_DECILE_COL}' (true) & '{PRED_DECILE_OUT_COL}' (predicted). "
          f"Tidak ada perhitungan ulang dari pcexp.")
    return df

# fungsi ini untuk menghitung jumlah ruta untuk setiap pasangan

def crosstab_decile(df_wil: pd.DataFrame) -> pd.DataFrame:
    n_total = len(df_wil)
    tab = (
        df_wil.groupby([TRUE_DECILE_COL, PRED_DECILE_OUT_COL])
        .size()
        .reset_index(name="jumlah_ruta")
    )
    tab["persentase_total"] = (tab["jumlah_ruta"] / n_total * 100).round(2)
    return tab, n_total

# fungsi ini membuat bubble matrix untuk satu wilayah (kabupaten/kota) dan satu model (XGB/RF)
def plot_decile_bubble(df_wil: pd.DataFrame, wilayah_label: str, color: str,
                        model_key: str, cfg: dict, out_path: str):
    tab, n_total = crosstab_decile(df_wil)
    tab = tab[tab["jumlah_ruta"] > 0]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # ukuran bubble proporsional terhadap persentase total, dikalikan skala agar terlihat
    sizes = tab["persentase_total"] * BUBBLE_SCALE

    # +1 hanya untuk posisi visual (internal 0-9 -> label figure 1-10)
    x_display = tab[TRUE_DECILE_COL] + 1
    y_display = tab[PRED_DECILE_OUT_COL] + 1

    ax.scatter(
        x_display, y_display,
        s=sizes, alpha=BUBBLE_ALPHA, color=color,
        edgecolors=BUBBLE_EDGECOLOR, linewidths=BUBBLE_EDGEWIDTH, zorder=3,
    )
    for (_, row), x_d, y_d in zip(tab.iterrows(), x_display, y_display):
        ax.annotate(
            f"{row['persentase_total']:.1f}",
            (x_d, y_d),
            ha="center", va="center", fontsize=FONT_BUBBLE_TEXT, zorder=4,
        )

    # Dua garis merah pemisah B40 (1-4) vs bukan B40 (5-10), tepat di 4.5
    ax.axvline(4.5, color="red", linewidth=1.6, zorder=2)
    ax.axhline(4.5, color="red", linewidth=1.6, zorder=2)

    # Sumbu diperluas sedikit di luar 1-10 (AXIS_PAD) supaya bubble di desil
    # 1 dan 10 tidak terpotong, TAPI tick yang ditampilkan hanya 1..10
    # (tidak ada tick 0 atau 11), sesuai gaya Figure 2.
    ax.set_xlim(1 - AXIS_PAD, 10 + AXIS_PAD)
    ax.set_ylim(1 - AXIS_PAD, 10 + AXIS_PAD)
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))
    ax.tick_params(axis="both", labelsize=FONT_TICK, length=3)

    ax.set_xlabel("Desil Aktual", fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("Desil Prediksi", fontsize=FONT_AXIS_LABEL)

    # Tampilan bersih ala paper: tanpa grid, tanpa colorbar/legend,
    # background putih, aspect persegi supaya bubble tidak lonjong.
    ax.grid(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    # Judul singkat, dua baris, tanpa menyebut nama kolom sumber prediksi
    # (supaya tidak terlalu panjang di dalam plot)
    ax.set_title(
        f"Desil Aktual vs Desil Prediksi of Households in {wilayah_label} in {YEAR_LABEL}\n"
        f"using {cfg['label']}",
        fontsize=FONT_TITLE,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    return tab, n_total


# Rumah tangga yang terlempar: aktual B40 (group40=0) tapi diprediksi bukan B40 (group40_pred=1)

def get_terlempar(df_wil: pd.DataFrame) -> pd.DataFrame:
    """group40 == 0 (aktual B40) & group40_pred == 1 (diprediksi bukan B40)."""
    return df_wil[(df_wil[GROUP40_COL] == 0) & (df_wil[GROUP40_PRED_COL] == 1)].copy()


def get_berhasil(df_wil: pd.DataFrame) -> pd.DataFrame:
    """group40 == 0 (aktual B40) & group40_pred == 0 (diprediksi B40) -> pembanding."""
    return df_wil[(df_wil[GROUP40_COL] == 0) & (df_wil[GROUP40_PRED_COL] == 0)].copy()


def error_decile_cells(terlempar: pd.DataFrame) -> pd.DataFrame:
    n_terlempar = len(terlempar)
    tab = (
        terlempar.groupby([TRUE_DECILE_COL, PRED_DECILE_OUT_COL])
        .size()
        .reset_index(name="jumlah_ruta")
        .sort_values("jumlah_ruta", ascending=False)
        .reset_index(drop=True)
    )
    if n_terlempar > 0:
        tab["persentase_dari_seluruh_ruta_terlempar"] = (tab["jumlah_ruta"] / n_terlempar * 100).round(2)
    else:
        tab["persentase_dari_seluruh_ruta_terlempar"] = np.nan
    return tab


# Effect size untuk tiap prediktor, membandingkan rumah tangga B40 terlempar vs B40 berhasil

def compare_characteristics(terlempar: pd.DataFrame, berhasil: pd.DataFrame,
                             predictors: list) -> pd.DataFrame:
    rows = []
    n_t, n_b = len(terlempar), len(berhasil)

    for var in predictors:
        if var not in terlempar.columns:
            continue

        is_numeric = var in NUMERIC_PREDICTORS

        if is_numeric:
            t_vals, b_vals = terlempar[var].dropna(), berhasil[var].dropna()
            t_mean, b_mean = t_vals.mean(), b_vals.mean()
            t_median, b_median = t_vals.median(), b_vals.median()
            t_iqr = t_vals.quantile(0.75) - t_vals.quantile(0.25)
            b_iqr = b_vals.quantile(0.75) - b_vals.quantile(0.25)
            diff = t_mean - b_mean
            effect = cohens_d(t_vals, b_vals)

            rows.append({
                "predictor": var, "type": "numeric", "tema": VAR_TO_THEME.get(var, "Lainnya"),
                "n_terlempar": n_t, "n_B40_berhasil": n_b,
                "nilai_terlempar": round(t_mean, 3), "nilai_B40_berhasil": round(b_mean, 3),
                "median_terlempar": round(t_median, 3), "median_B40_berhasil": round(b_median, 3),
                "IQR_terlempar": round(t_iqr, 3), "IQR_B40_berhasil": round(b_iqr, 3),
                "difference": round(diff, 3),
                "effect_size": round(effect, 3) if not np.isnan(effect) else np.nan,
                "effect_size_type": "Cohen's d (SMD)",
            })
        else:
            t_p = terlempar[var].mean()   # proporsi 0-1
            b_p = berhasil[var].mean()
            diff_pp = (t_p - b_p) * 100    # percentage point
            effect = cohens_h(t_p, b_p)

            rows.append({
                "predictor": var, "type": "binary", "tema": VAR_TO_THEME.get(var, "Lainnya"),
                "n_terlempar": n_t, "n_B40_berhasil": n_b,
                "nilai_terlempar": round(t_p * 100, 2), "nilai_B40_berhasil": round(b_p * 100, 2),
                "median_terlempar": np.nan, "median_B40_berhasil": np.nan,
                "IQR_terlempar": np.nan, "IQR_B40_berhasil": np.nan,
                "difference": round(diff_pp, 2),
                "effect_size": round(effect, 3),
                "effect_size_type": "Cohen's h",
            })

    result = pd.DataFrame(rows)
    result["interpretation_direction"] = result["effect_size"].apply(interpret_direction)
    result = result.sort_values(
        by="effect_size", key=lambda s: s.abs(), ascending=False, na_position="last"
    ).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    # urutan kolom akhir sesuai permintaan (+ kolom tambahan median/IQR yang berguna)
    col_order = [
        "rank", "predictor", "type", "tema",
        "n_terlempar", "n_B40_berhasil",
        "nilai_terlempar", "nilai_B40_berhasil",
        "median_terlempar", "median_B40_berhasil",
        "IQR_terlempar", "IQR_B40_berhasil",
        "difference", "effect_size", "effect_size_type", "interpretation_direction",
    ]
    return result[col_order]


def build_theme_summary(char_table: pd.DataFrame, top_n_list=(10, 15)) -> pd.DataFrame:
    """Ringkasan tema: jumlah prediktor tiap tema di Top-N, dan prediktor
    terkuat (|effect_size| terbesar) per tema di ANTARA SELURUH prediktor
    (bukan hanya yang masuk Top-N)."""
    rows = []
    strongest_per_theme = (
        char_table.dropna(subset=["effect_size"])
        .assign(abs_effect=lambda d: d["effect_size"].abs())
        .sort_values("abs_effect", ascending=False)
        .groupby("tema", as_index=False)
        .first()
        .set_index("tema")
    )

    for tema in THEMA_MAP.keys():
        row = {"tema": tema}
        for n in top_n_list:
            top_n = char_table[char_table["rank"] <= n]
            row[f"jumlah_di_top{n}"] = int((top_n["tema"] == tema).sum())
        if tema in strongest_per_theme.index:
            s = strongest_per_theme.loc[tema]
            row["prediktor_terkuat"] = s["predictor"]
            row["effect_size_terkuat"] = s["effect_size"]
            row["arah_terkuat"] = s["interpretation_direction"]
        else:
            row["prediktor_terkuat"] = None
            row["effect_size_terkuat"] = np.nan
            row["arah_terkuat"] = None
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(f"jumlah_di_top{top_n_list[-1]}", ascending=False)
    return summary.reset_index(drop=True)


def build_profile_narrative(char_table: pd.DataFrame, top_n: int = 15,
                             threshold: float = EFFECT_SIZE_THRESHOLD, max_items: int = 4) -> str:
    """Profil karakteristik otomatis berbasis Top-N, HANYA menyebut prediktor
    dengan |effect_size| >= threshold, bahasa non-kausal."""
    top = char_table[char_table["rank"] <= top_n]
    top = top[top["effect_size"].abs() >= threshold]

    lebih_tinggi = top[top["interpretation_direction"] == "lebih tinggi pada B40 terlempar"]["predictor"].tolist()
    lebih_rendah = top[top["interpretation_direction"] == "lebih rendah pada B40 terlempar"]["predictor"].tolist()

    if not lebih_tinggi and not lebih_rendah:
        return ("Pada Top {0} prediktor, tidak ditemukan perbedaan karakteristik yang "
                "cukup besar (|effect size| >= {1}) antara rumah tangga B40 terlempar "
                "dan B40 yang berhasil ditangkap model.").format(top_n, threshold)

    parts = []
    if lebih_tinggi:
        items = ", ".join(lebih_tinggi[:max_items])
        parts.append(f"lebih sering/lebih tinggi memiliki karakteristik {items}")
    if lebih_rendah:
        items = ", ".join(lebih_rendah[:max_items])
        parts.append(f"karakteristik {items} menunjukkan proporsi/nilai yang lebih rendah")

    return ("Dibandingkan rumah tangga B40 yang berhasil ditangkap model, kelompok "
            "rumah tangga B40 yang terlempar " + "; sementara ".join(parts) + ".")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for model_key, cfg in MODELS.items():
        df = load_and_prepare(model_key, cfg)

        for kab_code, wilayah_label in WILAYAH.items():
            fname_tag = f"{model_key}_{wilayah_label.lower().replace(' ', '_')}"
            print(f"\n================================================")
            print(f"{wilayah_label.upper()} — {cfg['label'].upper()}")
            print(f"================================================")

            df_wil = df[df[KAB_COL] == kab_code].copy()
            n_wil = len(df_wil)
            print(f"Jumlah seluruh rumah tangga: {n_wil}")

            # --- A. Bubble matrix (seluruh rumah tangga) ---
            # Warna diambil PER WILAYAH (WILAYAH_COLORS), bukan lagi dari
            # cfg["color"] (per algoritma) — supaya Kudus konsisten satu
            # warna dan Kota Tegal konsisten satu warna, lintas model.
            wilayah_color = WILAYAH_COLORS[kab_code]
            img_path = os.path.join(OUTPUT_DIR, f"decile_bubble_{fname_tag}.png")
            tab_decile, _ = plot_decile_bubble(
                df_wil, wilayah_label, wilayah_color, model_key, cfg, img_path
            )
            tab_decile.to_csv(
                os.path.join(OUTPUT_DIR, f"true_predicted_decile_{fname_tag}.csv"), index=False
            )
            total_obs_in_bubble = tab_decile["jumlah_ruta"].sum()
            assert total_obs_in_bubble == n_wil, "Jumlah bubble matrix tidak cocok dengan jumlah observasi!"

            # --- B. Rumah tangga terlempar & pembanding ---
            terlempar = get_terlempar(df_wil)
            berhasil = get_berhasil(df_wil)
            print(f"Jumlah B40 terlempar         : {len(terlempar)}")
            print(f"Jumlah B40 berhasil (pembanding): {len(berhasil)}")

            cells = error_decile_cells(terlempar)
            cells.to_csv(os.path.join(OUTPUT_DIR, f"error_decile_cells_{fname_tag}.csv"), index=False)

            if len(cells) > 0:
                print("\n5 kombinasi True Decile -> Predicted Decile terbesar (di antara RT terlempar):")
                for _, row in cells.head(5).iterrows():
                    print(f"  True {row[TRUE_DECILE_COL]} -> Pred {row[PRED_DECILE_OUT_COL]} : "
                          f"{row['jumlah_ruta']} ruta ({row['persentase_dari_seluruh_ruta_terlempar']}%)")
            else:
                print("\nTidak ada rumah tangga B40 yang terlempar di wilayah ini.")

            # --- C. Karakteristik (effect size comparable) ---
            if len(terlempar) > 0 and len(berhasil) > 0:
                char_table = compare_characteristics(terlempar, berhasil, BASE_PREDICTORS)
                char_table.to_csv(
                    os.path.join(OUTPUT_DIR, f"characteristics_terlempar_{fname_tag}.csv"), index=False
                )

                top15 = char_table[char_table["rank"] <= 15]
                top15.to_csv(
                    os.path.join(OUTPUT_DIR, f"top_characteristics_{fname_tag}.csv"), index=False
                )

                theme_summary = build_theme_summary(char_table, top_n_list=(10, 15))
                theme_summary.to_csv(
                    os.path.join(OUTPUT_DIR, f"theme_summary_{fname_tag}.csv"), index=False
                )

                print("\nTop 10 karakteristik paling membedakan (ranking by effect size):")
                for _, row in char_table[char_table["rank"] <= 10].iterrows():
                    print(f"  {row['rank']}. {row['predictor']} ({row['tema']}) — "
                          f"effect_size={row['effect_size']} [{row['effect_size_type']}] — "
                          f"{row['interpretation_direction']}")

                tema_top15_counts = theme_summary.set_index("tema")["jumlah_di_top15"]
                tema_top15_counts = tema_top15_counts[tema_top15_counts > 0].sort_values(ascending=False)
                print("\nTema paling banyak muncul di Top 15:")
                print(tema_top15_counts.to_string())

                narrative = build_profile_narrative(char_table, top_n=15)
                print(f"\nProfil karakteristik (Top 15, |effect size| >= {EFFECT_SIZE_THRESHOLD}):")
                print(f"  {narrative}")
            else:
                print("\nAnalisis karakteristik dilewati (RT terlempar atau RT pembanding kosong).")

    print(f"\nSemua output tersimpan di folder: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
