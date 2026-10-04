"""
SHAP ANALYSIS — SINGLE-STAGE vs TWO-STAGE PMT MODEL
Bab 4: setelah kinerja agregat & analisis per-kabupaten

"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore")

BASE_DIR   = "."
OUTPUT_DIR = os.path.join(BASE_DIR, "output_shap")
os.makedirs(os.path.join(OUTPUT_DIR, "global"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "local"), exist_ok=True)

KABUPATEN = {
    "magelang":   {"kode": 3371, "label": "Kota Magelang", "pola": "konsisten terbantu"},
    "tegal":      {"kode": 3376, "label": "Kota Tegal",   "pola": "bergantung algoritma"},
    "kudus":      {"kode": 3319, "label": "Kabupaten Kudus", "pola": "tidak terbantu"},
    "temanggung": {"kode": 3323, "label": "Kabupaten Temanggung", "pola": "tidak terbantu"},
}

# analisis shap global dibuat terpisah untuk setiap kabupaten x algoritma
# kata "global" berarti merangkum kontribusi dari banyak ruta di satu kabupaten 

ANALYSIS_MATRIX = {
    "magelang":   ["rf", "xgb"],
    "tegal":      ["rf", "xgb"],
    "kudus":      ["rf", "xgb"],
    "temanggung": ["rf", "xgb"],
}

# Local SHAP (studi kasus status-change B40) hanya untuk dua wilayah kunci
# Bab 4.2.7 — lihat catatan v6 poin 4 di atas.
NEWLY_CORRECT_KABS = ["tegal"]   # two-stage MEMPERBAIKI targeting B40
NEWLY_WRONG_KABS   = ["kudus"]        # two-stage MEMPERBURUK targeting B40

# Jumlah RT (1-3) yang diambil per kabupaten x algoritma x tipe kasus,
# setelah difilter oleh status-change group40 (lihat select_newly_correct /
# select_newly_wrong).
TOP_N_STATUS_CHANGE = 3

STAGE1_DEFAULT = "logreg"
CALIB_DEFAULT  = "raw"

SINGLE_MODEL_PATH = "output_single_stage_{algo}_timedef/models/{algo}_single_{kode}_timedef.pkl"
SINGLE_MASTER_CSV = "output_single_stage_{algo}_timedef/master_{algo}_single_2025_timedef.csv"

TWO_MODEL_PATH = "output_two_stage_{stage1}_{stage2}_{calib}/models/{stage2}_two_stage_{kode}_timedef.pkl"
TWO_MASTER_CSV = "output_two_stage_{stage1}_{stage2}_{calib}/master_{stage2}_two_stage_2025_timedef.csv"

KAB_COL_CANDIDATES = ["kode_kab_new", "kode_kab_x", "kode_kab_y", "kode_kab"]

ID_COL     = "URUT"
TARGET_COL = "lpcexp"    # model dilatih di skala LOG (dikonfirmasi dari sanity check run sebelumnya)
P_RENTAN_COL = "p_rentan"

# Kolom grup B40/B20 GROUND TRUTH yang SUDAH ADA di master CSV (tidak
# dihitung ulang, tidak ada threshold baru) — dipakai sebagai info status
# "sebenarnya" (group40) dan info tambahan (group20).
GROUP_COLS = ["group40", "group20"]

# Kolom status B40 MENURUT PREDIKSI model, sudah ada di masing-masing master
# CSV (single-stage & two-stage masing-masing punya kolomnya sendiri, nilainya
# beda karena prediksinya beda). Di-rename setelah merge menjadi
# "group40_single" / "group40_two" — TIDAK dihitung ulang, TIDAK ada
# threshold baru.
GROUP40_PRED_COL = "group40_pred"

BASE_PREDICTORS = [
    'h_hhcount', 'h_nmale', 'h_nfemale', 'h_nage04', 'h_nage519', 'h_nage2064', 'h_nage65up',
    'h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma', 'h_ngrad_d', 'h_ngrad_s', 'h_notgrad',
    'h_sec1_stat1', 'h_sec1_stat2', 'h_sec1_stat3', 'h_sec1_stat4', 'h_sec1_stat5', 'h_sec1_stat6',
    'h_sec2_stat1', 'h_sec2_stat2', 'h_sec2_stat3', 'h_sec2_stat4', 'h_sec2_stat5', 'h_sec2_stat6',
    'h_sec3_stat1', 'h_sec3_stat2', 'h_sec3_stat3', 'h_sec3_stat4', 'h_sec3_stat5', 'h_sec3_stat6',
    'h_neverschool', 'h_stillschool', 'h_notschool', 'h_house1', 'h_house2', 'h_house3', 'h_house4',
    'h_floor1', 'h_floor2', 'h_floor3', 'h_floor4', 'h_wall1', 'h_wall2', 'h_wall3', 'h_wall4',
    'h_roof1', 'h_roof2', 'h_roof3', 'h_roof4', 'h_dwater1', 'h_dwater2', 'h_dwater3', 'h_dwater4', 'h_dwater5',
    'h_lighting1', 'h_lighting2', 'h_lighting3', 'h_epower1', 'h_epower2', 'h_epower3',
    'h_cookingfuel1', 'h_cookingfuel2', 'h_cookingfuel3', 'h_cookingfuel4', 'h_cookingfuel5',
    'h_toiltype1', 'h_toiltype2', 'h_toiltype3', 'h_toiltype4', 'h_toiltype5', 'h_toiltype6',
    'h_septic1', 'h_septic2', 'h_septic3', 'h_septic4', 'h_septic5',
    'h_asset_lpg5kg', 'h_asset_fridge', 'h_asset_ac', 'h_asset_wheater', 'h_asset_phone', 'h_asset_computer',
    'h_asset_jewelry', 'h_asset_motorcycle', 'h_asset_boat', 'h_asset_motorboat', 'h_asset_car', 'h_asset_tv',
    'h_asset_smartphone', 'h_asset_land', 'h_lnluaslantai', 'h_nfamily', 'h_notmarried', 'h_married',
    'h_divorced', 'h_widowed',
]
FEATURE_COLS_SINGLE = BASE_PREDICTORS
FEATURE_COLS_TWO    = BASE_PREDICTORS + [P_RENTAN_COL]

BACKGROUND_SAMPLE_N = 150

FORCE_MAX_FEATURES = 9
# figure dibikin lebih tinggi & lebar supaya ada ruang buat judul di atas,
# label true value di bawah, dan label fitur yang dirotasi
FORCE_FIGSIZE = (20, 6)
# rotasi label fitur (derajat). 0 = horizontal (numpuk kalau barnya tipis).
# 45 = miring, jadi tetap kebaca walau ada fitur dominan yang bikin fitur
# lain punya bar sempit.
FORCE_TEXT_ROTATION = 45

DECISION_MAX_FEATURES = 15          # top-N fitur per rumah tangga (bukan global)
DECISION_FIGSIZE = (18, 9)
DECISION_FORCE_INCLUDE = [P_RENTAN_COL]   # p_rentan tetap ditampilkan walau di luar top-N

def detect_kab_col(df):
    for col in KAB_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(f"Tidak ada kolom kabupaten yang cocok di antara {KAB_COL_CANDIDATES}")

# memastikan label aktual tersedia
def check_group_cols(df):
    missing = [c for c in GROUP_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            "Kolom group40/group20 tidak ditemukan pada master CSV sehingga "
            "status B40/B20 ground-truth tidak dapat dibaca."
        )

# memastikan hasil klasifikasi tersedia
def check_group_pred_col(df):
    if GROUP40_PRED_COL not in df.columns:
        raise KeyError(
            f"Kolom '{GROUP40_PRED_COL}' tidak ditemukan pada master CSV sehingga "
            "perubahan status group40 (NEWLY CORRECT/NEWLY WRONG) tidak dapat dideteksi."
        )

# membaca csv dan menyaring kabupaten
def load_and_filter(csv_path, kode):
    # membaca seluruh CSV ke dalam df
    df = pd.read_csv(csv_path)
    # mencari nama kolom kabupatennya
    kab_col = detect_kab_col(df)
    # memilih baris yang kodenya cocok
    df_kab = df[df[kab_col] == kode].copy().reset_index(drop=True)
    if df_kab.empty:
        raise ValueError(f"Tidak ada baris untuk kode kabupaten {kode} di kolom '{kab_col}' ({csv_path})")
    return df_kab

# memisahkan model pohon dari pipeline
def unwrap_estimator(model):
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
        preprocessor = model[:-1] if len(model.steps) > 1 else None
        return estimator, preprocessor
    return model, None

# membuka file model
def load_model_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    return joblib.load(path)


def get_background_sample(X, n=BACKGROUND_SAMPLE_N, random_state=42):
    if len(X) <= n:
        return X
    return X.sample(n=n, random_state=random_state)

# menghitung nilai shap
def compute_shap_values(model, X):
    # 1. memisahkan model pohon dari prepocessingnya.
    estimator, preprocessor = unwrap_estimator(model)
    if preprocessor is not None:
        # 2. mengubah fitur dengan cara yang sama ketika model digunakan
        X_used = preprocessor.transform(X)
        if not isinstance(X_used, pd.DataFrame):
            X_used = pd.DataFrame(X_used, columns=X.columns, index=X.index)
    else:
        X_used = X
    # 3. menghitung nilai SHAP dengan TreeExplainer
    explainer = shap.TreeExplainer(estimator)
    # 4. menghitung kontribusi SHAP tiap fitur untuk tiap ruta pada X_used
    shap_values = explainer(X_used)
    # 5. fungsi mengembalikan alat explainer dan hasil shap_values
    return explainer, shap_values

# analisis shap global
# skrip global SHAP terpisah untuk setiap kabupaten/kota x algoritma
# kata global berarti merangkum kontribusi fitur dari banyak ruta di satu kabupaten

def plot_beeswarm(shap_values, title, save_path, max_display=18):
    plt.figure(figsize=(8, 7))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")

# fungsi ini membuat tabel untuk membadingkan seberapa besar kontribusi rata-rata tiap fitur pada model 
# single-stage dan two-stage. Fungsi ini tidak menghitung SHAP lagi; merangkum nilai SHAP yang sudah dihitung sblmnya 
def feature_shift_table(shap_values_single, cols_single, shap_values_two, cols_two):
    imp_single = pd.Series(
        # mengambil nilai SHAP untuk setiap ruta dan fitur 
        # np.abs -> mengubah nilai menjadi positif. Jadi yang dihitung besar kontribusinya tanpa 
        # melihat apakah fitur mendorong prediksi naik atau turun
        # mean(axis=0) -> menghitung rata-rata kontribusi fitur di semua ruta
        # index=cols_single -> memberi label kolom sesuai nama fitur
        np.abs(shap_values_single.values).mean(axis=0), index=cols_single, name="mean_abs_shap_single"
    )
    imp_two = pd.Series(
        np.abs(shap_values_two.values).mean(axis=0), index=cols_two, name="mean_abs_shap_two"
    )
    # kedua ringkasan disusun berdampingan. nama fitur menjadi indeks baris, jadi nilai untuk fitur yang sama bs dibandingkan
    table = pd.concat([imp_single, imp_two], axis=1)
    table["delta"] = table["mean_abs_shap_two"] - table["mean_abs_shap_single"]
    table["delta_pct"] = table["delta"] / table["mean_abs_shap_single"].replace(0, np.nan) * 100
    return table.sort_values("delta", ascending=False)

# analisis global shap untuk setiap pasangan kabupaten x algoritma
def run_global_shap(kab_key, algo, stage1=STAGE1_DEFAULT, calib=CALIB_DEFAULT):
    # algor : algoritma tahap 2, stage 1: algoritma tahap 1 default logreg, calib: penanda jenis keluaran tahap 1 
    # mengambil data untuk satu kabupaten
    info = KABUPATEN[kab_key]
    kode = info["kode"]
    print(f"\n=== GLOBAL SHAP — {info['label']} ({algo.upper()}) ===")

    # single-stage
    model_single = load_model_file(SINGLE_MODEL_PATH.format(algo=algo, kode=kode))
    df_single = load_and_filter(SINGLE_MASTER_CSV.format(algo=algo), kode)
    X_single = df_single[FEATURE_COLS_SINGLE].copy()
    # dari csv master, kode memilih kolom fitur yang memang diharapkan model single-stage
    # model membuat prediksi untuk semua ruta Tegal dan hasilnya disimpan di kolom pred
    df_single["pred"] = model_single.predict(X_single)   # prediksi utk SEMUA RT di kab ini

    # two-stage
    model_two = load_model_file(TWO_MODEL_PATH.format(stage1=stage1, stage2=algo, calib=calib, kode=kode))
    df_two = load_and_filter(TWO_MASTER_CSV.format(stage1=stage1, stage2=algo, calib=calib), kode)
    X_two = df_two[FEATURE_COLS_TWO].copy()
    # kode memilih fitur yang diperlukan model two-stage. Daftar ini mencakup fitur ruta dan p_rentan
    df_two["pred"] = model_two.predict(X_two)

    X_single_bg = get_background_sample(X_single)
    X_two_bg = get_background_sample(X_two)
    # 1. kirim model single-stage dan semua fitur single-stage ke SHAP 
    # 2. kirim model two-stage dan semua fitur two-stage ke SHAP 
    # 3. hitung kontribusi tiap fitur untuk ruta-ruta di kedua data itu
    explainer_single, shap_single = compute_shap_values(model_single, X_single_bg)
    explainer_two, shap_two = compute_shap_values(model_two, X_two_bg)

    out_dir = os.path.join(OUTPUT_DIR, "global")
    plot_beeswarm(
        shap_single, f"{info['label']} — {algo.upper()} Single-stage",
        os.path.join(out_dir, f"{kab_key}_{algo}_single_beeswarm.png"),
    )
    plot_beeswarm(
        shap_two, f"{info['label']} — {stage1.title()}→{algo.upper()} Two-stage ({calib})",
        os.path.join(out_dir, f"{kab_key}_{algo}_two_beeswarm.png"),
    )

    table = feature_shift_table(shap_single, FEATURE_COLS_SINGLE, shap_two, FEATURE_COLS_TWO)
    table_path = os.path.join(out_dir, f"feature_shift_{kab_key}_{algo}.csv")
    table.to_csv(table_path)
    print(f"  [saved] {table_path}")
    print(table.head(5)[["mean_abs_shap_single", "mean_abs_shap_two", "delta"]])

    return {
        "model_single": model_single, "model_two": model_two,
        "df_single": df_single, "df_two": df_two,
        "X_single": X_single, "X_two": X_two,
        "explainer_single": explainer_single, "explainer_two": explainer_two,
    }


def _merge_true_pred(df_single, df_two, id_col=ID_COL, target_col=TARGET_COL):
    """Gabungkan pred + status group40 single & two-stage untuk RT yang sama
    (key = URUT), lalu hitung improvement.

    - group40, group20  : ground truth, diambil dari df_single saja (nilainya
      sama di df_two untuk URUT yang sama) — TIDAK dihitung ulang.
    - group40_pred di df_single -> di-rename jadi "group40_single"
    - group40_pred di df_two    -> di-rename jadi "group40_two"
      (masing-masing SUDAH ADA di master CSV single/two-stage-nya sendiri,
      TIDAK dihitung ulang, TIDAK ada threshold baru).

    err_single/err_two/improvement dihitung persis seperti versi sebelumnya.
    """
    check_group_cols(df_single)
    check_group_cols(df_two)
    check_group_pred_col(df_single)
    check_group_pred_col(df_two)

    single_cols = df_single[[id_col, target_col, "pred"] + GROUP_COLS + [GROUP40_PRED_COL]].copy()
    single_cols = single_cols.rename(columns={GROUP40_PRED_COL: "group40_single"})

    two_cols = df_two[[id_col, "pred", GROUP40_PRED_COL]].copy()
    two_cols = two_cols.rename(columns={GROUP40_PRED_COL: "group40_two"})

    merged = single_cols.merge(two_cols, on=id_col, suffixes=("_single", "_two"))

    merged["err_single"] = (merged[target_col] - merged["pred_single"]).abs()
    merged["err_two"] = (merged[target_col] - merged["pred_two"]).abs()
    merged["improvement"] = merged["err_single"] - merged["err_two"]  # positif = two-stage lebih baik
    return merged


def select_newly_correct(df_single, df_two):
    """NEWLY CORRECT: RT B40 (group40==0, ground truth) yang AWALNYA salah
    diklasifikasikan oleh single-stage (group40_single==1, IE) lalu BENAR
    setelah two-stage (group40_two==0).

    Filter status-change diterapkan LEBIH DULU; hasilnya baru diurutkan dari
    improvement (perubahan estimasi lokal) terbesar ke terkecil.
    """
    merged = _merge_true_pred(df_single, df_two)
    candidates = merged[
        (merged["group40"] == 0)
        & (merged["group40_single"] == 1)
        & (merged["group40_two"] == 0)
    ]
    return candidates.sort_values("improvement", ascending=False).reset_index(drop=True)


def select_newly_wrong(df_single, df_two):
    """NEWLY WRONG: RT B40 (group40==0, ground truth) yang AWALNYA benar
    diklasifikasikan oleh single-stage (group40_single==0) lalu menjadi salah
    (IE) setelah two-stage (group40_two==1).

    Filter status-change diterapkan LEBIH DULU; hasilnya baru diurutkan dari
    RT yang errornya NAIK paling besar (improvement paling negatif) ke yang
    paling kecil kenaikannya.
    """
    merged = _merge_true_pred(df_single, df_two)
    candidates = merged[
        (merged["group40"] == 0)
        & (merged["group40_single"] == 0)
        & (merged["group40_two"] == 1)
    ]
    return candidates.sort_values("improvement", ascending=True).reset_index(drop=True)


# LOCAL SHAP

def format_feature_value(v):
    """Bulatin nilai fitur biar nggak nampilin semua desimal (kayak
    p_rentan=0.0120313987905232 atau h_lnluaslantai=5.303304908059076).
    Presisi disesuaikan skala nilainya, mirip gaya paper (NTL=668.3,
    NDVI=0.2786, croplands=42.19)."""
    if pd.isna(v):
        return v
    av = abs(v)
    if av >= 100:
        return round(v, 1)
    elif av >= 1:
        return round(v, 2)
    else:
        return round(v, 3)


def condense_shap_row(shap_values_row, max_features=FORCE_MAX_FEATURES):
    """Potong 1 baris SHAP jadi maksimal `max_features` fitur individual +
    1 bar gabungan "N other features" utk sisanya.

    NB: shap punya `contribution_threshold` bawaan buat nyembunyiin fitur
    kecil, tapi itu proporsi dihitung dari total_effect internal yang bisa
    salah hitung (numpy diam-diam nge-cast array [effect, value, nama] jadi
    array string, jadi presisi float-nya bisa kepotong). Makanya di sini kita
    kondensasi manual SEBELUM masuk ke shap.plots.force -> hasilnya pasti
    persis max_features+1 bar, nggak peduli threshold-nya.
    """
    values = np.asarray(shap_values_row.values, dtype=float).flatten()
    n = len(values)

    if shap_values_row.data is not None:
        data_vals = np.asarray(shap_values_row.data, dtype=float).flatten()
    else:
        data_vals = np.full(n, np.nan)

    if shap_values_row.feature_names is not None:
        feature_names = list(shap_values_row.feature_names)
    else:
        feature_names = [f"feature_{i}" for i in range(n)]

    base_value = float(np.asarray(shap_values_row.base_values).flatten()[0])

    if n <= max_features:
        return base_value, values, data_vals, feature_names

    order = np.argsort(-np.abs(values))
    keep_idx = order[:max_features]
    other_idx = order[max_features:]

    kept_values = values[keep_idx]
    kept_data = data_vals[keep_idx]
    kept_names = [feature_names[i] for i in keep_idx]

    other_sum = values[other_idx].sum()
    kept_values = np.append(kept_values, other_sum)
    kept_data = np.append(kept_data, np.nan)
    kept_names = kept_names + [f"{len(other_idx)} other features"]

    return base_value, kept_values, kept_data, kept_names


def plot_force_with_truth(shap_values_row, true_value, pred_value, title, save_path,
                           max_features=FORCE_MAX_FEATURES):
    base_value, values, data_vals, feature_names = condense_shap_row(shap_values_row, max_features)

    # features displayed as "" kalau nan (mis. bar "N other features"),
    # dan DIBULATIN biar nggak nampilin semua desimal float aslinya
    features_display = np.array(
        ["" if pd.isna(v) else format_feature_value(v) for v in data_vals], dtype=object
    )

    shap.plots.force(
        base_value,
        shap_values=values,
        features=features_display,
        feature_names=feature_names,
        matplotlib=True, show=False,
        contribution_threshold=0.0,  # sudah dikondensasi manual, jangan disaring lagi
        figsize=FORCE_FIGSIZE,
        # kalau ada 1 fitur dominan (p_rentan), bar fitur2 kecil jadi tipis
        # banget -> label horizontalnya numpuk. Rotasi label supaya tetap
        # kebaca walau barnya sempit, kayak di force plot 8-10 fitur di paper.
        text_rotation=FORCE_TEXT_ROTATION,
    )
    fig = plt.gcf()
    ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    xrange = xmax - xmin if xmax != xmin else 1.0
    pad = 0.08 * xrange
    new_xmin = min(xmin, true_value - pad)
    new_xmax = max(xmax, true_value + pad)
    ax.set_xlim(new_xmin, new_xmax)

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax != ymin else 1.0
    ax.axvline(true_value, color="green", linestyle="--", linewidth=1.2, zorder=5)
    ax.text(
        true_value, ymin - 0.12 * yrange, f"true {TARGET_COL}: {true_value:.2f}",
        color="green", fontsize=8, ha="center", va="top", clip_on=False,
    )

    # kasih jarak yang jelas antara judul figure dan area plot shap, supaya
    # "higher/lower" + f(x) bawaan shap nggak ketiban judul kita lagi
    fig.subplots_adjust(top=0.68, bottom=0.30)
    fig.suptitle(f"{title}\npredicted={pred_value:.3f}", fontsize=10, y=1.15)
    print(f"  [DEBUG] save_path = {repr(save_path)}")
    # Pastikan path absolut dan folder tersedia
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"  [DEBUG] save_path = {repr(save_path)}")

    try:
        # Buka file secara eksplisit agar Windows tidak bermasalah
        with open(save_path, "wb") as f:
            fig.savefig(
                f,
                format="png",
                dpi=200,
                bbox_inches="tight"
            )

        print(f"  [saved] {save_path}")

    except OSError as e:
        print(f"  [WARNING] savefig gagal dengan bbox_inches='tight': {e}")
        print("  [INFO] Mencoba penyimpanan tanpa bbox_inches='tight'...")

        # fallback
        fig.savefig(
            save_path,
            format="png",
            dpi=200
        )

        print(f"  [saved - fallback] {save_path}")

    finally:
        plt.close(fig)
# mengambil satu baris data rumah tangga tertentu (urut) dari dataframe df, lalu menghitung nilai SHAP untuk baris itu menggunakan explainer. Fungsi ini mengembalikan nilai SHAP, prediksi model, dan nilai target sebenarnya untuk rumah tangga tersebut.
def explain_household(urut, df, X_cols, explainer, id_col=ID_COL):
    row = df[df[id_col] == urut]
    if row.empty:
        raise ValueError(f"URUT {urut} tidak ditemukan di data ini.")
    # mengambil kolom prediktornya
    x_row = row[X_cols]
    # menghitung kontribusi SHAP untuk fitur-fitur pada baris rumah tangga ini
    sv = explainer(x_row)
    return sv[0], row["pred"].values[0], row[TARGET_COL].values[0]


def diagnose_p_rentan_effect(sv_two, pred_two, true_val, p_rentan_col=P_RENTAN_COL):
    """
    Hitung seberapa besar kontribusi SHAP p_rentan terhadap PERUBAHAN
    prediksi two-stage dibanding seandainya p_rentan tidak berkontribusi
    sama sekali — BUKAN pernyataan kausal ("p_rentan menyebabkan X"),
    melainkan dekomposisi aditif SHAP standar (perkiraan kontribusi marjinal
    pada model ini saja).

    Caranya: hitung ulang prediksi SEANDAINYA kontribusi p_rentan
    dihilangkan (pred_without = pred_two - shap_p_rentan), lalu
    bandingkan errornya:
      - err_with_p_rentan    = |true - pred_two|
      - err_without_p_rentan = |true - pred_without|
      - p_rentan_misleading  = True kalau err_with > err_without
        (artinya TANPA kontribusi p_rentan, prediksi model ini justru lebih
        dekat ke nilai sebenarnya -> pada RT ini, kontribusi p_rentan searah
        dengan PERUBAHAN prediksi yang menjauhkan dari true value, bukan
        bukti bahwa p_rentan "menyebabkan" kesalahan secara kausal)

    Return None kalau p_rentan_col tidak ada di feature_names (mis. kalau
    fungsi ini dipanggil untuk explanation single-stage).
    """
    feature_names = list(sv_two.feature_names) if sv_two.feature_names is not None else FEATURE_COLS_TWO
    if p_rentan_col not in feature_names:
        return None

    values = np.asarray(sv_two.values, dtype=float).flatten()
    idx = feature_names.index(p_rentan_col)
    shap_p_rentan = float(values[idx])

    pred_without = pred_two - shap_p_rentan
    err_with = abs(true_val - pred_two)
    err_without = abs(true_val - pred_without)
    misleading = bool(err_with > err_without)

    return {
        "shap_p_rentan": shap_p_rentan,
        "pred_two_without_p_rentan": pred_without,
        "err_with_p_rentan": err_with,
        "err_without_p_rentan": err_without,
        "p_rentan_misleading": misleading,
    }


# ==============================================================================
# 5b. LOCAL DECISION PLOT — BARU v6.1

def _consistency_check_pair(urut, row_single, row_two, sv_single, sv_two,
                             true_single, true_two, id_col=ID_COL, target_col=TARGET_COL):
    """Cek: pastikan pasangan single/two-stage benar-benar merujuk RT yang
    sama sebelum dipakai untuk decision plot. STOP (raise) kalau tidak cocok.
    Tidak melakukan merge ulang apa pun — murni validasi atas hasil
    explain_household() yang sudah dipanggil di run_local_shap_for_household().
    """
    urut_single = row_single[id_col].values[0]
    urut_two = row_two[id_col].values[0]
    if urut_single != urut or urut_two != urut:
        raise ValueError(
            f"[CONSISTENCY CHECK GAGAL] URUT tidak cocok: diminta={urut}, "
            f"single={urut_single}, two={urut_two}"
        )
    if not np.isclose(true_single, true_two, equal_nan=False):
        raise ValueError(
            f"[CONSISTENCY CHECK GAGAL] URUT={urut}: {target_col} single-stage "
            f"({true_single}) != {target_col} two-stage ({true_two})"
        )
    if sv_single.feature_names == sv_two.feature_names:
        raise ValueError(
            f"[CONSISTENCY CHECK GAGAL] URUT={urut}: feature_names SHAP single "
            f"dan two-stage identik — indikasi SHAP two-stage tidak dihitung dari "
            f"model two-stage (harusnya beda karena two-stage punya p_rentan)."
        )


def _select_local_top_features(sv_row, max_features=DECISION_MAX_FEATURES,
                                force_include=None):
    """Ranking TOP-N fitur berdasarkan |SHAP value| RUMAH TANGGA INI SAJA
    (bukan global importance). force_include (mis. p_rentan) selalu disisipkan
    walau tidak masuk top-N secara alami."""
    values = np.asarray(sv_row.values, dtype=float).flatten()
    data_vals = (np.asarray(sv_row.data, dtype=float).flatten()
                 if sv_row.data is not None else np.full(len(values), np.nan))
    feature_names = (list(sv_row.feature_names) if sv_row.feature_names is not None
                      else [f"feature_{i}" for i in range(len(values))])

    order = list(np.argsort(-np.abs(values)))
    top_idx = order[:max_features]

    if force_include:
        for fname in force_include:
            if fname in feature_names:
                fidx = feature_names.index(fname)
                if fidx not in top_idx:
                    top_idx.append(fidx)

    sub_values = values[top_idx]
    sub_data = data_vals[top_idx]
    sub_names = [feature_names[i] for i in top_idx]
    return sub_values, sub_data, sub_names


def plot_local_decision_comparison(sv_single, pred_single, sv_two, pred_two,
                                    title_main, save_path,
                                    max_features=DECISION_MAX_FEATURES,
                                    force_include=DECISION_FORCE_INCLUDE):
    """1 figure, 2 axes berdampingan (kiri=single-stage, kanan=two-stage),
    masing-masing pakai shap.plots.decision (classic SHAP decision plot).

    - Fitur di-ranking per rumah tangga (bukan global SHAP importance).
    - p_rentan dipaksa tampil di panel two-stage walau di luar top-N.
    - base_value, shap_values, prediksi masing-masing HARUS berasal dari
      model-nya sendiri (single -> single, two -> two) — tidak dicampur.
    - Tidak ada klaim kausal pada label; hanya "SHAP contribution".
    """
    base_single = float(np.asarray(sv_single.base_values).flatten()[0])
    base_two = float(np.asarray(sv_two.base_values).flatten()[0])

    vals_s, data_s, names_s = _select_local_top_features(sv_single, max_features, None)
    vals_t, data_t, names_t = _select_local_top_features(sv_two, max_features, force_include)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DECISION_FIGSIZE)

    plt.sca(ax1)
    shap.decision_plot(
        base_single, vals_s.reshape(1, -1), features=data_s.reshape(1, -1),
        feature_names=names_s, feature_order="importance",
        auto_size_plot=False, show=False, title=None,
    )
    ax1 = plt.gca()
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.set_title(f"(a) Single-stage (v1)\nPred = {pred_single:.3f}", fontsize=10)
    ax1.set_xlabel("SHAP contribution to model output", fontsize=9)

    plt.sca(ax2)
    shap.decision_plot(
        base_two, vals_t.reshape(1, -1), features=data_t.reshape(1, -1),
        feature_names=names_t, feature_order="importance",
        auto_size_plot=False, show=False, title=None,
    )
    ax2 = plt.gca()
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.set_title(f"(b) Two-stage (v2)\nPred = {pred_two:.3f}", fontsize=10)
    ax2.set_xlabel("SHAP contribution to model output", fontsize=9)

    fig.suptitle(title_main, fontsize=11, y=1.02)
    fig.tight_layout()

    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {save_path}")


# memanggil explain_household() untuk satu rumah tangga, lalu membuat force plot single-stage & two-stage, plus local decision plot berdampingan (v6.1), 
# dan menjalankan diagnostik p_rentan. Hasil ringkas ditambahkan ke summary_rows.
def run_local_shap_for_household(kab_key, algo, result, urut, pola_label, summary_rows,
                                  case_type, rank=None,
                                  group40=None, group40_single=None, group40_two=None,
                                  group20=None):
    """
    Jalankan force plot single-stage & two-stage untuk 1 rumah tangga, plus
    local decision plot berdampingan (BARU v6.1) dan diagnostik p_rentan
    (diagnose_p_rentan_effect). Hasil ringkas (termasuk status
    group40/group40_single/group40_two dan group20 informasional) ditambahkan
    ke summary_rows.

    case_type: "newly_correct" | "newly_wrong"
    rank     : urutan RT di dalam tipe kasus ini (1..TOP_N_STATUS_CHANGE)
    group40  : status B40 ground truth (0 = termasuk 40% terbawah)
    group40_single, group40_two : status B40 menurut prediksi single-stage /
               two-stage — dipakai untuk menunjukkan PERUBAHAN status
               targeting, bukan untuk perhitungan SHAP/force plot.
    group20  : informasi tambahan saja (tidak dipakai untuk filter Bab 4 ini).
    """
    info = KABUPATEN[kab_key]
    out_dir = os.path.join(OUTPUT_DIR, "local")

    sv_single, pred_single, true_val = explain_household(
        urut, result["df_single"], FEATURE_COLS_SINGLE, result["explainer_single"]
    )
    sv_two, pred_two, true_val_two = explain_household(
        urut, result["df_two"], FEATURE_COLS_TWO, result["explainer_two"]
    )

    # --- BARU v6.1: validasi pasangan single/two-stage sebelum decision plot ---
    row_single = result["df_single"][result["df_single"][ID_COL] == urut]
    row_two = result["df_two"][result["df_two"][ID_COL] == urut]
    _consistency_check_pair(urut, row_single, row_two, sv_single, sv_two,
                             true_val, true_val_two)
    # memakai base value dan SHAP ruta tsb untuk menggambarkan kontribusi fitur
    # membawa prediksi dari titik awal ke nilai prediksi akhir
    plot_force_with_truth(
        sv_single, true_val, pred_single,
        f"{info['label']} — {pola_label} ({algo.upper()} single-stage), URUT={urut}",
        os.path.join(out_dir, f"{kab_key}_{algo}_{pola_label}_single.png"),
    )
    plot_force_with_truth(
        sv_two, true_val, pred_two,
        f"{info['label']} — {pola_label} ({algo.upper()} two-stage), URUT={urut}",
        os.path.join(out_dir, f"{kab_key}_{algo}_{pola_label}_two.png"),
    )

    # --- BARU v6.1: local decision plot berdampingan (v1 vs v2) ---
    decision_path = os.path.join(
        out_dir, f"local_decision_{info['kode']}_{algo.upper()}_URUT{urut}_{case_type}.png"
    )
    plot_local_decision_comparison(
        sv_single, pred_single, sv_two, pred_two,
        title_main=f"{info['label']} — {algo.upper()} — URUT {urut} — {case_type.replace('_', ' ').title()}",
        save_path=decision_path,
    )

    diag = diagnose_p_rentan_effect(sv_two, pred_two, true_val)
    if diag is not None:
        arah = "searah PERUBAHAN yang menjauh dari true value" if diag["p_rentan_misleading"] else "membantu/netral"
        print(f"    [diag] shap_p_rentan={diag['shap_p_rentan']:.4f} | "
              f"err_with={diag['err_with_p_rentan']:.3f} err_without={diag['err_without_p_rentan']:.3f} "
              f"-> kontribusi p_rentan {arah} (bukan klaim kausal)")

    row = {
        "kabupaten": info["label"], "pola": info["pola"], "algo": algo,
        "case_type": case_type, "rank": rank,
        ID_COL: urut, "true_val": true_val,
        "pred_single": pred_single, "pred_two": pred_two,
        "improvement": abs(true_val - pred_single) - abs(true_val - pred_two),
        "group40": group40, "group40_single": group40_single, "group40_two": group40_two,
        "group20": group20,
    }
    if diag is not None:
        row.update(diag)
    summary_rows.append(row)
    return diag

# mencari ruta dengan perubahan status B40 (NEWLY CORRECT atau NEWLY WRONG) dan menjalankan analisis SHAP lokal untuk masing-masing
def run_status_change_analysis(kab_key, algo, result, case_type, summary_rows,
                                top_n=TOP_N_STATUS_CHANGE):
    """
    Jalankan studi kasus status-change B40 (NEWLY CORRECT atau NEWLY WRONG)
    untuk 1 kabupaten x 1 algoritma: filter RT berdasarkan perubahan status
    group40 dulu (select_newly_correct/select_newly_wrong), lalu ambil 1-3 RT
    teratas berdasarkan besarnya perubahan estimasi lokal, lalu jalankan
    local SHAP (single vs two) + diagnostik p_rentan untuk masing-masing.
    """
    info = KABUPATEN[kab_key]

    if case_type == "newly_correct":
        candidates = select_newly_correct(result["df_single"], result["df_two"])
        label_desc = "NEWLY CORRECT — B40 awalnya IE (salah) pada single-stage, menjadi benar setelah two-stage"
    elif case_type == "newly_wrong":
        candidates = select_newly_wrong(result["df_single"], result["df_two"])
        label_desc = "NEWLY WRONG — B40 awalnya benar pada single-stage, menjadi IE (salah) setelah two-stage"
    else:
        raise ValueError(f"case_type tidak dikenal: {case_type}")

    print(f"\n=== LOCAL SHAP — {info['label']} ({algo.upper()}) — {label_desc} ===")

    if candidates.empty:
        print(f"  [INFO] Tidak ada RT B40 dengan perubahan status ({case_type}) "
              f"untuk {info['label']} ({algo.upper()}).")
        return

    n_take = min(top_n, len(candidates))
    chosen = candidates.head(n_take)

    for rank, row in enumerate(chosen.itertuples(index=False), start=1):
        urut = getattr(row, ID_COL)
        improvement = row.improvement
        g40 = row.group40
        g40s = row.group40_single
        g40t = row.group40_two
        g20 = row.group20
        print(f"  #{rank} URUT={urut} (improvement={improvement:.3f} | "
              f"err_single={row.err_single:.3f} -> err_two={row.err_two:.3f} | "
              f"group40={g40} group40_single={g40s} group40_two={g40t} group20={g20})")
        pola_label = f"{case_type}_{rank}"
        run_local_shap_for_household(
            kab_key, algo, result, urut, pola_label, summary_rows,
            case_type=case_type, rank=rank,
            group40=g40, group40_single=g40s, group40_two=g40t, group20=g20,
        )


def main():
    results = {}
    for kab_key, algos in ANALYSIS_MATRIX.items():
        for algo in algos:
            results[(kab_key, algo)] = run_global_shap(kab_key, algo)

    summary_rows = []

    # --- Kota Magelang: NEWLY CORRECT (RF & XGB), menjelaskan kenapa
    #     two-stage MEMPERBAIKI targeting B40 ---
    for kab_key in NEWLY_CORRECT_KABS:
        for algo in ANALYSIS_MATRIX[kab_key]:
            r = results[(kab_key, algo)]
            run_status_change_analysis(kab_key, algo, r, "newly_correct", summary_rows)

    # --- Kabupaten Kudus: NEWLY WRONG (RF & XGB), menjelaskan kenapa
    #     two-stage MEMPERBURUK targeting B40 ---
    for kab_key in NEWLY_WRONG_KABS:
        for algo in ANALYSIS_MATRIX[kab_key]:
            r = results[(kab_key, algo)]
            run_status_change_analysis(kab_key, algo, r, "newly_wrong", summary_rows)

    summary_path = os.path.join(OUTPUT_DIR, "local", "local_cases_summary.csv")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[saved] {summary_path}")

    # Ringkasan cepat: berapa persen RT per kabupaten x tipe kasus yang
    # kontribusi SHAP p_rentan-nya "misleading" (searah perubahan yang
    # menjauh dari true value) — bukan klaim kausal, lihat docstring
    # diagnose_p_rentan_effect().
    if not summary_df.empty and "p_rentan_misleading" in summary_df.columns:
        print("\n=== RINGKASAN DIAGNOSTIK p_rentan (per kabupaten x tipe kasus status-change B40) ===")
        for (kab_label, case_type), g in summary_df.groupby(["kabupaten", "case_type"]):
            n_misleading = g["p_rentan_misleading"].sum()
            print(f"  {kab_label} [{case_type}]: {n_misleading}/{len(g)} RT dengan kontribusi p_rentan "
                  f"searah perubahan yang menjauh dari true value")

    print("\n=== SELESAI ===")
    print(f"Semua output ada di: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()