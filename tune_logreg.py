"""
Tuning hyperparameter stage 1 logistic regression 

CLI: 
python tune_logreg.py --data 2022:output/susenas_clean_2022.csv 2023:output/susenas_clean_2023.csv 2024:output/susenas_clean_2024.csv --gk gk_full_kab.xlsx --all_kabs --optimize_metric pr_auc
"""

import argparse, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

RANDOM_STATE      = 42
MIN_TRAIN_ROWS    = 50     # sama dengan ambang di script utama
MIN_CLASS_BOTH    = 5
MIN_POS_PER_FOLD  = 5      # sama dengan guard compute_safe_folds di script utama
TARGET_FOLDS      = 5      # jumlah fold "target" sebelum di-guard turun

VALID_OPTIMIZE_METRICS = ("brier", "roc_auc", "pr_auc")
SCORING_MAP = {"brier": "neg_brier_score", "roc_auc": "roc_auc", "pr_auc": "average_precision"}

KODE_MAP = {
    1472: 1473,
    9401: 9501, 9413: 9502, 9414: 9503, 9415: 9504,
    9404: 9604, 9411: 9608, 9410: 9605, 9412: 9601,
    9433: 9607, 9434: 9602, 9435: 9606, 9436: 9603,
    9402: 9702, 9417: 9708, 9416: 9707, 9418: 9704,
    9431: 9705, 9432: 9706, 9430: 9703, 9429: 9701,
    9107: 9202, 9106: 9203, 9108: 9201,
    9109: 9205, 9110: 9204, 9171: 9271,
}
PROV_EXPAND = {
    91: [91, 92], 92: [91, 92],
    94: [95, 96, 97], 95: [95, 96, 97], 96: [95, 96, 97], 97: [95, 96, 97],
}
PREDICTORS = [
    "h_hhcount","h_nmale","h_nfemale","h_nage04","h_nage519","h_nage2064","h_nage65up",
    "h_ngrad_sd","h_ngrad_smp","h_ngrad_sma","h_ngrad_d","h_ngrad_s","h_notgrad",
    "h_sec1_stat1","h_sec1_stat2","h_sec1_stat3","h_sec1_stat4","h_sec1_stat5","h_sec1_stat6",
    "h_sec2_stat1","h_sec2_stat2","h_sec2_stat3","h_sec2_stat4","h_sec2_stat5","h_sec2_stat6",
    "h_sec3_stat1","h_sec3_stat2","h_sec3_stat3","h_sec3_stat4","h_sec3_stat5","h_sec3_stat6",
    "h_neverschool","h_stillschool","h_notschool",
    "h_house1","h_house2","h_house3","h_house4",
    "h_floor1","h_floor2","h_floor3","h_floor4",
    "h_wall1","h_wall2","h_wall3","h_wall4",
    "h_roof1","h_roof2","h_roof3","h_roof4",
    "h_dwater1","h_dwater2","h_dwater3","h_dwater4","h_dwater5",
    "h_lighting1","h_lighting2","h_lighting3",
    "h_epower1","h_epower2","h_epower3",
    "h_cookingfuel1","h_cookingfuel2","h_cookingfuel3","h_cookingfuel4","h_cookingfuel5",
    "h_toiltype1","h_toiltype2","h_toiltype3","h_toiltype4","h_toiltype5","h_toiltype6",
    "h_septic1","h_septic2","h_septic3","h_septic4","h_septic5",
    "h_asset_lpg5kg","h_asset_fridge","h_asset_ac","h_asset_wheater","h_asset_phone",
    "h_asset_computer","h_asset_jewelry","h_asset_motorcycle","h_asset_boat",
    "h_asset_motorboat","h_asset_car","h_asset_tv","h_asset_smartphone","h_asset_land",
    "h_lnluaslantai","h_nfamily","h_notmarried","h_married","h_divorced","h_widowed",
]
# selama proses tuning, penalty = L2 memberi penalti pada koefisien yang terlalu besar.
# besar pengaruh penalti diatur menggunakan hyperparameter C, semakin kecil C semakin besar penalti. 
# solver = lbfgs adalah algoritma optimisasi yang digunakan untuk menemukan koefisien model.
LOGREG_GRID = [
    {
        "clf__penalty": ["l2"],
        "clf__C":       [0.01, 0.1, 1.0, 10.0],
        "clf__solver":  ["lbfgs"],
    },
]
# C mengatur kuat lemahnya penalti L2 
# jadi model mencari koefisien dengan meminimalkan logloss berbobot kelas + penalti L2
# jadi koefisien B tetap dipelajari dari data, tetapi kesalahan kelas minoritas diperhtiungkan lebih besar dan koef
# dan koef yang terlalu besar diberi penalti L2 kalau prediktor berkorelasi 
# L2 : mengecilkan koef tanpa melakukan seleksi fitur seperti L1 yang dapat membuat beberapa koef tepat nol -> mempertahankan semua fitur dalam model 
# solver lbfgs dipakai sbg algoritma optimasi untuk mencari koefisien B (MLE) karena kompatibel 
# dengan L2 dan merupakan pilihan umum yang robust di scikit learn
# batas untuk menentukan batas miskin (pcexp < THRESHOLD * garis kemiskinan) -> kerentanan kemiskinan
THRESHOLD = 1.5

# target_folds: jumlah fold yang diinginkan misal ya 5 atau 10, 
# min cls jumlah data pada kelas yang lebih sedikit
# min_pos_per_fold: jumlah minimum data positif per fold (misal 5)
def compute_safe_folds(target_folds, min_cls, min_pos_per_fold=MIN_POS_PER_FOLD):
    return max(2, min(target_folds, min_cls // min_pos_per_fold))

def parse_data_args(items):
    out = {}
    for it in items:
        if ":" not in it: continue # skip jika input gaada :
        y, p = it.split(":", 1) # dibagi di tanda :, misal 2020: data/susenas_clean_2020 -> 2020 y dan p nya data/susenas_clean_2020
        out[int(y)] = p # int(y) -> 2020, p -> data/susenas_clean_2020
    return out

def load_year(path, year):
    df = pd.read_csv(path)
    p_col = next((c for c in ["PROP","R101","IDPROV"] if c in df.columns), "R101")
    k_col = next((c for c in ["KAB","R102","IDKAB"]  if c in df.columns), "R102")
    df["year"] = year
    df["kode_kab_new"] = (df[p_col].astype(int)*100 + df[k_col].astype(int)).replace(KODE_MAP)
    df["kode_prov_new"] = df["kode_kab_new"] // 100
    return df

def load_gk(excel_path):
    df = pd.read_excel(excel_path).dropna(subset=["kode_kab"]).copy() # membuang baris yang kdoe kabupatennya kosong dan.copy membuat salinan tabel 
    df["kode_kab"] = df["kode_kab"].astype(int) # memastikan kode kab bilangan bulat
    year_cols = [c for c in df.columns if str(c).isdigit()] # mencari kolom yang namanya angka misal tahun 2020
    gk_long = df[["kode_kab"] + year_cols].melt(
        id_vars="kode_kab", var_name="year", value_name="gk") # mengambil kode kab dan kolom tahun
    gk_long["year"] = gk_long["year"].astype(int)
    return gk_long.dropna(subset=["gk"])

def attach_label(df, gk_data):
    df = df.merge(gk_data, left_on=["kode_kab_new","year"], #dimerge aja cari kode_kab_new yang sama dan tahun yang sama
                  right_on=["kode_kab","year"], how="left") # how="left" berarti tetap pertahankan semua baris dari df
    df = df.dropna(subset=["gk"]) # membuang baris yang gk-nya kosong
    df["miskin"] = (df["pcexp"] < THRESHOLD * df["gk"]).astype(int) # membuat kolom miskin, jika pcexp < THRESHOLD * gk maka miskin = 1, jika tidak miskin = 0
    return df

def ensure_predictors(df):
    for col in PREDICTORS:
        if col not in df.columns: df[col] = 0
    return df

def tune_one_kab(kab, df_kab, optimize_metric):
    X = df_kab[PREDICTORS].values # ambil semua kolom prediktor
    y = df_kab["miskin"].values # ambil kolom miskin sebagai target 
    n = len(y) # jumlah baris data pada tiap kabupaten kota tsb 

    counts = np.bincount(y.astype(int)) if n > 0 else np.array([0]) # menghitung berapa banyak label 0 dan label 1
    min_cls = counts.min() if len(counts) >= 2 else 0 # banyak nilai minoritas
    prevalence = y.mean() if n > 0 else np.nan # 2 label jadi rumah tangga berlabel dicari prev misal 20 dari 100 ya 20% atau 0,2

    if min_cls < MIN_CLASS_BOTH or n < MIN_TRAIN_ROWS:
        return {
            "kode_kab_new": kab, "n": n, "n_pos": int(counts[1]) if len(counts) > 1 else 0,
            "prevalence": round(prevalence, 4) if n > 0 else np.nan,
            "cv_folds_used": np.nan, "best_C": np.nan, "best_penalty": np.nan,
            "best_solver": np.nan, "best_score": np.nan,
            "status": f"SKIPPED (min_cls={min_cls} < {MIN_CLASS_BOTH} atau n={n} < {MIN_TRAIN_ROWS})",
        }

    n_splits = compute_safe_folds(TARGET_FOLDS, min_cls) # menentukan jumlah k fold
    scoring = SCORING_MAP[optimize_metric] # menentukan ukuran yang dipakai untuk membandingkan model

    pipe = Pipeline([
        ("scaler", StandardScaler()), # menyamakan skala variabel prediktor 
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)), # melatih model
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE) # data dibagi menjadi beberapa fold dengan stratifiedkFold berusaha menjaga proporsi label miskin dan tidak miskin di setiap fold
    gs = GridSearchCV(pipe, LOGREG_GRID, cv=cv, scoring=scoring, n_jobs=1, refit=False)

    try:
        gs.fit(X, y)
        bp = gs.best_params_
        return {
            "kode_kab_new": kab, "n": n, "n_pos": int(counts[1]),
            "prevalence": round(prevalence, 4),
            "cv_folds_used": n_splits,
            "best_C": bp["clf__C"], "best_penalty": bp["clf__penalty"],
            "best_solver": bp["clf__solver"], "best_score": round(gs.best_score_, 5),
            "status": "OK",
        }
    except Exception as e:
        return {
            "kode_kab_new": kab, "n": n, "n_pos": int(counts[1]),
            "prevalence": round(prevalence, 4),
            "cv_folds_used": n_splits, "best_C": np.nan, "best_penalty": np.nan,
            "best_solver": np.nan, "best_score": np.nan,
            "status": f"ERROR: {e}",
        }

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, metavar="YEAR:FILE",
                         help="Data Susenas per tahun, format YEAR:PATH.")
    parser.add_argument("--gk", required=True, help="Path file garis kemiskinan (.xlsx).")
    parser.add_argument("--provinsi", type=int, nargs="*", default=[],
                         help="Filter provinsi (opsional).")
    parser.add_argument("--kabs", type=int, nargs="*", default=None,
                         help="Daftar kode_kab_new spesifik yang mau diuji. "
                              "Kalau tidak diisi, pakai --n_sample atau --all_kabs.")
    parser.add_argument("--n_sample", type=int, default=None,
                         help="Jumlah kabupaten yang di-random-sample untuk diuji "
                              "(dipakai kalau --kabs tidak diisi).")
    parser.add_argument("--all_kabs", action="store_true",
                         help="Uji SEMUA kabupaten yang ada di data (bisa lama).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE,
                         help="Random seed untuk --n_sample.")
    parser.add_argument("--optimize_metric", choices=list(VALID_OPTIMIZE_METRICS),
                         default="pr_auc", help="Metric yang dioptimalkan di GridSearchCV.")
    parser.add_argument("--output", default="./output_tuning_per_kab")
    args = parser.parse_args()

    if not args.kabs and not args.n_sample and not args.all_kabs:
        print("[ERROR] Harus isi salah satu: --kabs, --n_sample, atau --all_kabs")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    print("[1/3] Loading GK & Susenas...")
    gk_data  = load_gk(args.gk)
    data_map = parse_data_args(args.data)

    pool = []
    for y, p in sorted(data_map.items()):
        if not os.path.exists(p):
            print(f"  [WARN] {p} skip")
            continue
        df = load_year(p, y)
        df = attach_label(df, gk_data)
        df = ensure_predictors(df)
        pool.append(df)
        print(f"  {y}: {len(df):,} baris")

    df_all = pd.concat(pool, ignore_index=True)

    if args.provinsi:
        prov_set = set()
        for p in args.provinsi:
            prov_set.add(p)
            prov_set.update(PROV_EXPAND.get(p, []))
        df_all = df_all[df_all["kode_prov_new"].isin(prov_set)].copy()

    available_kabs = sorted(df_all["kode_kab_new"].unique())
    print(f"  Total kabupaten tersedia: {len(available_kabs)}")

    print("\n[2/3] Menentukan kabupaten yang diuji...")
    if args.kabs:
        missing = [k for k in args.kabs if k not in available_kabs]
        if missing:
            print(f"  [WARN] Kode kab tidak ditemukan di data, di-skip: {missing}")
        target_kabs = [k for k in args.kabs if k in available_kabs]
    elif args.all_kabs:
        target_kabs = available_kabs
    else:
        rng = np.random.default_rng(args.seed)
        n_sample = min(args.n_sample, len(available_kabs))
        target_kabs = sorted(rng.choice(available_kabs, size=n_sample, replace=False).tolist())

    print(f"  Kabupaten yang diuji ({len(target_kabs)}): {target_kabs}")

    print(f"\n[3/3] Tuning per kabupaten (optimize_metric={args.optimize_metric})...")
    results = []
    for i, kab in enumerate(target_kabs):
        df_kab = df_all[df_all["kode_kab_new"] == kab]
        res = tune_one_kab(kab, df_kab, args.optimize_metric)
        results.append(res)
        print(f"  [{i+1}/{len(target_kabs)}] Kab {kab}: {res['status']} "
              f"| best_C={res['best_C']} | n={res['n']} | n_pos={res['n_pos']}")

    report = pd.DataFrame(results)
    out_path = os.path.join(args.output, f"tuning_per_kab_{args.optimize_metric}.csv")
    report.to_csv(out_path, index=False)
    ok = report[report["status"] == "OK"]
    print("\n" + "="*60)
    print("RINGKASAN ROBUSTNESS")
    print("="*60)
    if len(ok) == 0:
        print("Tidak ada kabupaten yang berhasil di-tuning (semua di-skip/error).")
    else:
        print(f"Kabupaten berhasil di-tuning : {len(ok)} / {len(target_kabs)}")
        c_counts = ok["best_C"].value_counts()
        print("\nDistribusi best_C terpilih antar kabupaten:")
        for c_val, cnt in c_counts.items():
            pct = 100 * cnt / len(ok)
            print(f"  C={c_val:<6} -> {cnt} kabupaten ({pct:.0f}%)")
        modus_c = c_counts.idxmax()
        pct_modus = 100 * c_counts.max() / len(ok)
        print(f"\nC paling dominan: {modus_c} (dipilih di {pct_modus:.0f}% kabupaten yang diuji)")
        if pct_modus >= 70:
            print("-> Hyperparameter tampak ROBUST lintas kabupaten (mayoritas kuat sepakat).")
        elif pct_modus >= 50:
            print("-> Hyperparameter CUKUP konsisten, tapi ada variasi yang perlu didiskusikan.")
        else:
            print("-> Hyperparameter BERVARIASI cukup besar antar kabupaten — pertimbangkan "
                  "untuk membahas ini sebagai limitation, atau bandingkan dengan hasil "
                  "global_tune() di pipeline utama.")

    print(f"\nReport lengkap tersimpan di: {out_path}")


if __name__ == "__main__":
    main()