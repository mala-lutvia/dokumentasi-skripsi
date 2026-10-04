"""
Tuning Hyperparameter XGBoost Per Kabupaten — Robustness Check
"""

import argparse, os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from joblib import Parallel, delayed
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE      = 42
MIN_TRAIN_ROWS    = 50     
MIN_CLASS_BOTH    = 5
MIN_POS_PER_FOLD  = 5     
TARGET_FOLDS      = 5

VALID_OPTIMIZE_METRICS = ("brier", "roc_auc", "pr_auc")
SCORING_MAP = {"brier": "neg_brier_score", "roc_auc": "roc_auc", "pr_auc": "average_precision"}

XGB_GRID = {
    "n_estimators":  [300, 500, 700],
    # mengatur berapa banyak skor pohon baru ditambahkan ke prediksi
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth":     [3, 5, 7],
    # mengatur seberapa kuat skor daun pohon ditahan agar tidak terlalu besar
    "reg_lambda":    [0.01, 0.1, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.7],  
    "subsample": [0.5, 0.7, 0.9],         
}

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
THRESHOLD = 1.5

def compute_safe_folds(target_folds, min_cls, min_pos_per_fold=MIN_POS_PER_FOLD):
    return max(2, min(target_folds, min_cls // min_pos_per_fold))

# ini tu rasio scale pos weight untuk tiap fold misal fold 1 itu ada 5 miskin 10 tidak brrt 10/5 = 2
# Secara intuitif, karena kelas miskin lebih sedikit, 
# model didorong untuk lebih memperhatikan kesalahan pada kelas miskin
def get_scale_pos_weight(y):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    return neg / pos if pos > 0 else 1.0

def parse_data_args(items):
    out = {}
    for it in items:
        if ":" not in it: continue
        y, p = it.split(":", 1)
        out[int(y)] = p
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
    df = pd.read_excel(excel_path).dropna(subset=["kode_kab"]).copy()
    df["kode_kab"] = df["kode_kab"].astype(int)
    year_cols = [c for c in df.columns if str(c).isdigit()]
    gk_long = df[["kode_kab"] + year_cols].melt(
        id_vars="kode_kab", var_name="year", value_name="gk")
    gk_long["year"] = gk_long["year"].astype(int)
    return gk_long.dropna(subset=["gk"])

def attach_label(df, gk_data):
    df = df.merge(gk_data, left_on=["kode_kab_new","year"],
                  right_on=["kode_kab","year"], how="left")
    df = df.dropna(subset=["gk"])
    df["miskin"] = (df["pcexp"] < THRESHOLD * df["gk"]).astype(int)
    return df

def ensure_predictors(df):
    for col in PREDICTORS:
        if col not in df.columns: df[col] = 0
    return df

def make_xgb_for_grid(scale_weight, **grid_params):
    return XGBClassifier(
        # tugasnya klasifikasi biner dan model menghasilkan probabilitas
        objective        = "binary:logistic",
        # log loss menilai kualitas probabilitas prediksi, semakin kecil semakin baik
        eval_metric      = "logloss",
        # memberi bobot lebih besar pada kelas positif pada saat model dilatih. kalau kelas 
        # miskin lebih sedikit, bobot ini dapat membuat model lebih memperhatikan kesalahan pada kelas tersebut. 
        scale_pos_weight = scale_weight,
        # memakai metode histogram untuk mencari pemisahan pada pohon. Metode ini mengelompokkan nilai fitur ke dalam bin sehingga pencarian split dapat lebih efisien
        tree_method      = "hist",
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
        verbosity        = 0,
        **grid_params,
    )

def _fold_score(y_valid, proba, optimize_metric):
    if optimize_metric == "brier":
        return -brier_score_loss(y_valid, proba)
    elif optimize_metric == "roc_auc":
        return roc_auc_score(y_valid, proba)
    elif optimize_metric == "pr_auc":
        return average_precision_score(y_valid, proba)
    else:
        raise ValueError(f"optimize_metric tidak dikenal: {optimize_metric}")

def score_one_param_combo(grid_params, X, y, cv_splits, optimize_metric):
    # menyiapkan tempat untuk menghitung skor tiap putaran CV untuk satu kombinasi hyperparameter 
    fold_scores = []
    # oof fold
    for train_idx, valid_idx in cv_splits:
        # memisahkan prediktor dan label untuk bagian data latih
        X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
        y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]

        # scale_pos_weight fold-wise: dihitung dari y_train_fold
        scale_weight_fold = get_scale_pos_weight(y_train_fold)

        model = make_xgb_for_grid(scale_weight_fold, **grid_params)
        # model belajar hanya dari bagian latih 
        model.fit(X_train_fold, y_train_fold)
        # skor probabilitas dibandingkan dengan label asli di fold validasi, menggunakan metrik yang dipilih
        proba = model.predict_proba(X_valid_fold)[:, 1]
        fold_scores.append(_fold_score(y_valid_fold, proba, optimize_metric))
    # nilai rata-rata inilah skor untuk kombinasi hyperparameter tersebut. 
    return float(np.mean(fold_scores))

def tune_one_kab(kab, df_kab, optimize_metric):
    X = df_kab[PREDICTORS].values
    y = df_kab["miskin"].values
    n = len(y)

    counts = np.bincount(y.astype(int)) if n > 0 else np.array([0])
    min_cls = counts.min() if len(counts) >= 2 else 0
    prevalence = y.mean() if n > 0 else np.nan

    if min_cls < MIN_CLASS_BOTH or n < MIN_TRAIN_ROWS:
        return {
            "kode_kab_new": kab, "n": n, "n_pos": int(counts[1]) if len(counts) > 1 else 0,
            "prevalence": round(prevalence, 4) if n > 0 else np.nan,
            "cv_folds_used": np.nan, "scale_pos_weight": np.nan,
            "best_n_estimators": np.nan, "best_learning_rate": np.nan,
            "best_max_depth": np.nan, "best_reg_lambda": np.nan, 
            "best_colsample_bytree": np.nan, "best_subsample": np.nan,
            "best_score": np.nan, 
            "status": f"SKIPPED (min_cls={min_cls} < {MIN_CLASS_BOTH} atau n={n} < {MIN_TRAIN_ROWS})",
        }

    n_splits = compute_safe_folds(TARGET_FOLDS, min_cls)
    scale_weight_global = get_scale_pos_weight(y)
    # Karena stratified, pembagian fold berusaha menjaga proporsi miskin dan tidak miskin
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_splits = list(cv.split(X, y))

    param_grid_list = list(ParameterGrid(XGB_GRID))

    try:
        mean_scores = Parallel(n_jobs=-1)(
            delayed(score_one_param_combo)(params, X, y, cv_splits, optimize_metric)
            for params in param_grid_list
        )
        mean_scores = np.array(mean_scores)
        best_idx = int(np.argmax(mean_scores))
        bp = param_grid_list[best_idx]
        best_score = mean_scores[best_idx]

        return {
            "kode_kab_new": kab, "n": n, "n_pos": int(counts[1]),
            "prevalence": round(prevalence, 4),
            "cv_folds_used": n_splits, "scale_pos_weight": round(scale_weight_global, 3),
            "best_n_estimators": bp["n_estimators"],
            "best_learning_rate": bp["learning_rate"],
            "best_max_depth": bp["max_depth"],
            "best_reg_lambda": bp["reg_lambda"],
            "best_colsample_bytree": bp["colsample_bytree"],
            "best_subsample": bp["subsample"],
            "best_score": round(float(best_score), 5),
            "status": "OK",
        }
    except Exception as e:
        return {
            "kode_kab_new": kab, "n": n, "n_pos": int(counts[1]),
            "prevalence": round(prevalence, 4),
            "cv_folds_used": n_splits, "scale_pos_weight": round(scale_weight_global, 3),
            "best_n_estimators": np.nan, "best_learning_rate": np.nan,
            "best_max_depth": np.nan, "best_reg_lambda": np.nan, "best_colsample_bytree": np.nan, "best_subsample": np.nan, "best_score": np.nan,
            "status": f"ERROR: {e}",
        }

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", required=True, metavar="YEAR:FILE")
    parser.add_argument("--gk", required=True)
    parser.add_argument("--provinsi", type=int, nargs="*", default=[])
    parser.add_argument("--kabs", type=int, nargs="*", default=None,
                         help="Daftar kode_kab_new spesifik yang mau diuji.")
    parser.add_argument("--n_sample", type=int, default=None,
                         help="Jumlah kabupaten random-sample (kalau --kabs kosong).")
    parser.add_argument("--all_kabs", action="store_true",
                         help="Uji SEMUA kabupaten (bisa lama — 729 kombinasi x tiap kab).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--optimize_metric", choices=list(VALID_OPTIMIZE_METRICS),
                         default="pr_auc", help="Metric yang dioptimalkan pada CV.")
    parser.add_argument("--output", default="./output_tuning_xgb_per_kab")
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

    print(f"\n[3/3] Tuning XGB per kabupaten (grid={XGB_GRID}, "
          f"optimize_metric={args.optimize_metric})...")
    results = []
    for i, kab in enumerate(target_kabs):
        df_kab = df_all[df_all["kode_kab_new"] == kab]
        res = tune_one_kab(kab, df_kab, args.optimize_metric)
        results.append(res)
        print(f"  [{i+1}/{len(target_kabs)}] Kab {kab}: {res['status']} "
              f"| n_est={res['best_n_estimators']} lr={res['best_learning_rate']} "
              f"depth={res['best_max_depth']} lambda={res['best_reg_lambda']} "
              f"colsample_bytree={res['best_colsample_bytree']} subsample={res['best_subsample']} "
              f"| n={res['n']} n_pos={res['n_pos']}")

    report = pd.DataFrame(results)
    out_path = os.path.join(args.output, f"tuning_xgb_per_kab_{args.optimize_metric}.csv")
    report.to_csv(out_path, index=False)

    ok = report[report["status"] == "OK"]
    print("\n" + "="*60)
    print("RINGKASAN ROBUSTNESS")
    print("="*60)
    if len(ok) == 0:
        print("Tidak ada kabupaten yang berhasil di-tuning (semua di-skip/error).")
    else:
        print(f"Kabupaten berhasil di-tuning : {len(ok)} / {len(target_kabs)}")
        for param in ["best_n_estimators", "best_learning_rate", "best_max_depth", "best_reg_lambda", "best_colsample_bytree", "best_subsample"]:
            counts = ok[param].value_counts()
            modus = counts.idxmax()
            pct = 100 * counts.max() / len(ok)
            print(f"\nDistribusi {param}:")
            for val, cnt in counts.items():
                print(f"  {val:<8} -> {cnt} kabupaten ({100*cnt/len(ok):.0f}%)")
            print(f"  Modus: {modus} ({pct:.0f}% kabupaten sepakat)")

    print(f"\nReport lengkap tersimpan di: {out_path}")


if __name__ == "__main__":
    main()