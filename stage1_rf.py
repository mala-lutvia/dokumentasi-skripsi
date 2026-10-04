"""
Stage 1 — P_Rentan via Random Forest

"""

import argparse, gc, logging, os, sys, warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

THRESHOLD         = 1.5
RANDOM_STATE      = 42
N_FOLDS_CAL       = 5     # ceiling cv untuk OOF per kab DAN inner CalibratedClassifierCV
MIN_TRAIN_ROWS    = 50
MIN_CLASS_BOTH    = 5
MIN_POS_PER_FOLD  = 5

MIN_SAMPLES_LEAF  = 3
MIN_SAMPLES_SPLIT = 2
CLASS_WEIGHT      = "balanced"

VALID_CALIBRATIONS = ("none", "sigmoid")

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

def compute_safe_folds(target_folds, min_cls, min_pos_per_fold=MIN_POS_PER_FOLD):
    return max(2, min(target_folds, min_cls // min_pos_per_fold))

def parse_data_args(items):
    out = {}
    for item in items:
        if ":" in item:
            year, path = item.split(":", 1)
            out[int(year)] = path
    return out

def load_clean_data(path, year):
    df = pd.read_csv(path)
    p_col = next((c for c in ["PROP", "R101", "IDPROV"] if c in df.columns), "R101")
    k_col = next((c for c in ["KAB",  "R102", "IDKAB"]  if c in df.columns), "R102")
    df["year"] = year
    df["kode_kab_new"]  = (df[p_col].astype(int) * 100 + df[k_col].astype(int)).replace(KODE_MAP)
    df["kode_prov_new"] = df["kode_kab_new"] // 100
    return df

def load_poverty_line(excel_path):
    df = pd.read_excel(excel_path).dropna(subset=["kode_kab"]).copy()
    df["kode_kab"] = df["kode_kab"].astype(int)
    year_cols = [c for c in df.columns if str(c).isdigit()]
    gk_long = df[["kode_kab"] + year_cols].melt(
        id_vars="kode_kab", var_name="year", value_name="gk"
    )
    gk_long["year"] = gk_long["year"].astype(int)
    return gk_long.dropna(subset=["gk"])

def ensure_predictors(df):
    for col in PREDICTORS:
        if col not in df.columns:
            df[col] = 0
    return df

def make_model(n_estimators, max_depth):
    return RandomForestClassifier(
        n_estimators      = n_estimators,
        max_depth         = max_depth,
        min_samples_leaf  = MIN_SAMPLES_LEAF,
        min_samples_split = MIN_SAMPLES_SPLIT,
        class_weight      = CLASS_WEIGHT,
        random_state      = RANDOM_STATE,
        n_jobs            = 1,
    )

def fit_calibrated(X, y, n_estimators, max_depth, calibration_method, n_folds_cal):
    counts  = np.bincount(y.astype(int))
    min_cls = counts.min() if len(counts) >= 2 else 0
    base = make_model(n_estimators, max_depth)

    if min_cls >= 2 and len(y) >= 10:
        n_splits = compute_safe_folds(n_folds_cal, min_cls)
        calibrated = CalibratedClassifierCV(
            base, method=calibration_method, cv=n_splits, ensemble=False,
        )
        calibrated.fit(X, y)
        return calibrated
    else:
        base.fit(X, y)
        return base

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    roc   = roc_auc_score(y_true, y_prob)          if len(np.unique(y_true)) > 1 else np.nan
    pr    = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    f1    = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    return roc, pr, f1, brier

def process_kab(kab, df_tr, df_te, n_estimators, max_depth, calibration, n_folds_cal,
                 model_dir, save_models):
    df_tr = df_tr.copy().reset_index(drop=True)
    df_te = df_te.copy().reset_index(drop=True)
    X_tr, y_tr = df_tr[PREDICTORS].values, df_tr["miskin"].values
    X_te, y_te = df_te[PREDICTORS].values, df_te["miskin"].values

    counts_tr  = np.bincount(y_tr.astype(int)) if len(y_tr) > 0 else np.array([0])
    min_cls    = counts_tr.min() if len(counts_tr) >= 2 else 0
    prev_train = y_tr.mean() if len(y_tr) > 0 else np.nan
    prev_test  = y_te.mean() if len(y_te) > 0 else np.nan
    # membuat tempat untuk menyimpan oof prob
    oof_prob     = np.full(len(df_tr), np.nan)
    oof_prob_raw = np.full(len(df_tr), np.nan)

    if min_cls >= MIN_CLASS_BOTH and len(df_tr) >= MIN_TRAIN_ROWS:
        n_outer = compute_safe_folds(n_folds_cal, min_cls)
        skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=RANDOM_STATE)

        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            try:
                # membuat oof prob
                m_raw = make_model(n_estimators, max_depth)
                m_raw.fit(X_tr[tr_idx], y_tr[tr_idx])
                # satu baris train mendapat skor dari model yang tidak dilatih menggunakan baris itu sendiri
                oof_prob_raw[val_idx] = m_raw.predict_proba(X_tr[val_idx])[:, 1]

                if calibration != "none":
                    m_cal = fit_calibrated(
                        X_tr[tr_idx], y_tr[tr_idx], n_estimators, max_depth,
                        calibration, n_folds_cal,
                    )
                    oof_prob[val_idx] = m_cal.predict_proba(X_tr[val_idx])[:, 1]
                else:
                    oof_prob[val_idx] = oof_prob_raw[val_idx]

            except Exception as e:
                logger.debug(f"Kab {kab} fold {fold_i} error: {e}")
                oof_prob[val_idx]     = prev_train
                oof_prob_raw[val_idx] = prev_train
    else:
        oof_prob[:]     = prev_train
        oof_prob_raw[:] = prev_train

    df_tr["p_rentan"]     = oof_prob
    df_tr["p_rentan_raw"] = oof_prob_raw
    # prediksi untuk data test 
    test_prob     = np.full(len(df_te), prev_train)
    test_prob_raw = np.full(len(df_te), prev_train)
    try:
        final_raw = make_model(n_estimators, max_depth)
        # model raw dilatih menggunakan seluruh x_tr, y_tr, lalu memprediksi x_te dan hasilnya disimpan sbg p rentan raw 
        final_raw.fit(X_tr, y_tr)
        if len(df_te) > 0:
            test_prob_raw = final_raw.predict_proba(X_te)[:, 1]

        if calibration != "none":
            final_model = fit_calibrated(X_tr, y_tr, n_estimators, max_depth,
                                          calibration, n_folds_cal)
            if len(df_te) > 0:
                test_prob = final_model.predict_proba(X_te)[:, 1]
            if save_models:
                joblib.dump(final_model, os.path.join(model_dir, f"rf_{kab}_{calibration}.pkl"))
        else:
            test_prob = test_prob_raw
            if save_models:
                joblib.dump(final_raw, os.path.join(model_dir, f"rf_{kab}_raw.pkl"))

    except Exception as e:
        logger.warning(f"Kab {kab} final model error: {e}")
    # jadi intinya prediksi train berasal dari OOF; prediksi test berasal dari model yang dilatih pada seluruh train
    # menyimpan test prob ke p_rentan misal susenas 2025 
    df_te["p_rentan"]     = test_prob
    df_te["p_rentan_raw"] = test_prob_raw

    valid_o = ~np.isnan(oof_prob)
    # menghitung metrik p rentan oof 
    m_oof  = (compute_metrics(y_tr[valid_o], oof_prob[valid_o])
              if valid_o.sum() > 0 and len(np.unique(y_tr[valid_o])) > 1 else (np.nan,) * 4)
    # menghitung metrik p rentan test 
    m_test = (compute_metrics(y_te, test_prob)
              if len(df_te) > 0 and len(np.unique(y_te)) > 1 else (np.nan,) * 4)
    # mengembalikan raw hasil report dengan nama kolom kab, metrik oof, metrik test, prev train prev test prev lift
    mrow = ([kab] + list(m_oof) + list(m_test) +
            [len(df_tr), len(df_te), round(prev_train, 4), round(prev_test, 4)])
    return df_tr, df_te, mrow


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 Random Forest — pemodelan p_rentan dengan hyperparameter "
                     "TETAP (fixed), hasil robustness test terpisah "
                     "(tune_rf_per_kab.py). Tidak ada tuning otomatis di script ini.")
    parser.add_argument("--data",           nargs="+", required=True, metavar="YEAR:FILE")
    parser.add_argument("--gk",             required=True)
    parser.add_argument("--year_test",      type=int, required=True)
    parser.add_argument("--provinsi",       type=int, nargs="*", default=[])
    parser.add_argument("--calibration",    choices=list(VALID_CALIBRATIONS), default="sigmoid")
    parser.add_argument("--output",         default="./output_stage1_rf_fixed")
    parser.add_argument("--no_save_models", action="store_true")
    parser.add_argument("--fixed_n_estimators", type=int, required=True,
                         help="[WAJIB] Hasil robustness test (tune_rf_per_kab.py).")
    parser.add_argument("--fixed_max_depth", type=int, required=True,
                         help="[WAJIB] Hasil robustness test.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model_dir = os.path.join(args.output, "models")
    os.makedirs(model_dir, exist_ok=True)

    print(f"[CONFIG] calibration={args.calibration} | N_FOLDS_CAL={N_FOLDS_CAL} (target)")
    print(f"[CONFIG] Fixed hyperparameter (dari robustness test terpisah): "
          f"n_estimators={args.fixed_n_estimators} max_depth={args.fixed_max_depth}")
    print(f"[CONFIG] Fixed (bukan CLI/tuning): min_samples_leaf={MIN_SAMPLES_LEAF} "
          f"min_samples_split={MIN_SAMPLES_SPLIT} class_weight={CLASS_WEIGHT}")

    print("\n[1/4] Loading GK & Susenas...")
    gk_data  = load_poverty_line(args.gk)
    data_map = parse_data_args(args.data)
    if args.year_test not in data_map:
        print(f"[ERROR] year_test {args.year_test} tidak ada di --data")
        sys.exit(1)

    pool = {}
    for year, path in sorted(data_map.items()):
        if not os.path.exists(path):
            print(f"  [WARN] {path} skip")
            continue
        df = load_clean_data(path, year)
        df = df.merge(gk_data, left_on=["kode_kab_new", "year"],
                       right_on=["kode_kab", "year"], how="inner")
        df["miskin"] = (df["pcexp"] < THRESHOLD * df["gk"]).astype(int)
        df = ensure_predictors(df)
        pool[year] = df
        print(f"  {year}: {len(df):,} baris | miskin={df['miskin'].mean():.1%}")

    train_years  = [y for y in sorted(pool.keys()) if y != args.year_test]
    df_train_all = pd.concat([pool[y] for y in train_years], ignore_index=True)
    df_test_all  = pool[args.year_test].copy()

    if args.provinsi:
        prov_set = set()
        for p in args.provinsi:
            prov_set.add(p)
            prov_set.update(PROV_EXPAND.get(p, []))
        df_train_all = df_train_all[df_train_all["kode_prov_new"].isin(prov_set)].copy()
        df_test_all  = df_test_all[df_test_all["kode_prov_new"].isin(prov_set)].copy()

    print(f"\n  Train {train_years}: {len(df_train_all):,} baris | "
          f"miskin={df_train_all['miskin'].mean():.1%}")
    print(f"  Test  {args.year_test}: {len(df_test_all):,} baris | "
          f"miskin={df_test_all['miskin'].mean():.1%}")

    kabs = sorted(df_train_all["kode_kab_new"].unique())
    print(f"\n[2/4] OOF per kab ({len(kabs)} kab, calibration={args.calibration}, "
          f"n_estimators={args.fixed_n_estimators}, max_depth={args.fixed_max_depth})...")

    df_train_all = df_train_all.reset_index(drop=True)
    df_test_all  = df_test_all.reset_index(drop=True)
    df_train_all["p_rentan"]     = np.nan
    df_train_all["p_rentan_raw"] = np.nan
    df_test_all["p_rentan"]      = np.nan
    df_test_all["p_rentan_raw"]  = np.nan

    all_metrics = []
    for i, kab in enumerate(kabs):
        mask_tr = df_train_all["kode_kab_new"] == kab
        mask_te = df_test_all["kode_kab_new"]  == kab
        df_k_tr = df_train_all[mask_tr]
        df_k_te = df_test_all[mask_te]
        if len(df_k_tr) < 10:
            continue

        df_out_tr, df_out_te, mrow = process_kab(
            kab, df_k_tr, df_k_te, args.fixed_n_estimators, args.fixed_max_depth,
            args.calibration, N_FOLDS_CAL, model_dir, not args.no_save_models,
        )

        df_train_all.loc[mask_tr, "p_rentan"]     = df_out_tr["p_rentan"].values
        df_train_all.loc[mask_tr, "p_rentan_raw"] = df_out_tr["p_rentan_raw"].values
        if not df_k_te.empty:
            df_test_all.loc[mask_te, "p_rentan"]     = df_out_te["p_rentan"].values
            df_test_all.loc[mask_te, "p_rentan_raw"] = df_out_te["p_rentan_raw"].values
        all_metrics.append(mrow)

        if (i + 1) % 50 == 0 or (i + 1) == len(kabs):
            roc_m = np.nanmean([m[5] for m in all_metrics])
            pr_m  = np.nanmean([m[6] for m in all_metrics])
            print(f"  [{i+1:3d}/{len(kabs)}] avg Test ROC={roc_m:.3f} PR={pr_m:.3f}")
        gc.collect()

    print("\n[3/4] Menyimpan hasil...")
    cols = ["Kab", "OOF_ROC","OOF_PR","OOF_F1","OOF_Brier",
            "Test_ROC","Test_PR","Test_F1","Test_Brier",
            "N_Train","N_Test","Prev_Train","Prev_Test"]
    report = pd.DataFrame(all_metrics, columns=cols).replace([np.inf, -np.inf], np.nan)
    report["PR_lift"] = report["Test_PR"] / report["Prev_Test"].replace(0, np.nan)

    avg = report.mean(numeric_only=True).to_frame().T
    avg["Kab"] = "AVERAGE"
    report = pd.concat([report, avg], ignore_index=True)

    calib_tag = args.calibration
    report_path = os.path.join(
        args.output, f"report_stage1_rf_fixed_{calib_tag}_{args.year_test}.csv")
    report.to_csv(report_path, index=False)

    m = report[report["Kab"] == "AVERAGE"].iloc[0]
    print(f"\n[SUMMARY] Train years: {train_years}")
    print(f"[SUMMARY] calibration={args.calibration} | n_estimators={args.fixed_n_estimators} "
          f"| max_depth={args.fixed_max_depth}")
    print(f"[SUMMARY] OOF  — ROC:{m['OOF_ROC']:.4f} PR:{m['OOF_PR']:.4f} Brier:{m['OOF_Brier']:.4f}")
    print(f"[SUMMARY] Test — ROC:{m['Test_ROC']:.4f} PR:{m['Test_PR']:.4f} Brier:{m['Test_Brier']:.4f}")
    print(f"[SUMMARY] Avg PR lift: {m['PR_lift']:.2f}x")

    print("\n[4/4] Menyimpan prediksi per tahun...")
    year_dfs = {y: df_train_all[df_train_all["year"] == y].copy() for y in train_years}
    year_dfs[args.year_test] = df_test_all

    for y, df_out in sorted(year_dfs.items()):
        if df_out["p_rentan"].isna().any():
            df_out["p_rentan"] = df_out.groupby("kode_kab_new")["p_rentan"].transform(
                lambda x: x.fillna(x.median()))
            df_out["p_rentan"] = df_out["p_rentan"].fillna(0.5)
        if "p_rentan_raw" in df_out.columns and df_out["p_rentan_raw"].isna().any():
            df_out["p_rentan_raw"] = df_out.groupby("kode_kab_new")["p_rentan_raw"].transform(
                lambda x: x.fillna(x.median()))
            df_out["p_rentan_raw"] = df_out["p_rentan_raw"].fillna(0.5)
        fname = f"susenas_prentan_rf_fixed_{calib_tag}_{y}.csv"
        df_out.to_csv(os.path.join(args.output, fname), index=False)
        print(f"  Saved {y}: {len(df_out):,} baris → {fname}")

    print(f"\n[SELESAI] Report: {report_path} | Models: {model_dir}")


if __name__ == "__main__":
    main()