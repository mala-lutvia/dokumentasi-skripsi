"""
Stage 1 — P_Rentan via Logistic Regression
"""

import argparse, os, gc, sys, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

THRESHOLD       = 1.5
RANDOM_STATE    = 42
N_FOLDS_CAL     = 5    # cv untuk kalibrasi & OOF per kab
                       
MIN_TRAIN_ROWS = 50
MIN_CLASS_BOTH = 5
MIN_POS_PER_FOLD = 5   # target minimum sampel kelas minoritas per fold validasi

VALID_PENALTIES = ("l2",)
VALID_SOLVERS   = ("lbfgs",)

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

# n_folds_cal = 5
# min_pos_per_fold = 5 
# Stratified K-Fold dipakai untuk menjaga proporsi kelas dan mengurangi risiko fold tidak memiliki kelas minoritas
def compute_safe_folds(target_folds, min_cls, min_pos_per_fold=MIN_POS_PER_FOLD):
    return max(2, min(target_folds, min_cls // min_pos_per_fold))

# masukin data dan tahun misal -- 2022: output/susenas_clean -> 2022: output/susenas_clean
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

def _base_pipeline(C, penalty, solver):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C, penalty=penalty, solver=solver,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

# fungsi kalibrasi 
def fit_calibrated(X, y, C, penalty, solver, random_state=RANDOM_STATE):
    counts = np.bincount(y.astype(int))
    min_cls = counts.min() if len(counts) >= 2 else 0

    if min_cls >= 2 and len(y) >= 10:
        n_splits = compute_safe_folds(N_FOLDS_CAL, min_cls)
        base = _base_pipeline(C, penalty, solver)
        calibrated = CalibratedClassifierCV(
            base,
            method="sigmoid",
            cv=n_splits,
            ensemble=False,
        )
        calibrated.fit(X, y)
        return calibrated
    else:
        base = _base_pipeline(C, penalty, solver)
        base.fit(X, y)
        return base
# pipeline Logistic Regression tanpa kalibrasi
def make_raw_pipeline(C, penalty, solver):
    return _base_pipeline(C, penalty, solver)

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    roc   = roc_auc_score(y_true, y_prob)          if len(np.unique(y_true)) > 1 else np.nan
    pr    = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    f1    = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    return roc, pr, f1, brier

def process_kab(kab, df_tr, df_te, fixed_C, fixed_penalty, fixed_solver, model_dir,
                 save_models, calibrate=True):
    df_tr = df_tr.copy().reset_index(drop=True)
    df_te = df_te.copy().reset_index(drop=True)
    X_tr, y_tr = df_tr[PREDICTORS].values, df_tr["miskin"].values
    X_te, y_te = df_te[PREDICTORS].values, df_te["miskin"].values

    counts_tr  = np.bincount(y_tr.astype(int)) if len(y_tr) > 0 else np.array([0])
    min_cls    = counts_tr.min() if len(counts_tr) >= 2 else 0
    # menghitung proporsi label 1 pada data train misal, 20% label di train = 1 -> prev_train 0,20
    prev_train = y_tr.mean() if len(y_tr) > 0 else np.nan
    prev_test  = y_te.mean() if len(y_te) > 0 else np.nan
    # membuat tempat kosong untuk prediksi tiap baris train
    # oof dibuat supaya setiap baris train diprediksi oleh model yang tdk dilatih menggunakan baris itu
    oof_prob     = np.full(len(df_tr), np.nan)
    oof_prob_raw = np.full(len(df_tr), np.nan)

    if min_cls >= MIN_CLASS_BOTH and len(df_tr) >= MIN_TRAIN_ROWS:
        n_outer = compute_safe_folds(N_FOLDS_CAL, min_cls)
        skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=RANDOM_STATE)

        for fold_i, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
            try:
                m_raw = make_raw_pipeline(fixed_C, fixed_penalty, fixed_solver)
                m_raw.fit(X_tr[tr_idx], y_tr[tr_idx])
                oof_prob_raw[val_idx] = m_raw.predict_proba(X_tr[val_idx])[:, 1]

                if calibrate:
                    m_cal = fit_calibrated(
                        X_tr[tr_idx], y_tr[tr_idx],
                        fixed_C, fixed_penalty, fixed_solver,
                        random_state=RANDOM_STATE + fold_i,
                    )
                    oof_prob[val_idx] = m_cal.predict_proba(X_tr[val_idx])[:, 1]
                else:
                    oof_prob[val_idx] = oof_prob_raw[val_idx]

            except Exception:
                oof_prob[val_idx]     = prev_train
                oof_prob_raw[val_idx] = prev_train
    else:
        oof_prob[:]     = prev_train
        oof_prob_raw[:] = prev_train
    # menempelkan prediksi OOF ke tabel train
    df_tr["p_rentan"]     = oof_prob
    df_tr["p_rentan_raw"] = oof_prob_raw
    # melatih model final untuk memprediksi test
    test_prob     = np.full(len(df_te), prev_train)
    test_prob_raw = np.full(len(df_te), prev_train)
    try:
        final_raw = make_raw_pipeline(fixed_C, fixed_penalty, fixed_solver)
        final_raw.fit(X_tr, y_tr)
        if len(df_te) > 0:
            # jika kalibrasi tdk aktif
            test_prob_raw = final_raw.predict_proba(X_te)[:, 1]

        if calibrate:
            final_model = fit_calibrated(
                X_tr, y_tr,
                fixed_C, fixed_penalty, fixed_solver,
                random_state=RANDOM_STATE,
            )
            if len(df_te) > 0:
                # berbeda dg OOF, model final ini dialtih memakai seluruh data train Kabupaten A 
                test_prob = final_model.predict_proba(X_te)[:, 1]
            if save_models:
                joblib.dump(final_model, os.path.join(model_dir, f"logreg_{kab}_cal.pkl"))
        else:
            test_prob = test_prob_raw
            if save_models:
                joblib.dump(final_raw, os.path.join(model_dir, f"logreg_{kab}_raw.pkl"))

    except Exception as e:
        print(f"   [WARN] Kab {kab} final model error: {e}")

    df_te["p_rentan"]     = test_prob
    df_te["p_rentan_raw"] = test_prob_raw

    valid_o = ~np.isnan(oof_prob)
    # menilai prediksi OOF train
    m_oof  = compute_metrics(y_tr[valid_o], oof_prob[valid_o]) \
             if valid_o.sum() > 0 and len(np.unique(y_tr[valid_o])) > 1 else (np.nan,)*4
    # menilai prediksi ke tahun test
    m_test = compute_metrics(y_te, test_prob) \
             if len(df_te) > 0 and len(np.unique(y_te)) > 1 else (np.nan,)*4
    # mengembalikan hasil, kode_kab, metrik oof, test, jumlah data, serta prevalensi.
    mrow = ([kab] + list(m_oof) + list(m_test) +
            [len(df_tr), len(df_te), round(prev_train, 4), round(prev_test, 4)])
    return df_tr, df_te, mrow


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1 LogReg — pemodelan p_rentan dengan hyperparameter "
                     "TETAP (fixed), hasil robustness test terpisah "
                     "(tune_hyperparam_per_kab.py). Tidak ada tuning otomatis "
                     "di script ini.")
    parser.add_argument("--data",           nargs="+", required=True, metavar="YEAR:FILE")
    parser.add_argument("--gk",             required=True)
    parser.add_argument("--year_test",      type=int, required=True)
    parser.add_argument("--provinsi",       type=int, nargs="*", default=[])
    parser.add_argument("--output",         default="./output_stage1_logreg")
    parser.add_argument("--no_save_models", action="store_true")
    parser.add_argument("--no_calibrate", action="store_true",
                         help="Skip CalibratedClassifierCV sepenuhnya — p_rentan "
                              "diambil langsung dari model raw.")
    # ── REVISI 7: fixed hyperparameter WAJIB, tidak ada fallback tuning ────
    parser.add_argument("--fixed_C", type=float, required=True,
                         help="[WAJIB] Nilai C hasil robustness test per kabupaten "
                              "(lihat tune_hyperparam_per_kab.py). Tidak ada "
                              "tuning otomatis di script ini.")
    parser.add_argument("--fixed_penalty", choices=list(VALID_PENALTIES), required=True,
                         help="[WAJIB] Penalty hasil robustness test.")
    parser.add_argument("--fixed_solver", choices=list(VALID_SOLVERS), required=True,
                         help="[WAJIB] Solver hasil robustness test.")
    args = parser.parse_args()

    calibrate   = not args.no_calibrate
    calib_tag   = "cal" if calibrate else "raw"

    os.makedirs(args.output, exist_ok=True)
    model_dir = os.path.join(args.output, "models")
    os.makedirs(model_dir, exist_ok=True)

    print(f"[CONFIG] calibrate={calibrate} | N_FOLDS_CAL={N_FOLDS_CAL} (target)")
    print(f"[CONFIG] Fixed hyperparameter (dari robustness test terpisah): "
          f"C={args.fixed_C} penalty={args.fixed_penalty} solver={args.fixed_solver}")

    print("\n[1/3] Loading GK & Susenas...")
    gk_data  = load_gk(args.gk)
    data_map = parse_data_args(args.data)
    if args.year_test not in data_map:
        print(f"[ERROR] year_test {args.year_test} tidak ada di --data"); sys.exit(1)

    pool = {}
    for y, p in sorted(data_map.items()):
        if not os.path.exists(p): print(f"  [WARN] {p} skip"); continue
        df = load_year(p, y)
        df = attach_label(df, gk_data)
        df = ensure_predictors(df)
        pool[y] = df
        print(f"  {y}: {len(df):,} baris | miskin={df['miskin'].mean():.1%}")

    train_years  = [y for y in sorted(pool.keys()) if y != args.year_test]
    df_train_all = pd.concat([pool[y] for y in train_years], ignore_index=True)
    df_test_all  = pool[args.year_test].copy()

    if args.provinsi:
        prov_set = set()
        for p in args.provinsi: prov_set.add(p); prov_set.update(PROV_EXPAND.get(p, []))
        df_train_all = df_train_all[df_train_all["kode_prov_new"].isin(prov_set)].copy()
        df_test_all  = df_test_all[df_test_all["kode_prov_new"].isin(prov_set)].copy()

    print(f"\n  Train {train_years}: {len(df_train_all):,} baris | miskin={df_train_all['miskin'].mean():.1%}")
    print(f"  Test  {args.year_test}: {len(df_test_all):,} baris | miskin={df_test_all['miskin'].mean():.1%}")

    kabs = sorted(df_train_all["kode_kab_new"].unique())
    print(f"\n[2/3] OOF per kab (fixed_C={args.fixed_C}, penalty={args.fixed_penalty}, "
          f"{len(kabs)} kab, calibrate={calibrate})...")

    df_train_all = df_train_all.reset_index(drop=True)
    df_test_all  = df_test_all.reset_index(drop=True)
    df_train_all["p_rentan"] = np.nan
    df_test_all["p_rentan"]  = np.nan
    df_train_all["p_rentan_raw"] = np.nan
    df_test_all["p_rentan_raw"]  = np.nan

    all_metrics = []
    for i, kab in enumerate(kabs):
        mask_tr = df_train_all["kode_kab_new"] == kab
        mask_te = df_test_all["kode_kab_new"]  == kab
        df_k_tr = df_train_all[mask_tr]
        df_k_te = df_test_all[mask_te]
        if len(df_k_tr) < 10: continue

        df_out_tr, df_out_te, mrow = process_kab(
            kab, df_k_tr, df_k_te, args.fixed_C, args.fixed_penalty, args.fixed_solver,
            model_dir, not args.no_save_models, calibrate=calibrate)

        df_train_all.loc[mask_tr, "p_rentan"] = df_out_tr["p_rentan"].values
        df_train_all.loc[mask_tr, "p_rentan_raw"] = df_out_tr["p_rentan_raw"].values
        if not df_k_te.empty:
            df_test_all.loc[mask_te, "p_rentan"] = df_out_te["p_rentan"].values
            df_test_all.loc[mask_te, "p_rentan_raw"] = df_out_te["p_rentan_raw"].values
        all_metrics.append(mrow)

        if (i+1) % 50 == 0 or (i+1) == len(kabs):
            roc_m = np.nanmean([m[5] for m in all_metrics])
            pr_m  = np.nanmean([m[6] for m in all_metrics])
            print(f"  [{i+1:3d}/{len(kabs)}] avg Test ROC={roc_m:.3f} PR={pr_m:.3f}")

    print("\n[3/3] Menyimpan hasil...")
    cols = ["Kab",
            "OOF_ROC","OOF_PR","OOF_F1","OOF_Brier",
            "Test_ROC","Test_PR","Test_F1","Test_Brier",
            "N_Train","N_Test","Prev_Train","Prev_Test"]
    report = pd.DataFrame(all_metrics, columns=cols).replace([np.inf,-np.inf], np.nan)
    report["PR_lift"] = report["Test_PR"] / report["Prev_Test"].replace(0, np.nan)

    avg = report.mean(numeric_only=True).to_frame().T
    avg["Kab"] = "AVERAGE"
    report = pd.concat([report, avg], ignore_index=True)

    report_path = os.path.join(
        args.output, f"report_stage1_logreg_fixed_{calib_tag}_{args.year_test}.csv")
    report.to_csv(report_path, index=False)

    m = report[report["Kab"] == "AVERAGE"].iloc[0]
    print(f"\n[SUMMARY] Train years: {train_years} | Fixed C={args.fixed_C} | "
          f"penalty={args.fixed_penalty} | solver={args.fixed_solver}")
    print(f"[SUMMARY] calibrate={calibrate} | N_FOLDS_CAL(target)={N_FOLDS_CAL}")
    print(f"[SUMMARY] OOF  — ROC:{m['OOF_ROC']:.4f} PR:{m['OOF_PR']:.4f} Brier:{m['OOF_Brier']:.4f}")
    print(f"[SUMMARY] Test — ROC:{m['Test_ROC']:.4f} PR:{m['Test_PR']:.4f} Brier:{m['Test_Brier']:.4f}")
    print(f"[SUMMARY] Avg PR lift: {m['PR_lift']:.2f}x")

    year_dfs = {y: df_train_all[df_train_all["year"]==y].copy() for y in train_years}
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
        fname = f"susenas_prentan_logreg_fixed_{calib_tag}_{y}.csv"
        df_out.to_csv(os.path.join(args.output, fname), index=False)
        print(f"  Saved {y}: {len(df_out):,} baris → {fname}")

    print(f"\n[SELESAI] Report: {report_path} | Models: {model_dir}")


if __name__ == "__main__":
    main()