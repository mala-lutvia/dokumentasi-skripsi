"""
No Tuning Ridge Regression PMT per-kabupaten — Single-Stage & Two-Stage
========================================================================
Linear Regression + L2 Regularization (Ridge) sebagai ML linear baseline.
Fixed params setara XGBoost Siti Mariyah (2023) — NO TUNING.

Mapping parameter (Hoerl & Kennard 1970, Hastie et al. 2009 ESL Ch.3):
  reg_lambda=0.01  (XGB)  → alpha=0.01    (Ridge L2 penalty)
  n_estimators=700 (XGB)  → max_iter=1000 (konvergensi solver)
  —                       → solver='lsqr' (efisien untuk high-dim data,
                                           Paige & Saunders 1982)

Kenapa Ridge bukan OLS?
  OLS tidak punya regularisasi → tidak comparable dengan XGB yang punya reg_lambda.
  Ridge = Linear Regression + L2, setara filosofi regularisasi XGB.
  Ref: Hoerl & Kennard (1970) "Ridge Regression: Biased Estimation for
       Nonorthogonal Problems", Technometrics, 12(1), 55-67.

Kenapa StandardScaler wajib?
  Ridge sensitif terhadap skala fitur — L2 penalty menghukum koefisien besar.
  Tanpa scaling, fitur skala besar mendominasi penalty → bias estimasi.
  Pipeline memastikan scaler di-fit hanya pada training data (no leakage).
  Ref: Hastie, Tibshirani & Friedman (2009) ESL Ch.3

Evaluasi:
  - Holdout: fit di training pool (semua tahun != year_test), predict ke year_test.
  - CV (OOF, 5-fold) : dijalankan HANYA di training pool, parameter fixed
    (no tuning), untuk bukti stabilitas/generalisasi internal. Tidak menyentuh
    data year_test sama sekali. Hasil OOF dari semua fold digabung dulu, baru
    metrik dihitung sekali (bukan rata-rata metrik per fold). Pipeline
    (StandardScaler → Ridge) dibuat ulang tiap fold agar scaler tidak leak
    antar fold.

Perbedaan mode:
  single    → BASE_PREDICTORS (90 fitur)
  two_stage → BASE_PREDICTORS + p_rentan (91 fitur)
             p_rentan sudah ada di file input (output Stage 1)

Output otomatis: ./output_ridge_<mode>_<timedef|notimedef>/

CLI single no-timedef:
  python baseline_ridge.py \
    --data 2022:susenas_clean_2022.csv 2023:susenas_clean_2023.csv \
           2024:susenas_clean_2024.csv 2025:susenas_clean_2025.csv \
    --year_test 2025 --mode single

CLI two-stage timedef:
  python baseline_ridge.py \
    --data 2022:susenas_prentan_2022.csv 2023:susenas_prentan_2023.csv \
           2024:susenas_prentan_2024.csv 2025:susenas_prentan_2025.csv \
    --year_test 2025 --mode two_stage --timedef gk_full_kab.xlsx

Filter provinsi (opsional):
  --provinsi 11 12 13
"""

import argparse, gc, logging, os, sys, warnings, joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

KODE_MAP = {
    1472:1473, 9401:9501, 9413:9502, 9414:9503, 9415:9504,
    9404:9604, 9411:9608, 9410:9605, 9412:9601, 9433:9607,
    9434:9602, 9435:9606, 9436:9603, 9402:9702, 9417:9708,
    9416:9707, 9418:9704, 9431:9705, 9432:9706, 9430:9703,
    9429:9701, 9107:9202, 9106:9203, 9108:9201, 9109:9205,
    9110:9204, 9171:9271,
}
PROV_EXPAND = {
    91:[91,92], 92:[91,92],
    94:[95,96,97], 95:[95,96,97], 96:[95,96,97], 97:[95,96,97],
}

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
    'h_asset_lpg5kg','h_asset_fridge','h_asset_ac','h_asset_wheater','h_asset_phone',
    'h_asset_computer','h_asset_jewelry','h_asset_motorcycle','h_asset_boat',
    'h_asset_motorboat','h_asset_car','h_asset_tv','h_asset_smartphone','h_asset_land',
    'h_lnluaslantai','h_nfamily','h_notmarried','h_married','h_divorced','h_widowed',
]

# Fixed params wobcke-mariyah-2023-machine-learning-and-data-augmentation-in-the-proxy-means-test-for-poverty-targeting.en.id.
RIDGE_PARAMS = {
    'alpha':    0.001,
    'max_iter': 500,
    'solver':   'auto', # default solver 'auto' menurut scikit learn 1.5.2
}


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_data_args(items):
    return {int(y): p for item in items for y, p in [item.split(":", 1)] if ":" in item}

def load_year(path, year):
    df = pd.read_csv(path)
    p_col = next((c for c in ['PROP','R101'] if c in df.columns), 'R101')
    k_col = next((c for c in ['KAB', 'R102'] if c in df.columns), 'R102')
    if p_col not in df.columns or k_col not in df.columns:
        logger.warning(f"{path}: kolom prov/kab tidak ditemukan.")
    df['year']          = year
    df['kode_kab_new']  = (df[p_col].astype(int)*100 + df[k_col].astype(int)).replace(KODE_MAP)
    df['kode_prov_new'] = df['kode_kab_new'] // 100
    return df

def load_timedef(excel_path):
    df = pd.read_excel(excel_path).dropna(subset=['kode_kab']).copy()
    df['kode_kab'] = df['kode_kab'].astype(int)
    needed = ['timedef_22','timedef_23','timedef_24','timedef_25']
    if missing := [c for c in needed if c not in df.columns]:
        logger.error(f"Kolom timedef tidak ditemukan: {missing}"); sys.exit(1)
    td = df[['kode_kab']+needed].melt(id_vars='kode_kab', value_vars=needed,
                                       var_name='td_col', value_name='timedef')
    td['year'] = td['td_col'].str.extract(r'(\d+)').astype(int) + 2000
    return td[['kode_kab','year','timedef']].dropna()

def compute_decile_ranks(values):
    n = len(values)
    if n < 10: return np.zeros(n, dtype=np.int32)
    arr = np.asarray(values)
    sorted_idx = np.argsort(arr)
    ranks = np.zeros(n, dtype=np.int32)
    d_size = n // 10
    for i in range(10):
        start = i * d_size
        end   = (i+1) * d_size if i < 9 else n
        ranks[sorted_idx[start:end]] = i
    return ranks

def compute_group_logic(df, decile_col='decile', suffix=''):
    df[f'group20{suffix}'] = np.where(df[decile_col] <= 1, 0, 1)
    df[f'group40{suffix}'] = np.where(df[decile_col] <= 3, 0, 1)
    df[f'group60{suffix}'] = np.where(df[decile_col] <= 5, 0, 1)
    return df

def inclusion_exclusion(df, pred_group, ref_group):
    cm = confusion_matrix(df[ref_group], df[pred_group], labels=[0,1])
    tp, fp, fn = cm[0,0], cm[0,1], cm[1,0]
    return (fp/(tp+fp) if (tp+fp)>0 else 0), (fn/(tp+fn) if (tp+fn)>0 else 0)

def metrics_evaluation(df, pred_col='pred_accum', actual_col='pcexp'):
    df = df.copy()
    df['decile_pred'] = compute_decile_ranks(df[pred_col].values)
    df = compute_group_logic(df, 'decile_pred', '_pred')
    tau_dcl = kendalltau(df['decile_pred'], df['decile']).correlation
    tau20   = kendalltau(df['group20_pred'], df['group20']).correlation
    tau40   = kendalltau(df['group40_pred'], df['group40']).correlation
    tau60   = kendalltau(df['group60_pred'], df['group60']).correlation
    ie20,ee20 = inclusion_exclusion(df,'group20_pred','group20')
    ie40,ee40 = inclusion_exclusion(df,'group40_pred','group40')
    ie60,ee60 = inclusion_exclusion(df,'group60_pred','group60')
    acc20 = accuracy_score(df['group20'], df['group20_pred'])
    acc40 = accuracy_score(df['group40'], df['group40_pred'])
    acc60 = accuracy_score(df['group60'], df['group60_pred'])

    # RMSE dihitung GLOBAL dari seluruh baris df (bukan per-subgroup B20/B40/B60).
    # Kolom RMSE20/RMSE40/RMSE60 tetap dipertahankan (struktur report tidak
    # berubah) tapi ketiganya diisi nilai RMSE global yang sama.
    rmse_global = np.sqrt(mean_squared_error(df[actual_col], df[pred_col]))
    rmse20 = rmse_global
    rmse40 = rmse_global
    rmse60 = rmse_global

    return [tau_dcl,tau20,tau40,tau60,ie20,ee20,ie40,ee40,ie60,ee60,
            acc20,acc40,acc60,rmse20,rmse40,rmse60], df

def make_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  Ridge(**RIDGE_PARAMS)),
    ])

N_METRICS = 16  # panjang list yang dikembalikan metrics_evaluation

def oof_cv_metrics(df_tr, target_col, PREDICTORS, model_builder, actual_col,
                    n_splits=5, seed=42):
    """
    Out-of-fold (OOF) CV, cv=5, pada training pool per-kabupaten.
    Parameter fixed (no tuning) — hanya untuk bukti generalisasi internal,
    bukan untuk mencari hyperparameter baru. Tidak pernah menyentuh year_test.
    model_builder dipanggil ulang tiap fold, sehingga scaler Ridge selalu
    di-fit hanya pada training fold (no leakage antar fold).

    Return:
        pooled   : list 16 metrik, dihitung SEKALI dari gabungan prediksi OOF
                   seluruh fold (bukan rata-rata metrik per fold).
        fold_std : list 16 nilai std metrik antar fold, sebagai ukuran
                   variabilitas/stabilitas CV.
    """
    if len(df_tr) < n_splits * 20:
        return [np.nan] * N_METRICS, [np.nan] * N_METRICS

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = df_tr.reset_index(drop=True).copy()
    oof_pred = np.full(len(X), np.nan)
    per_fold_metrics = []

    for train_idx, val_idx in kf.split(X):
        tr_fold, va_fold = X.iloc[train_idx], X.iloc[val_idx].copy()
        model = model_builder()
        model.fit(tr_fold[PREDICTORS], tr_fold[target_col])
        pred = np.expm1(model.predict(va_fold[PREDICTORS]))
        oof_pred[val_idx] = pred
        va_fold['pred_accum'] = pred
        m, _ = metrics_evaluation(va_fold, 'pred_accum', actual_col)
        per_fold_metrics.append(m)

    X['pred_accum'] = oof_pred
    valid = X.dropna(subset=['pred_accum'])
    if valid.empty:
        pooled = [np.nan] * N_METRICS
    else:
        pooled, _ = metrics_evaluation(valid, 'pred_accum', actual_col)

    fold_std = (np.nanstd(np.array(per_fold_metrics), axis=0).tolist()
                if per_fold_metrics else [np.nan] * N_METRICS)
    return pooled, fold_std


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data',      nargs='+', required=True, metavar='YEAR:FILE')
    parser.add_argument('--year_test', type=int,  required=True)
    parser.add_argument('--mode',      choices=['single','two_stage'], required=True,
                        help='single     → BASE_PREDICTORS (90 fitur)\n'
                             'two_stage → BASE_PREDICTORS + p_rentan (91 fitur)')
    parser.add_argument('--timedef',   default=None)
    parser.add_argument('--provinsi',  nargs='*', type=int, default=[])
    parser.add_argument('--output',    default=None)
    parser.add_argument('--no_save_models', action='store_true')
    parser.add_argument('--cv_splits', type=int, default=5,
                        help='Jumlah fold untuk OOF CV internal (default: 5)')
    args = parser.parse_args()

    use_timedef = args.timedef is not None
    mode_label  = 'timedef' if use_timedef else 'notimedef'
    target_col  = 'lntimepcexp' if use_timedef else 'lpcexp'
    actual_col  = 'timepcexp'   if use_timedef else 'pcexp'
    PREDICTORS  = BASE_PREDICTORS + (['p_rentan'] if args.mode == 'two_stage' else [])

    logger.info(f"Ridge No Tuning | mode={args.mode.upper()} | {mode_label.upper()} | "
                f"n_features={len(PREDICTORS)} | cv_splits={args.cv_splits}")
    logger.info(f"alpha={RIDGE_PARAMS['alpha']} — setara reg_lambda=0.01 XGB Siti Mariyah (2023)")
    logger.info("Pipeline: StandardScaler → Ridge (Hoerl & Kennard 1970)")

    out_dir   = args.output or f'./output_ridge_{args.mode}_{mode_label}'
    model_dir = os.path.join(out_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    td_long = load_timedef(args.timedef) if use_timedef else None

    pool = []
    for y, fp in sorted(parse_data_args(args.data).items()):
        if not os.path.exists(fp): logger.warning(f"{fp} tidak ditemukan, skip."); continue
        df = load_year(fp, y)
        if use_timedef:
            df = df.merge(td_long, left_on=['kode_kab_new','year'],
                          right_on=['kode_kab','year'], how='left').dropna(subset=['timedef'])
            df['timepcexp']   = df['pcexp'] * df['timedef']
            df['lntimepcexp'] = np.log1p(df['timepcexp'])
        if 'lpcexp' not in df.columns: df['lpcexp'] = np.log1p(df['pcexp'])
        pool.append(df); logger.info(f"Loaded {y}: {len(df):,} baris")

    full = pd.concat(pool, ignore_index=True); del pool; gc.collect()

    if args.provinsi:
        prov_set = set()
        for p in args.provinsi: prov_set.add(p); prov_set.update(PROV_EXPAND.get(p,[]))
        full = full[full['kode_prov_new'].isin(prov_set)]

    full         = compute_group_logic(full, 'decile')
    df_train_all = full[full['year'] != args.year_test].copy()
    df_test_all  = full[full['year'] == args.year_test].copy()

    if args.mode == 'two_stage' and 'p_rentan' not in df_train_all.columns:
        logger.error("p_rentan tidak ditemukan! Pastikan input adalah output Stage 1."); sys.exit(1)

    for c in PREDICTORS:
        if c not in df_train_all.columns: df_train_all[c] = 0

    # Simpan params yang dipakai untuk dokumentasi
    pd.DataFrame([RIDGE_PARAMS]).to_csv(
        os.path.join(out_dir, f'params_ridge_{args.mode}_{mode_label}.csv'), index=False)

    summary, master_rows = [], []
    kabs = sorted(df_train_all['kode_kab_new'].unique())
    logger.info(f"Processing {len(kabs)} kabupaten...")

    for i, kab in enumerate(kabs):
        df_tr = df_train_all[df_train_all['kode_kab_new'] == kab].copy()
        df_te = df_test_all [df_test_all ['kode_kab_new'] == kab].copy()
        if len(df_tr) < 100 or df_te.empty:
            logger.warning(f"Skip kab {kab} | train={len(df_tr)} test={len(df_te)}")
            continue

        for c in PREDICTORS:
            if c not in df_tr.columns: df_tr[c] = 0
            if c not in df_te.columns: df_te[c] = 0

        # Pipeline fit per kab — scaler di-fit hanya pada training kab ini
        model = make_pipeline()
        model.fit(df_tr[PREDICTORS], df_tr[target_col])

        if not args.no_save_models:
            joblib.dump(model, os.path.join(
                model_dir, f'ridge_{args.mode}_{kab}_{mode_label}.pkl'))

        df_te = df_te.copy()
        df_te['pred_log']      = model.predict(df_te[PREDICTORS])
        df_te['pred_accum']    = np.expm1(df_te['pred_log'])
        df_te['y_hat_rupiah']  = df_te['pred_accum']

        metrics, df_eval = metrics_evaluation(df_te, 'pred_accum', actual_col)

        cv_pooled, cv_std = oof_cv_metrics(df_tr, target_col, PREDICTORS, make_pipeline,
                                            actual_col, n_splits=args.cv_splits)

        summary.append([kab//100, kab] + metrics + cv_pooled + cv_std)

        logger.info(
            f"Kab {kab} | "
            f"tr={len(df_tr):,} te={len(df_te):,} | "
            f"RMSE20={metrics[-3]:.0f} RMSE40={metrics[-2]:.0f} RMSE60={metrics[-1]:.0f} "
            f"TauDCL={metrics[0]:.3f} | CV_TauDCL={cv_pooled[0]:.3f}"
        )
        master_rows.append(df_eval)
        if (i+1) % 50 == 0: logger.info(f"Progress: {i+1}/{len(kabs)}")
        gc.collect()

    COLS = ['Prov','Kab','TauDCL','Tau20','Tau40','Tau60',
            'IE20','EE20','IE40','EE40','IE60','EE60',
            'Acc20','Acc40','Acc60','RMSE20','RMSE40','RMSE60']
    CV_COLS  = ['CV_'  + c for c in COLS[2:]]
    STD_COLS = ['STD_' + c for c in COLS[2:]]
    COLS = COLS + CV_COLS + STD_COLS

    report = pd.DataFrame(summary, columns=COLS).replace([np.inf,-np.inf], np.nan)
    avg = report.mean(numeric_only=True).to_frame().T
    avg['Prov'] = 'AVERAGE'; avg['Kab'] = 'AVERAGE'
    report = pd.concat([report, avg], ignore_index=True)

    rpath = os.path.join(out_dir, f'report_ridge_{args.mode}_{args.year_test}_{mode_label}.csv')
    report.to_csv(rpath, index=False)
    if master_rows:
        pd.concat(master_rows, ignore_index=True).to_csv(
            os.path.join(out_dir, f'master_ridge_{args.mode}_{args.year_test}_{mode_label}.csv'),
            index=False)

    logger.info(f"\n[SELESAI] Ridge {args.mode.upper()} — {mode_label.upper()}")
    logger.info(f"  Output : {out_dir}")
    logger.info(f"  Report : {rpath}")

if __name__ == '__main__':
    main()