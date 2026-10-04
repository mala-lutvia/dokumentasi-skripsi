import argparse # membaca pilihan yang ditulis setelah nama script di CLI 
import os # untuk memeriksa file, membaca file, dan membuat folder keluaran
import gc # meminta python membersihkan objek besar yang sudah tidak dipakai
import re # membersihkan nama kolom dengan pola teks
import sys # mengatur batas ukuran field saat membaca CSV
import csv
import warnings # menyebunyikan pesan peringatan
import numpy as np # mengolah data dan tabel
import pandas as pd
from functools import reduce # menggabungkan beberapa tabel hasil olahan satu per satu

warnings.filterwarnings("ignore")

try:
    import pyarrow.parquet as pq # membaca file parquet
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

# Bypass csv field size limit
_max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(_max_int)
        break
    except OverflowError:
        _max_int = int(_max_int / 10)
# untuk konsistensi atau harmonisasi variabel antartahun karena nama kolom pada
# kuesioner Susenas bisa berubah antar tahun
COLUMN_MAPPING = {
    2025: dict(
        c_prov='PROP', c_kab='KAB', c_sch='R611', c_ijz='R615',
        c_kawin='R404', c_sex='R405', c_age='R407', c_krt = 'R403', c_jk_krt = 'R405',
        c_sec='R705', c_stat='R706', c_jam='R707',
        c_house='R1602', c_floor='R1608', c_wall='R1607', c_roof='R1606',
        c_water='R1610A', c_elec='R1616B1', c_light='R1616', c_fuel='R1617',
        c_toiletA='R1609A', c_toiletB='R1609B', c_toiletC='R1609C', c_luas='R1604', asset_prefix='R1801',
        c_smart='R802', c_family='R1601', c_internet='R808',
    ),
    2024: dict(
        c_prov='R101', c_kab='R102', c_sch='R610', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407', c_krt = 'R403', c_jk_krt = 'R405',
        c_sec='R706', c_stat='R707', c_jam='R708',
        c_house='R1802', c_floor='R1808', c_wall='R1807', c_roof='R1806A',
        c_water='R1810A', c_elec='R1816B1', c_light='R1816', c_fuel='R1817',
        c_toiletA='R1809A', c_toiletB='R1809B', c_toiletC='R1809C', c_luas='R1804', asset_prefix='R2001',
        c_smart='R802', c_family='R1801', c_internet='R808',
    ),
    2023: dict(
        c_prov='R101', c_kab='R102', c_sch='R610', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407', c_krt = 'R403', c_jk_krt = 'R405',
        c_sec='R706', c_stat='R707', c_jam='R708',
        c_house='R1802', c_floor='R1808', c_wall='R1807', c_roof='R1806',
        c_water='R1810A', c_elec='R1816B1', c_light='R1816', c_fuel='R1817',
        c_toiletA='R1809A', c_toiletB='R1809B', c_toiletC='R1809C', c_luas='R1804', asset_prefix='R2001',
        c_smart='R802', c_family='R1801', c_internet='R808',
    ),
    2022: dict(
        c_prov='R101', c_kab='R102', c_sch='R610', c_ijz='R614',
        c_kawin='R404', c_sex='R405', c_age='R407', c_krt = 'R403', c_jk_krt = 'R405',
        c_sec='R705', c_stat='R706', c_jam='R707',
        c_house='R1802', c_floor='R1808', c_wall='R1807', c_roof='R1806',
        c_water='R1810A', c_elec='R1816B1', c_light='R1816', c_fuel='R1817',
        c_toiletA='R1809A', c_toiletB='R1809B', c_toiletC='R1809C', c_luas='R1804', asset_prefix='R2001',
        c_smart='R802', c_family='R1801', c_internet='R808',
    ),
}

EDU_LIST = ['h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma', 'h_ngrad_d', 'h_ngrad_s', 'h_notgrad']
SCHOOL_LIST = ['h_neverschool', 'h_stillschool', 'h_notschool']
SEC_LIST = [f'h_sec{s}_stat{st}' for s in [1, 2, 3] for st in range(1, 7)]
ASSET_MAP = {
    'A': 'lpg5kg', 'B': 'fridge', 'C': 'ac', 'D': 'wheater', 'E': 'phone',
    'F': 'computer', 'G': 'jewelry', 'H': 'motorcycle', 'I': 'boat',
    'J': 'motorboat', 'K': 'car', 'L': 'tv', 'M': 'land',
}
CATEGORIES = {
    'house': ['h_house1', 'h_house2', 'h_house3', 'h_house4'],
    'floor': ['h_floor1', 'h_floor2', 'h_floor3', 'h_floor4'],
    'wall': ['h_wall1', 'h_wall2', 'h_wall3', 'h_wall4'],
    'roof': ['h_roof1', 'h_roof2', 'h_roof3', 'h_roof4'],
    'dwater': ['h_dwater1', 'h_dwater2', 'h_dwater3', 'h_dwater4', 'h_dwater5'],
    'epower': ['h_epower1', 'h_epower2', 'h_epower3'],
    'lighting': ['h_lighting1', 'h_lighting2', 'h_lighting3'],
    'cookingfuel': ['h_cookingfuel1', 'h_cookingfuel2', 'h_cookingfuel3', 'h_cookingfuel4', 'h_cookingfuel5'],
    'toiltype': [f'h_toiltype{i}' for i in range(1, 7)],
    'septic': [f'h_septic{i}' for i in range(1, 6)]
}

# menebak pemisah CSV berdasarkan baris pertama file, jika ada | maka pakai |, jika ada ; dan jumlah ; lebih banyak dari , maka pakai ;, selain itu pakai ,
def detect_separator(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
    if '|' in first_line: return '|'
    if ';' in first_line and first_line.count(';') > first_line.count(','): return ';'
    return ','

# 1. mengubah nama menjadi teks dengan str(c)
# 2. menghapus karakter bawaan yang bukan huruf, angka, atau garis bawah
# 3. mengubah huruf menjadi kapital
# 4. akhiran _x dihapus
# membersihkan nama columns misalnya menjadikan huruf kapital dan menghapus karakter khusus
def normalize_columns(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', str(c)).strip().upper() for c in df.columns]
    df = df.rename(columns={c: c[:-2] for c in df.columns if c.endswith('_X')})
    return df
# mencari kolom ID rumah tangga yang namanya mengandung URUT, IDRT -> lalu menyamakan menjadi URUT
def normalize_urut(df):
    u_col = next((c for c in df.columns if 'URUT' in c or 'IDRT' in c or 'ID_RUTA' in c), None)
    if u_col and u_col != 'URUT':
        df.rename(columns={u_col: 'URUT'}, inplace=True)
    if 'URUT' in df.columns:
        df['URUT'] = pd.to_numeric(df['URUT'], errors='coerce').fillna(0).astype(np.int64)
    return df

# membaca file CSV atau Parquet
def load_file(path):
    if not path or not os.path.exists(path): return None
    if path.endswith('.parquet'):
        if not PARQUET_AVAILABLE: return None
        df = pd.read_parquet(path)
    else:
        sep = detect_separator(path)
        df = pd.read_csv(path, sep=sep, dtype=str, on_bad_lines='skip', low_memory=False)
    return normalize_columns(df)

# membagi data menjadi 10 bagian desil
def compute_decile_ranks(vals):
    n = len(vals) # jumlah rumah tangga
    if n < 10: return np.zeros(n, dtype=int)
    sorted_idx = np.argsort(vals) # mengurutkan dari nilai terkecil hingga terbesar
    ranks = np.empty(n, dtype=int) # membuat tempat untuk menyimpan nomor desil setiap rumah tangga
    d_size = n // 10 # menghitung jumlah rumah tangga per desil
    for i in range(10):
        start = i * d_size
        end = (i + 1) * d_size if i < 9 else n
        ranks[sorted_idx[start:end]] = i
    return ranks

# menghitung ART per kategori misal kategori pendidikan, sektor, dan lain-lain
def get_wide(df, col, categories=None):
    if col not in df.columns or df.empty: return pd.DataFrame(columns=['URUT'])
    temp = df[['URUT', col]].copy() # ambil hanya URUT dan kolom kategori
    temp[col] = temp[col].astype(str).replace(['nan', 'None', '', ' '], 'unknown') # ganti kategori yang kosong
    counts = temp.groupby(['URUT', col], observed=False).size().reset_index(name='count') # menghitung banyaknya anggota per rumah tangga dan kategori
    wide = counts.pivot(index='URUT', columns=col, values='count').fillna(0) # mengubah nama kategori menjadi nama kolom
    if categories:
        for c in categories:
            if c not in wide.columns: wide[c] = 0
        wide = wide[categories]
    if 'unknown' in wide.columns: wide = wide.drop(columns=['unknown']) # membuang kolom unknown
    return wide.reset_index() # mengembalikan URUT menjadi kolom biasa

# fungsi ini membaca argumen -- map yang berbentuk nama_internal=nama_kolom_file
def parse_custom_map(items):
    out = {}
    if not items: return out
    for it in items:
        if '=' not in it: continue
        k, v = it.split('=', 1)
        out[k.strip()] = v.strip()
    return out

# mengambil mapping bawaaan untuk tahun yang terpilih
def get_column_mapping(year, custom_map=None):
    base = COLUMN_MAPPING.get(year, {}).copy()
    if custom_map: base.update(custom_map) # jika ada custom_map maka mengganti nilai mapping terkait
    return base if base else None 
# Contoh: untuk tahun 2022, mapping bawaan menetapkan c_prov='R101'.
# Jika CLI memberi c_prov=PROP, nilai yang dipakai menjadi PROP

# menggabungkan KOR dan KP
def join_kor_kp(kor_path, kp_path, output_folder=".", year=None):
    print("\n[JOIN] KOR + KP")
    df_kor = load_file(kor_path)
    df_kp = load_file(kp_path)
    if df_kor is None or df_kp is None: return None
    df_kor = normalize_urut(df_kor)
    df_kp = normalize_urut(df_kp)
    # mengambil kolom yang ada di KP tapi tidak ada di KOR, lalu menambahkan kolom URUT untuk digabungkan
    cols_to_use = df_kp.columns.difference(df_kor.columns).tolist() + ['URUT']
    # mempertahankan semua baris KOR dan menempelkan kolom KP yang cocok berdasarkan URUT
    df_raw_ruta = df_kor.merge(df_kp[cols_to_use], on='URUT', how='left')
    out_path = os.path.join(output_folder, f"raw_ruta_{year}.csv" if year else "raw_ruta.csv")
    df_raw_ruta.to_csv(out_path, index=False)
    print(f" -> {out_path} | total {len(df_raw_ruta):,}")
    return out_path

# menambahkan data rumah tangga ke data individu berdasarkan URUT, misal menambahkan data rumah tangga ke data anggota rumah tangga
def merge_ruta_individu(ruta_path, individu_path, output_folder=".", year=None):
    print(f"\n[MERGE] Ruta + Individu")
    df_indiv = load_file(individu_path)
    df_indiv = normalize_urut(df_indiv)
    df_ruta = load_file(ruta_path)
    df_ruta = normalize_urut(df_ruta).drop_duplicates(subset=['URUT'])
    # memilih kolom ruta yang belum tersedia di file individu, lalu menambahkan kolom URUT untuk digabungkan
    cols_to_use = list(df_ruta.columns.difference(df_indiv.columns)) + ['URUT']
    merged = df_indiv.merge(df_ruta[cols_to_use], on='URUT', how='left')
    out_path = os.path.join(output_folder, f"{str(year)[-2:]}_susenas_merged_{year}.csv" if year else "susenas_merged.csv")
    merged.to_csv(out_path, index=False)
    print(f" -> BERHASIL! {out_path}")
    return out_path

# mengolah data individu menjadi fitur rumah tangga
def process_block(df, col, year):
    c = col
    target_candidates = ['KAPITA', 'PCEXP', 'KAPITA_X']
    found = next((t for t in target_candidates if t in df.columns), None)
    df['pcexp'] = pd.to_numeric(df[found], errors='coerce').fillna(0) if found else 0
    must_numeric = [c['c_sec'], c['c_stat'], c['c_jam'], c['c_age'], c['c_ijz'], c['c_light'], c['c_elec'], c['c_prov'], c['c_kab'], c['c_toiletA'], c['c_toiletB'], c['c_toiletC'], c['c_luas']]
    for cn in must_numeric:
        if cn in df.columns: df[cn] = pd.to_numeric(df[cn], errors='coerce').fillna(0)

    if c['c_sch'] in df.columns:
        sch = pd.to_numeric(df[c['c_sch']], errors='coerce')
        df['school'] = np.select([sch == 1, sch == 2, sch == 3], ['h_neverschool', 'h_stillschool', 'h_notschool'], default=None)
    if c['c_ijz'] in df.columns:
        ijz = pd.to_numeric(df[c['c_ijz']], errors='coerce').fillna(25)
        df['ijazah'] = np.select(
            [(ijz >= 1) & (ijz <= 5), (ijz > 5) & (ijz <= 10), (ijz > 10) & (ijz <= 17), ijz.isin([18, 19]), ijz.isin([20, 21, 22, 23, 24]), ijz.isin([25])],
            ['h_ngrad_sd', 'h_ngrad_smp', 'h_ngrad_sma', 'h_ngrad_d', 'h_ngrad_s', 'h_notgrad'], default='h_notgrad'
        )
    if c['c_kawin'] in df.columns:
        df['marriage'] = pd.to_numeric(df[c['c_kawin']], errors='coerce').map({1: 'h_notmarried', 2: 'h_married', 3: 'h_divorced', 4: 'h_widowed'})
    if c['c_sex'] in df.columns:
        df['gender'] = pd.to_numeric(df[c['c_sex']], errors='coerce').map({1: 'h_nmale', 2: 'h_nfemale'})
    if c['c_age'] in df.columns:
        age = df[c['c_age']]
        df['age_cat'] = np.select([age <= 4, age <= 19, age <= 64, age > 64], ['h_nage04', 'h_nage519', 'h_nage2064', 'h_nage65up'], default=None)
    if c['c_krt'] in df.columns and c['c_jk_krt'] in df.columns:
        krt = pd.to_numeric(df[c['c_krt']], errors='coerce')
        jk_krt = pd.to_numeric(df[c['c_jk_krt']], errors='coerce')
        df['h_krt_female'] = np.where((krt == 1) & (jk_krt == 2), 1, 0)
    if c['c_stat'] in df.columns:
        df['work_status'] = pd.to_numeric(df[c['c_stat']], errors='coerce').map({1: 'h_stat1', 2: 'h_stat2', 3: 'h_stat2', 4: 'h_stat3', 5: 'h_stat4', 6: 'h_stat5'})
    if c['c_sec'] in df.columns and c['c_stat'] in df.columns:
        sec = pd.to_numeric(df[c['c_sec']], errors='coerce')
        stat = pd.to_numeric(df[c['c_stat']], errors='coerce')
        df['sector'] = np.nan
        m1 = (sec <= 6)
        df.loc[m1 & (stat == 1), 'sector'] = 'h_sec1_stat1'
        df.loc[m1 & (stat == 2), 'sector'] = 'h_sec1_stat2'
        df.loc[m1 & (stat == 3), 'sector'] = 'h_sec1_stat3'
        df.loc[m1 & (stat.isin([4,5])), 'sector'] = 'h_sec1_stat4'
        df.loc[m1 & (stat.isin([6,7])), 'sector'] = 'h_sec1_stat5'
        df.loc[m1 & (stat == 8), 'sector'] = 'h_sec1_stat6'
        m2 = sec.isin([7, 8, 9, 11])
        df.loc[m2 & (stat == 1), 'sector'] = 'h_sec2_stat1'
        df.loc[m2 & (stat == 2), 'sector'] = 'h_sec2_stat2'
        df.loc[m2 & (stat == 3), 'sector'] = 'h_sec2_stat3'
        df.loc[m2 & (stat.isin([4,5])), 'sector'] = 'h_sec2_stat4'
        df.loc[m2 & (stat.isin([6,7])), 'sector'] = 'h_sec2_stat5'          
        df.loc[m2 & (stat == 8), 'sector'] = 'h_sec2_stat6'
        # Logika Sektor 3 (Sektor >= 12 ATAU Sektor >= 10)
        # Di Python: (A | B & C) berarti A OR (B AND C)
        def apply_m3(s_val):
            mask = (sec >= 12) | (sec >= 10) & (stat == s_val)
            if s_val in [4, 5]: mask = (sec >= 12) | (sec >= 10) & (stat.isin([4, 5]))
            if s_val in [6, 7]: mask = (sec >= 12) | (sec >= 10) & (stat.isin([6, 7]))
            return mask
        df.loc[(sec >= 12) | (sec >= 10) & (stat == 1), 'sector'] = 'h_sec3_stat1'
        df.loc[(sec >= 12) | (sec >= 10) & (stat == 2), 'sector'] = 'h_sec3_stat2'
        df.loc[(sec >= 12) | (sec >= 10) & (stat == 3), 'sector'] = 'h_sec3_stat3'
        df.loc[(sec >= 12) | (sec >= 10) & (stat.isin([4, 5])), 'sector'] = 'h_sec3_stat4'
        df.loc[(sec >= 12) | (sec >= 10) & (stat.isin([6, 7])), 'sector'] = 'h_sec3_stat5'
        df.loc[(sec >= 12) | (sec >= 10) & (stat == 8), 'sector'] = 'h_sec3_stat6'
    
    # menghitung ringkasan jumlah anggota rumah tangga
    agg_dict = {'h_hhcount': ('URUT', 'size')}
    if c['c_jam'] in df.columns: agg_dict['h_avg_workhours'] = (c['c_jam'], 'mean')
    indiv_base = df.groupby('URUT').agg(**agg_dict).reset_index()

    indiv_dfs = [indiv_base]
    if 'h_krt_female' in df.columns:
        indiv_dfs.append(
            df.groupby('URUT')['h_krt_female']
            .max()
            .reset_index()
        )
    for col_name, cats in [('ijazah', EDU_LIST), ('sector', SEC_LIST), ('school', SCHOOL_LIST), ('marriage', None), ('gender', None), ('age_cat', None)]:
        if col_name in df.columns: indiv_dfs.append(get_wide(df, col_name, categories=cats))
    
    indiv_final = reduce(lambda l, r: pd.merge(l, r, on='URUT', how='left'), indiv_dfs).fillna(0)
    # rt mengambil satu baris per rumah tangga untuk variabel yang nilainya 
    # memang dicatat pada tingkat rumah tangga
    rt = df.drop_duplicates(subset=['URUT']).copy()
    rt['lpcexp'] = np.log1p(rt['pcexp'])
    rt['h_nfamily'] = pd.to_numeric(rt.get(c['c_family'], 1), errors='coerce').fillna(1)
    rt['h_luaslantai'] = pd.to_numeric(rt.get(c['c_luas'], 0), errors='coerce').fillna(0)
    rt['h_lnluaslantai'] = np.log1p(rt['h_luaslantai'].replace(0, np.nan)).fillna(0)

    asset_cols = []
    for suffix, name in ASSET_MAP.items():
        col_key, col_name = f"{c['asset_prefix']}{suffix}", f'h_asset_{name}'
        # kalau kode aset bernilai 1 fitur hasilnya 1 (memiliki); kalau bukan 1 hasilnya 0 (tidak memiliki)
        rt[col_name] = np.where(pd.to_numeric(rt.get(col_key, 0), errors='coerce') == 1, 1, 0)
        asset_cols.append(col_name)
    # diperiksa per rumah tangga: hasilnya 1 jika setidaknya satu anggota tercatat memiliki atau menggunakan item dengan kode 1
    for col_key, target_col in [('c_smart', 'h_asset_smartphone'),
                                ('c_internet', 'h_asset_internet')]:
        if c[col_key] in df.columns:
            mapped = df.groupby('URUT')[c[col_key]].apply(
                lambda x: 1 if (pd.to_numeric(x, errors='coerce') == 1).any() else 0
            )
            rt = rt.merge(mapped.rename(target_col), on='URUT', how='left')
            rt[target_col] = rt[target_col].fillna(0).astype(int)
        else:
            rt[target_col] = 0

    def s_num(cn): return pd.to_numeric(rt.get(cn, 0), errors='coerce').fillna(0)

    rt['house'] = np.select([s_num(c['c_house']) == 1, s_num(c['c_house']).isin([3, 5]), s_num(c['c_house']) == 2, s_num(c['c_house']) == 4], ['h_house1', 'h_house2', 'h_house3', 'h_house4'], default=None)
    rt['floor'] = np.select([s_num(c['c_floor']) <= 3, s_num(c['c_floor']) == 4, s_num(c['c_floor']).isin([5, 6]), s_num(c['c_floor']) >= 7], ['h_floor1', 'h_floor2', 'h_floor3', 'h_floor4'], default=None)
    rt['wall'] = np.select([s_num(c['c_wall']) == 1, s_num(c['c_wall']).isin([2, 3]), s_num(c['c_wall']).isin([4, 6]), s_num(c['c_wall']) == 7], ['h_wall1', 'h_wall2', 'h_wall3', 'h_wall4'], default=None)
    rt['roof'] = np.select([s_num(c['c_roof']) == 1, s_num(c['c_roof']) == 2, s_num(c['c_roof']).isin([3, 4, 5, 6]), s_num(c['c_roof']).isin([7, 8])], ['h_roof1', 'h_roof2', 'h_roof3', 'h_roof4'], default=None)
    rt['dwater'] = np.select([s_num(c['c_water']) == 1, s_num(c['c_water']).isin([2, 3]), s_num(c['c_water']).isin([4, 5, 7]), s_num(c['c_water']).isin([6, 8]), s_num(c['c_water']) >= 9], ['h_dwater1', 'h_dwater2', 'h_dwater3', 'h_dwater4', 'h_dwater5'], default=None)
    rt['epower'] = np.select([s_num(c['c_elec']) == 1, s_num(c['c_elec']) == 2, s_num(c['c_elec']) >= 3], ['h_epower1', 'h_epower2', 'h_epower3'], default=None)
    rt['lighting'] = np.select([s_num(c['c_light']) <= 2, s_num(c['c_light']) == 3, s_num(c['c_light']) == 4], ['h_lighting1', 'h_lighting2', 'h_lighting3'], default=None)
    rt['cookingfuel'] = np.select([s_num(c['c_fuel']) == 1, s_num(c['c_fuel']) == 3, s_num(c['c_fuel']).isin([2, 4, 6]), s_num(c['c_fuel']).isin([7, 8, 9, 10, 11]), s_num(c['c_fuel']) == 0], ['h_cookingfuel1', 'h_cookingfuel2', 'h_cookingfuel3', 'h_cookingfuel4', 'h_cookingfuel5'], default=None)
    tA, tB, tC = s_num(c['c_toiletA']), s_num(c['c_toiletB']), s_num(c['c_toiletC'])
    rt['toiltype'] = np.nan
    rt.loc[(tA == 1) & (tB == 1), 'toiltype'] = 'h_toiltype1'
    rt.loc[(tA.isin([2,3])) & (tB == 1), 'toiltype'] = 'h_toiltype2'
    rt.loc[(tA == 4) & (tB == 1), 'toiltype'] = 'h_toiltype3'
    rt.loc[(tA == 1) & (tB == 2), 'toiltype'] = 'h_toiltype4'
    rt.loc[(tA.isin([2,3])) & (tB == 2), 'toiltype'] = 'h_toiltype5'
    rt.loc[(tA == 4) & (tB == 2), 'toiltype'] = 'h_toiltype6'
    rt['septic'] = np.nan
    rt.loc[(tA == 1) & (tC == 1), 'septic'] = 'h_septic1'
    rt.loc[(tA.isin([2,3])) & (tC == 1), 'septic'] = 'h_septic2'
    rt.loc[(tA == 4) & (tC == 1), 'septic'] = 'h_septic3'
    rt.loc[(tA == 1) & (tC == 2), 'septic'] = 'h_septic4'
    rt.loc[(tA.isin([2,3])) & (tC == 2), 'septic'] = 'h_septic5'

    id_cols = ['URUT', c['c_prov'], c['c_kab'], 'pcexp', 'lpcexp', 'h_luaslantai', 'h_lnluaslantai', 'h_nfamily', 'house', 'floor', 'wall', 'roof', 'dwater', 'epower', 'lighting', 'cookingfuel', 'toiltype', 'septic']
    return pd.merge(indiv_final, rt[id_cols + asset_cols + ['h_asset_internet', 'h_asset_smartphone']], on='URUT', how='outer').fillna(0)

def preprocess(merged_path, year, col, output_folder=".", provinsi_filter=None):
    df = load_file(merged_path)
    if df is None: return None
    # merapikan kolom ID rumah tangga -> mengubah menjadi URUT
    df = normalize_urut(df)

    # Clean special
    for c_fix in [col['c_sec'], col['c_stat'], col['c_elec']]:
        if c_fix in df.columns:
            df[c_fix] = df[c_fix].astype(str).str.strip().replace([''], '1')
    
    if provinsi_filter:
        df = df[pd.to_numeric(df[col['c_prov']], errors='coerce').isin(provinsi_filter)]
    # memanggil process_block untuk mengolah informasi ART dan kondisi rumah -> tabel fitur dengan URUT sbg ID ruta
    final = process_block(df, col, year)
    
    # Aggregation
    num_cols = final.select_dtypes(include=[np.number]).columns.tolist()
    if 'URUT' in num_cols: num_cols.remove('URUT')
    # kolom yang dianggap menggambarkan ruta, bukan jumlah karakteristik dari seluruh anggotanya.
    rt_level = ['pcexp', 'lpcexp', 'h_luaslantai', 'h_lnluaslantai', col['c_prov'], col['c_kab'], 'h_nfamily', 'h_krt_female']

    # menentukan aturan agregasi untuk setiap kolom: jika kolom ada di rt_level atau merupakan aset, ambil nilai maksimum; jika kolom numerik, jumlahkan; selain itu ambil nilai pertama.
    agg_rules = {}
    for cn in final.columns:
        if cn == 'URUT': continue
        if cn in rt_level or cn.startswith('h_asset_'): agg_rules[cn] = 'max'
        elif cn in num_cols: agg_rules[cn] = 'sum'
        else: agg_rules[cn] = 'first'
    # semua baris dengan URUT yang sama dikelompokkan dan diringkas menjadi satu baris.
    final = final.groupby('URUT').agg(agg_rules).reset_index()
    # pendidikan tertinggi yang ditamatkan itu dalam bentuk proporsi
    for ec in EDU_LIST:
        if ec in final.columns: final[ec] = final[ec] / final['h_hhcount'].replace(0, 1)

    for col_name, cats in CATEGORIES.items():
        if col_name in final.columns: final[col_name] = pd.Categorical(final[col_name], categories=cats)
    # dummy 0/1 dan one hot encoding
    final = pd.get_dummies(final, columns=list(CATEGORIES.keys()), prefix='', prefix_sep='', dtype=int)
    final['kode_kab'] = final[col['c_prov']].astype(int) * 100 + final[col['c_kab']].astype(int)
    # data dikelompokkan menurut kode_kab, lalu untuk setiap kab, nilai pcexp rumah tangganya dikirim ke fungsi compute_decile_ranks untuk menentukan desilnya. Hasilnya disimpan di kolom 'decile'.
    final['decile'] = final.groupby('kode_kab')['pcexp'].transform(lambda x: compute_decile_ranks(x.values))

    out_path = os.path.join(output_folder, f"susenas_clean_{year}.csv")
    final.to_csv(out_path, index=False)
    print(f"DONE: {out_path} | Baris: {len(final):,}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description='Preprocessing Susenas PMT')
    parser.add_argument('--tahun', required=True, type=int)
    parser.add_argument('--output', default='.')
    parser.add_argument('--provinsi', type=int, nargs='*', default=[])
    parser.add_argument('--map', type=str, nargs='*', default=[])
    sk = parser.add_mutually_exclusive_group(required=True)
    sk.add_argument('--kor', type=str)
    sk.add_argument('--ruta', type=str)
    sk.add_argument('--merged', type=str)
    parser.add_argument('--kp', type=str)
    parser.add_argument('--individu', type=str)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    custom_map = parse_custom_map(args.map)
    col = get_column_mapping(args.tahun, custom_map)

    if args.kor:
        raw_ruta = join_kor_kp(args.kor, args.kp, args.output, year=args.tahun)
        merged_path = merge_ruta_individu(raw_ruta, args.individu, args.output, args.tahun) if args.individu else raw_ruta
    elif args.ruta:
        merged_path = merge_ruta_individu(args.ruta, args.individu, args.output, args.tahun)
    else:
        merged_path = args.merged

    preprocess(merged_path, args.tahun, col, args.output, args.provinsi)

if __name__ == '__main__':
    main()