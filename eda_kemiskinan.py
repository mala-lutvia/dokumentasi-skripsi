"""
EDA Karakteristik Rumah Tangga — 2 Kelompok Kesejahteraan (v4 — LENGKAP)
=======================================================================

═══════════════════════════════════════════════════════
TIPE VARIABEL & CARA HITUNG (penting untuk interpretasi)
═══════════════════════════════════════════════════════
[COUNT]        Jumlah orang dalam RT (level rumah tangga) → mean()
               → label "Rata-rata Jumlah ART/Keluarga"
               h_hhcount, h_nfamily,
               h_nmale, h_nfemale,
               h_nage04, h_nage519, h_nage2064, h_nage65up

[PROPORSI-ART] Proporsi ART dalam satu RT (0-100%) → mean() (sudah dalam
               skala %, tidak dikali 100 lagi saat divisualisasikan)
               h_neverschool_pct, h_stillschool_pct, h_notschool_pct

[PROPORSI-DIDIK] Proporsi ART menurut ijazah tertinggi (0.0-1.0 di data
               mentah) → mean()×100 → label "Rata-rata Proporsi ART (%)"
               h_ngrad_sd, h_ngrad_smp, h_ngrad_sma, h_ngrad_d, h_ngrad_s,
               h_notgrad

[DUMMY]        Kepemilikan aset 0/1 per RT → mean()×100 → label "% RT
               yang Memiliki"
               h_asset_*

[ONE-HOT]      Fasilitas rumah, mutually exclusive → mean()×100 → stacked
               bar, total=100%
               h_floor*, h_wall*, h_roof*, h_dwater*, h_cookingfuel*,
               h_toiltype*, h_lighting*, h_epower*

[KONTINU]      Nilai numerik asli → mean()
               h_lnluaslantai

CLI:
    python eda_kemiskinan_final_v4.py \\
        --data 2022:susenas_2022.csv 2023:susenas_2023.csv \\
        --gk gk_full_kab.xlsx \\
        --provinsi 33 34 \\
        --output ./output_eda
"""

from __future__ import annotations

import argparse
import gc
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

C_BOT = "#C0504D"  # 40% Terbawah — merah lembut (bata muda)
C_TOP = "#4472C4"  # 60% Teratas  — biru

GROUP_LABELS  = ["bottom40", "top60"]
GROUP_DISPLAY = {"bottom40": "40% Terbawah", "top60": "60% Teratas"}
GROUP_COLORS  = {"bottom40": C_BOT, "top60": C_TOP}
# Posisi baku kelompok pada grafik yang membandingkan dua batang/seri.
GROUP_X = np.array([0.0, 0.65])


def _set_group_axis(ax: plt.Axes) -> None:
    """Atur label dan jarak konsisten untuk sumbu dua kelompok."""
    ax.set_xticks(GROUP_X)
    ax.set_xticklabels([GROUP_DISPLAY[g] for g in GROUP_LABELS])
    ax.set_xlim(-0.50, 1.15)

LEGEND_HANDLES = [
    mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_DISPLAY[g])
    for g in GROUP_LABELS
]

# Palet kualitatif tambahan untuk variabel NON-ordinal (gender, kelompok
# umur, status sekolah) — disamakan gayanya dengan palet pastel akademik
# yang sudah dipakai di chart_07/08/09 (kualitas hunian, air-sanitasi,
# energi), supaya seluruh chart terasa konsisten satu keluarga warna.
PALETTE_KATEGORIKAL = ["#A8D5BA", "#B5D4E8", "#F9E07F", "#F4A96A", "#E8877A", "#D4A5C9"]
C_SCHOOL = PALETTE_KATEGORIKAL[:3]   # 3 status sekolah

# Nama provinsi (BPS): 33 = Jawa Tengah, 34 = DI Yogyakarta
PROV_NAMES = {33: "Jawa Tengah", 34: "DI Yogyakarta"}
# Palet kualitatif ColorBrewer "Dark2" agar berbeda dari palet kelompok
# kesejahteraan di atas.
PROV_COLORS = {33: "#1B9E77", 34: "#D95F02"}

plt.rcParams.update({
    "font.family":            "serif",
    "font.serif":             ["Times New Roman"],
    "axes.spines.top":        False,
    "axes.spines.right":      False,
    "axes.edgecolor":         "#444444",
    "axes.linewidth":         0.8,
    "axes.grid":              True,
    "axes.grid.axis":         "y",
    "grid.color":             "#e8e8e8",
    "grid.linestyle":         "--",
    "grid.linewidth":         0.5,
    "xtick.color":            "#333333",
    "ytick.color":            "#333333",
    "xtick.labelsize":        9.5,
    "ytick.labelsize":        9.5,
    "legend.fontsize":        9,
    "legend.framealpha":      0.9,
    "legend.edgecolor":       "#cccccc",
    "figure.dpi":             300,
    "axes.titlesize":         11,
    "axes.labelsize":         10,
})

SOURCE_TEXT = "Sumber: Susenas, diolah"

VARIABLE_LABELS = {
    "h_asset_smartphone": "Smartphone", "h_asset_car": "Mobil",
    "h_asset_motorcycle": "Motor", "h_asset_computer": "Komputer",
    "h_asset_ac": "AC", "h_asset_fridge": "Kulkas", "h_asset_tv": "TV",
    "h_asset_land": "Tanah", "h_asset_jewelry": "Perhiasan",
    "h_asset_lpg5kg": "LPG 5kg", "h_asset_wheater": "Pemanas Air",
    "h_asset_boat": "Perahu", "h_asset_motorboat": "Perahu Motor",
    "h_asset_phone": "Telepon", "h_ngrad_s": "Tamat Sarjana+",
    "h_ngrad_d": "Tamat Diploma", "h_ngrad_sma": "Tamat SMA",
    "h_ngrad_smp": "Tamat SMP", "h_ngrad_sd": "Tamat SD",
    "h_notgrad": "Tidak Tamat SD", "h_lnluaslantai": "ln(Luas Lantai)",
    "h_hhcount": "Jumlah ART", "h_nfamily": "Jumlah Keluarga",
    "h_nmale": "Jumlah ART Laki-laki", "h_nfemale": "Jumlah ART Perempuan",
    "h_nage04": "Jumlah ART Balita (0–4 Thn)",
    "h_nage519": "Jumlah ART Anak/Remaja (5–19 Thn)",
    "h_nage2064": "Jumlah ART Usia Produktif (20–64 Thn)",
    "h_nage65up": "Jumlah ART Lansia (65+ Thn)",
    "h_neverschool_pct": "Tidak/Belum Pernah Sekolah (%)",
    "h_stillschool_pct": "Masih Sekolah (%)", "h_notschool_pct": "Tidak Sekolah Lagi (%)",
}


def _label(col: str) -> str:
    return VARIABLE_LABELS.get(col, col.replace("h_", "").replace("_", " ").title())


# ─────────────────────────────────────────────────────────────────────────────
# Dataclass Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    threshold: float = 1.5  # disimpan untuk backward-compat

    kode_map: Dict[int, int] = field(default_factory=lambda: {
        1472: 1473,
        9401: 9501, 9413: 9502, 9414: 9503, 9415: 9504,
        9404: 9604, 9411: 9608, 9410: 9605, 9412: 9601,
        9433: 9607, 9434: 9602, 9435: 9606, 9436: 9603,
        9402: 9702, 9417: 9708, 9416: 9707, 9418: 9704,
        9431: 9705, 9432: 9706, 9430: 9703, 9429: 9701,
        9107: 9202, 9106: 9203, 9108: 9201, 9109: 9205,
        9110: 9204, 9171: 9271,
    })
    prov_expand: Dict[int, List[int]] = field(default_factory=lambda: {
        91: [91, 92], 92: [91, 92],
        94: [95, 96, 97], 95: [95, 96, 97],
        96: [95, 96, 97], 97: [95, 96, 97],
    })

    # v4: ukuran RT tetap COUNT (rata-rata jumlah)
    feat_hhsize: List[str] = field(default_factory=lambda: [
        "h_hhcount", "h_nfamily",
    ])
    # v4: gender DIKEMBALIKAN ke COUNT (rata-rata jumlah ART per gender)
    feat_gender_count: List[str] = field(default_factory=lambda: [
        "h_nmale", "h_nfemale",
    ])
    # v4: kelompok umur DIKEMBALIKAN ke COUNT (rata-rata jumlah ART per kelompok umur)
    feat_age_count: List[str] = field(default_factory=lambda: [
        "h_nage04", "h_nage519", "h_nage2064", "h_nage65up",
    ])
    # v4: kolom mentah (count) yang dikonversi jadi proporsi (%) -- HANYA
    # status sekolah, gender & umur tidak lagi dikonversi.
    raw_count_to_pct: List[str] = field(default_factory=lambda: [
        "h_neverschool", "h_stillschool", "h_notschool",
    ])
    # v4: berisi kolom PROPORSI (%) hasil konversi di atas
    feat_school_status: List[str] = field(default_factory=lambda: [
        "h_neverschool_pct", "h_stillschool_pct", "h_notschool_pct",
    ])
    feat_education_level: List[str] = field(default_factory=lambda: [
        "h_ngrad_sd", "h_ngrad_smp", "h_ngrad_sma",
        "h_ngrad_d", "h_ngrad_s", "h_notgrad",
    ])
    feat_house_onehot: List[str] = field(default_factory=lambda: [
        # Status/kategori hunian one-hot (label substantif perlu mengikuti
        # codebook sumber; visualisasi sementara memakai Kategori 1-4).
        "h_house1", "h_house2", "h_house3", "h_house4",
        "h_floor1", "h_floor2", "h_floor3", "h_floor4",
        "h_wall1",  "h_wall2",  "h_wall3",  "h_wall4",
        "h_roof1",  "h_roof2",  "h_roof3",  "h_roof4",
        "h_dwater1","h_dwater2","h_dwater3","h_dwater4","h_dwater5",
        "h_cookingfuel1","h_cookingfuel2","h_cookingfuel3","h_cookingfuel4","h_cookingfuel5",
        "h_toiltype1","h_toiltype2","h_toiltype3","h_toiltype4","h_toiltype5","h_toiltype6",
        "h_lighting1","h_lighting2","h_lighting3",
        "h_epower1","h_epower2","h_epower3",
    ])
    feat_house_cont: List[str] = field(default_factory=lambda: ["h_lnluaslantai"])
    feat_asset: List[str] = field(default_factory=lambda: [
        "h_asset_lpg5kg","h_asset_fridge","h_asset_ac","h_asset_wheater",
        "h_asset_phone","h_asset_computer","h_asset_jewelry",
        "h_asset_motorcycle","h_asset_boat","h_asset_motorboat",
        "h_asset_car","h_asset_tv","h_asset_smartphone","h_asset_land",
        "h_asset_internet",
    ])

    @property
    def feat_house(self) -> List[str]:
        return self.feat_house_onehot + self.feat_house_cont

    @property
    def predictors(self) -> List[str]:
        return (
            self.feat_hhsize + self.feat_gender_count + self.feat_age_count
            + self.feat_school_status
            + self.feat_education_level + self.feat_house + self.feat_asset
        )

    # Variabel yang dipakai buat ranking gap & tabel justifikasi PMT.
    # Sengaja dibatasi ke DUMMY/PROPORSI/KONTINU biar skala "gap" antar
    # variabel bisa dibandingkan apple-to-apple (semua dalam basis %,
    # kecuali h_lnluaslantai yang levelnya beda dan dipisah).
    @property
    def gap_ranking_vars(self) -> List[str]:
        return self.feat_asset + self.feat_education_level + ["h_lnluaslantai"]


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class DataLoader:
    def __init__(self, config: Config) -> None:
        self.cfg = config

    def load_year(self, path: str, year: int) -> pd.DataFrame:
        df    = pd.read_csv(path)
        p_col = self._first_existing(df, ["PROP","R101","IDPROV"], "R101")
        k_col = self._first_existing(df, ["KAB", "R102","IDKAB"],  "R102")
        df["year"]        = year
        df["kode_kab_new"] = (
            df[p_col].astype(int) * 100 + df[k_col].astype(int)
        ).replace(self.cfg.kode_map)
        df["kode_prov_new"] = df["kode_kab_new"] // 100
        df = self.add_proportion_columns(df)
        return df

    def add_proportion_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not all(c in df.columns for c in self.cfg.raw_count_to_pct):
            return df
        # denominator khusus status sekolah (kolom count asli tidak diubah)
        df["school_count"] = df[self.cfg.raw_count_to_pct].sum(axis=1)
        denom = df["school_count"].replace(0, np.nan)   # hindari pembagian 0
        for col in self.cfg.raw_count_to_pct:
            df[f"{col}_pct"] = (df[col] / denom * 100).fillna(0.0)
        return df

    def load_gk(self, excel_path: str) -> pd.DataFrame:
        gk = pd.read_excel(excel_path).dropna(subset=["kode_kab"]).copy()
        gk["kode_kab"] = gk["kode_kab"].astype(int)
        year_cols = [c for c in [2022,2023,2024,2025] if c in gk.columns]
        gk_long = gk[["kode_kab"]+year_cols].melt(
            id_vars="kode_kab", value_vars=year_cols,
            var_name="year", value_name="gk")
        gk_long["year"] = gk_long["year"].astype(int)
        return gk_long

    @staticmethod
    def _first_existing(df, candidates, default):
        return next((c for c in candidates if c in df.columns), default)

    @staticmethod
    def parse_data_args(items: List[str]) -> Dict[int, str]:
        result: Dict[int, str] = {}
        for item in items:
            if ":" not in item:
                continue
            y, p = item.split(":", 1)
            result[int(y)] = p
        return result


# ─────────────────────────────────────────────────────────────────────────────
# PovertyLabeler
# ─────────────────────────────────────────────────────────────────────────────

class PovertyLabeler:
    """
    Label kelompok kesejahteraan per kabupaten, per tahun.
    v4: hanya 2 kelompok (Bottom 40% vs Top 60%), ala World Bank shared
    prosperity indicator -- threshold tunggal di persentil ke-40.
    Kolom baru: pcexp_raw, gk_raw, kelompok (str), miskin (int 0/1).
    """
    def __init__(self, config: Config) -> None:
        self.cfg = config

    def attach_label(self, df: pd.DataFrame, gk_data: pd.DataFrame, year: int) -> pd.DataFrame:
        # Hindari merge seluruh DataFrame (ratusan ribu baris x banyak kolom)
        # yang membuat pandas menyalin blok data besar. Lookup ini hanya
        # menambahkan satu kolom GK berdasarkan kode kabupaten dan tahun.
        relevant_gk = gk_data.loc[
            gk_data["kode_kab"].isin(df["kode_kab_new"].unique())
            & gk_data["year"].isin(df["year"].unique()),
            ["kode_kab", "year", "gk"],
        ]
        gk_lookup = relevant_gk.set_index(["kode_kab", "year"])["gk"]
        if not gk_lookup.index.is_unique:
            duplicated_keys = gk_lookup.index[gk_lookup.index.duplicated()].unique().tolist()
            raise ValueError(
                "Kode kabupaten/tahun yang dipakai punya data GK duplikat: "
                f"{duplicated_keys[:10]}"
            )
        df_keys = pd.MultiIndex.from_arrays(
            [df["kode_kab_new"].to_numpy(), df["year"].to_numpy()],
            names=["kode_kab", "year"],
        )
        df["gk"] = gk_lookup.reindex(df_keys).to_numpy()
        valid_gk = df["gk"].notna()
        if not valid_gk.all():
            df = df.loc[valid_gk].copy()
        df["pcexp_raw"] = df["pcexp"]
        df["gk_raw"]    = df["gk"]

        def _assign(sub):
            p40 = sub["pcexp_raw"].quantile(0.40)
            return pd.Series(
                np.where(sub["pcexp_raw"] <= p40, "bottom40", "top60"),
                index=sub.index)

        df["kelompok"] = df.groupby("kode_kab_new", group_keys=False).apply(_assign)
        df["miskin"]   = df["kelompok"].map({"bottom40": 0, "top60": 1})

        counts = df["kelompok"].value_counts().to_dict()
        print(f"  [CHECK {year}] pcexp_mean={df['pcexp_raw'].mean():,.0f} | "
              f"gk_mean={df['gk_raw'].mean():,.0f} | "
              + " | ".join(f"{g}={counts.get(g,0):,}" for g in GROUP_LABELS))
        return df

    def apply_province_filter(self, df: pd.DataFrame, prov_list: List[int]) -> pd.DataFrame:
        if not prov_list:
            return df
        prov_set: set = set()
        for p in prov_list:
            prov_set.add(p)
            prov_set.update(self.cfg.prov_expand.get(p, []))
        filtered = df[df["kode_prov_new"].isin(prov_set)].copy()
        print(f"\n[INFO] Filter wilayah aktif: {sorted(prov_set)}")
        for y in sorted(filtered["year"].unique()):
            sub  = filtered[filtered["year"] == y]
            dist = sub["kelompok"].value_counts(normalize=True) * 100
            print(f"  Tahun {y}: {len(sub):,} baris | "
                  + " | ".join(f"{g}={dist.get(g,0):.1f}%" for g in GROUP_LABELS))
        return filtered

    def filter_raw_provinces(self, df: pd.DataFrame, prov_list: List[int]) -> pd.DataFrame:
        """Filter data mentah sebelum pelabelan agar tahap berikutnya hemat memori."""
        if not prov_list or "kode_prov_new" not in df.columns:
            return df
        prov_set: set = set()
        for p in prov_list:
            prov_set.add(p)
            prov_set.update(self.cfg.prov_expand.get(p, []))
        return df.loc[df["kode_prov_new"].isin(prov_set)].copy()


# ─────────────────────────────────────────────────────────────────────────────
# SummaryAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

class SummaryAnalyzer:
    def __init__(self, config: Config, output_dir: str) -> None:
        self.cfg        = config
        self.output_dir = output_dir

    def build_national_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        cols  = [c for c in self.cfg.predictors if c in df.columns]
        summ  = df.groupby("kelompok")[cols].mean().T.reindex(columns=GROUP_LABELS)
        summ["gap_bottom_top"]     = summ["bottom40"] - summ["top60"]
        summ["abs_gap_bottom_top"] = summ["gap_bottom_top"].abs()
        summ  = summ.sort_values("abs_gap_bottom_top", ascending=False)
        summ.to_csv(self._p("summary_nasional.csv"))
        return summ

    def build_geo_summaries(self, df: pd.DataFrame) -> None:
        cols = [c for c in self.cfg.predictors if c in df.columns]
        df.groupby(["kode_prov_new","kelompok"])[cols].mean()\
          .to_csv(self._p("summary_provinsi.csv"))
        df.groupby(["kode_kab_new","kelompok"])[cols].mean()\
          .to_csv(self._p("summary_kabupaten.csv"))

    def export_all_tables(self, df: pd.DataFrame) -> None:
        self._export_karakteristik(df)
        self._export_ringkasan(df)
        self._export_art(df)
        self._export_pendidikan(df)
        self._export_perumahan(df)
        self._export_aset(df)
        self._export_ranking_gap(df)
        self._export_justifikasi_pmt(df)
        self._export_provinsi_kelompok(df)
        print("[OK] Semua tabel CSV berhasil dibuat")

    def _export_karakteristik(self, df: pd.DataFrame) -> None:
        cols = [c for c in self.cfg.predictors if c in df.columns]
        rows = [{
            "variabel":         c,
            "bottom40":         df.loc[df["kelompok"]=="bottom40", c].mean(),
            "top60":            df.loc[df["kelompok"]=="top60",    c].mean(),
            "gap_bottom_top":   df.loc[df["kelompok"]=="bottom40", c].mean()
                              - df.loc[df["kelompok"]=="top60",    c].mean(),
        } for c in cols]
        tbl = pd.DataFrame(rows)
        tbl["abs_gap_bottom_top"] = tbl["gap_bottom_top"].abs()
        # NB: kolom "variabel" bercampur skala (count murni, % 0-100, proporsi
        # 0-1, dummy 0-1) — cek header modul untuk tipe tiap variabel sebelum
        # menafsirkan angka mentah di tabel ini.
        tbl.sort_values("abs_gap_bottom_top", ascending=False)\
           .to_csv(self._p("tabel_karakteristik_lengkap.csv"), index=False)

    def _export_ringkasan(self, df: pd.DataFrame) -> None:
        counts = df["kelompok"].value_counts()
        total  = len(df)
        pd.DataFrame({
            "Jumlah":     [total]+[counts.get(g,0) for g in GROUP_LABELS],
            "Persentase": [100.0]+[counts.get(g,0)/total*100 for g in GROUP_LABELS],
        }, index=["Total_RT"]+GROUP_LABELS).to_csv(self._p("ringkasan_kemiskinan.csv"))

    def _export_art(self, df: pd.DataFrame) -> None:
        if "h_hhcount" not in df.columns:
            return
        tmp = df.copy()
        tmp["kat_art"] = pd.cut(tmp["h_hhcount"], bins=[0,4,7,999],
                                labels=["ART≤4","5≤ART≤7","ART>7"])
        (pd.crosstab(tmp["kat_art"], tmp["kelompok"], normalize="index")*100)\
            .reindex(columns=GROUP_LABELS).round(2).to_csv(self._p("tabel_art.csv"))

    def _export_pendidikan(self, df: pd.DataFrame) -> None:
        # h_neverschool_pct/h_stillschool_pct/h_notschool_pct SUDAH dalam
        # skala 0-100 (proporsi terhadap school_count, yaitu jumlah ART yang
        # memiliki status partisipasi sekolah, BUKAN h_hhcount), jadi tidak
        # dikali 100 lagi di sini.
        school_pct_cols = [c for c in self.cfg.feat_school_status if c in df.columns]
        educ_pct_cols   = [c for c in self.cfg.feat_education_level if c in df.columns]
        frames = []
        if school_pct_cols:
            tbl = df.groupby("kelompok")[school_pct_cols].mean().T\
                    .reindex(columns=GROUP_LABELS).round(2)
            frames.append(tbl)
        if educ_pct_cols:
            tbl = df.groupby("kelompok")[educ_pct_cols].mean().mul(100).T\
                    .reindex(columns=GROUP_LABELS).round(2)
            tbl.index = [f"{c}_pct_ART" for c in tbl.index]
            frames.append(tbl)
        if frames:
            pd.concat(frames).to_csv(self._p("tabel_pendidikan.csv"))

    def _export_perumahan(self, df: pd.DataFrame) -> None:
        oh_cols   = [c for c in self.cfg.feat_house_onehot if c in df.columns]
        cont_cols = [c for c in self.cfg.feat_house_cont   if c in df.columns]
        tables = []
        if oh_cols:
            tables.append(df.groupby("kelompok")[oh_cols].mean().mul(100).T
                          .reindex(columns=GROUP_LABELS))
        if cont_cols:
            tables.append(df.groupby("kelompok")[cont_cols].mean().T
                          .reindex(columns=GROUP_LABELS))
        if tables:
            pd.concat(tables).round(3).to_csv(self._p("tabel_perumahan.csv"))

    def _export_aset(self, df: pd.DataFrame) -> None:
        excluded_assets = {"h_asset_phone", "h_asset_boat", "h_asset_motorboat"}
        cols = [c for c in self.cfg.feat_asset
                if c in df.columns and c not in excluded_assets]
        if not cols:
            return
        tbl = df.groupby("kelompok")[cols].mean().T\
                .reindex(columns=GROUP_LABELS).mul(100).round(2)
        tbl["gap_bottom_top"] = (tbl["bottom40"] - tbl["top60"]).round(2)
        tbl["abs_gap"] = tbl["gap_bottom_top"].abs()
        tbl = tbl.sort_values("abs_gap", ascending=False).drop(columns="abs_gap")
        tbl.to_csv(self._p("tabel_aset.csv"), index_label="aset")

    # ── Ranking variabel pembeda (gap 40% Terbawah vs 60% Teratas) ──────────
    def _export_ranking_gap(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.cfg.gap_ranking_vars if c in df.columns]
        rows = []
        for c in cols:
            b40 = df.loc[df["kelompok"] == "bottom40", c].mean()
            top = df.loc[df["kelompok"] == "top60", c].mean()
            is_pct = c != "h_lnluaslantai"  # semua kecuali luas lantai dalam basis 0-1 -> %
            b40_disp = b40 * 100 if is_pct else b40
            top_disp = top * 100 if is_pct else top
            rows.append({
                "variabel": c,
                "label": _label(c),
                "bottom40": round(b40_disp, 2),
                "top60": round(top_disp, 2),
                "gap": round(b40_disp - top_disp, 2),
                "abs_gap": round(abs(b40_disp - top_disp), 2),
                "satuan": "%" if is_pct else "ln(m2)",
            })
        tbl = pd.DataFrame(rows).sort_values("abs_gap", ascending=False).reset_index(drop=True)
        tbl.to_csv(self._p("tabel_ranking_gap.csv"), index=False)
        return tbl

    # ── Tabel justifikasi variabel untuk PMT ────────────
    def _export_justifikasi_pmt(self, df: pd.DataFrame) -> pd.DataFrame:
        ranking = self._export_ranking_gap(df)
        top15 = ranking.head(15).copy()
        top15["interpretasi"] = top15.apply(
            lambda r: (f"RT 40% Terbawah memiliki {r['label'].lower()} rata-rata "
                       f"{r['bottom40']:.1f}{r['satuan']} dibanding {r['top60']:.1f}{r['satuan']} "
                       f"pada RT 60% Teratas (selisih {abs(r['gap']):.1f} poin)."),
            axis=1)
        top15.to_csv(self._p("tabel_justifikasi_pmt.csv"), index=False)

        with open(self._p("narasi_justifikasi_pmt.txt"), "w", encoding="utf-8") as f:
            f.write("Draft narasi Bab 4 -- Justifikasi Pemilihan Variabel untuk Model PMT\n")
            f.write("=" * 75 + "\n\n")
            f.write(
                "Variabel dengan kesenjangan (gap) terbesar antara kelompok 40%\n"
                "Terbawah dan 60% Teratas diasumsikan memiliki daya diskriminasi yang\n"
                "lebih tinggi dalam membedakan kondisi kesejahteraan rumah tangga,\n"
                "sehingga layak digunakan sebagai prediktor pada model Proxy Means\n"
                "Test (PMT).\n\n"
                "Lima variabel dengan kesenjangan terbesar:\n"
            )
            for _, r in top15.head(5).iterrows():
                f.write(f"  - {r['interpretasi']}\n")
        return top15

    # ── Ringkasan per provinsi per kelompok ──────────────
    def _export_provinsi_kelompok(self, df: pd.DataFrame) -> pd.DataFrame:
        if "kode_prov_new" not in df.columns:
            return pd.DataFrame()
        tbl = (df.groupby(["kode_prov_new", "kelompok"]).size()
               .rename("n").reset_index())
        tbl["provinsi"] = tbl["kode_prov_new"].map(PROV_NAMES).fillna(tbl["kode_prov_new"].astype(str))
        tbl["pct_dalam_provinsi"] = tbl.groupby("kode_prov_new")["n"].transform(lambda x: x / x.sum() * 100)
        tbl = tbl.sort_values(["kode_prov_new", "kelompok"])
        tbl.to_csv(self._p("summary_provinsi_kelompok.csv"), index=False)
        return tbl

    def _p(self, fn: str) -> str:
        return os.path.join(self.output_dir, fn)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers chart
# ─────────────────────────────────────────────────────────────────────────────

def _save(
    fig: plt.Figure,
    output_dir: str,
    filename: str,
    crop_to_content: bool = True,
) -> None:
    if crop_to_content:
        fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, filename), dpi=300,
        bbox_inches="tight" if crop_to_content else None,
    )
    plt.close(fig)
    print(f"  [OK] {filename}")


def _source(fig: plt.Figure, y: float = 0.01) -> None:
    pass  # catatan kaki dinonaktifkan


def _grouped_bar_n(
    ax: plt.Axes,
    categories: List[str],
    values_dict: Dict[str, List[float]],
    value_fmt: str = "{:.2f}",
    group_labels: Optional[List[str]] = None,
) -> None:
    """Grouped bar generik -- otomatis menyesuaikan lebar/posisi bar
    berdasarkan jumlah kelompok (dulu selalu 3, sekarang 2)."""
    group_labels = group_labels or GROUP_LABELS
    n      = len(group_labels)
    x      = np.arange(len(categories))
    # Setiap pasangan berdempetan; lebar dan jarak pusat sama di semua chart.
    width = 0.45
    pair_spacing = width
    offsets = [(i - (n - 1) / 2) * pair_spacing for i in range(n)]

    for i, grp in enumerate(group_labels):
        vals = values_dict[grp]
        bars = ax.bar(x + offsets[i], vals, width,
                      color=GROUP_COLORS[grp], edgecolor="#555555",
                      linewidth=0.5, label=GROUP_DISPLAY[grp])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.annotate(
                    value_fmt.format(val),
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="normal",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_axisbelow(True)


def _stacked_bar_pct(
    ax: plt.Axes,
    means: pd.DataFrame,       # index = GROUP_LABELS, columns = cols (0-100 scale)
    cols: List[str],
    col_labels: List[str],
    colors: List[str],
    min_label_pct: float = 3.0,
) -> None:
    """Stacked 100% bar chart util untuk variabel proporsi (status sekolah).
    means harus sudah dalam skala 0-100."""
    x_labels = [GROUP_DISPLAY[g] for g in GROUP_LABELS]
    # Dekatkan kedua batang agar perbandingan lebih mudah dibaca dan ruang
    # kosong di tengah grafik tidak terlalu lebar.
    x_positions = GROUP_X
    bottom = np.zeros(len(GROUP_LABELS))
    for col, lbl, clr in zip(cols, col_labels, colors):
        vals = means[col].values
        bars = ax.bar(x_positions, vals, bottom=bottom, label=lbl, color=clr,
                      edgecolor="#444444", linewidth=0.6, width=0.5)
        for bar, val, bot in zip(bars, vals, bottom):
            if val >= min_label_pct:
                ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f"{val:.1f}%",
                        ha="center", va="center", fontsize=8.5, fontweight="bold", color="black")
        bottom += vals
    ax.set_ylim(0, 100)
    _set_group_axis(ax)
    ax.set_ylabel("Proporsi ART (%)", fontweight="normal")
    ax.set_axisbelow(True)


# ─────────────────────────────────────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────────────────────────────────────

class Visualizer:
    def __init__(self, config: Config, output_dir: str) -> None:
        self.cfg = config
        self.od  = output_dir

    def render_all(self, df: pd.DataFrame) -> None:
        print("[4/6] Membuat chart visualisasi EDA...")
        self._c01_distribusi_pcexp(df)
        self._c02_komposisi_kelompok(df)
        self._c03a_jumlah_art(df)
        self._c03a2_komposisi_gender(df)
        self._c03b_struktur_umur(df)
        self._c03c_ukuran_rt_stacked(df)
        self._c04_status_sekolah_proporsi(df)
        self._c05_pendidikan_proporsi(df)
        self._c06_aset_dummy(df)
        self._c07_kualitas_material(df)
        self._c08_air_sanitasi(df)
        self._c09_energi(df)
        self._c10_luas_lantai(df)
        self._c11_ranking_gap(df)
        self._c12_provinsi_perbandingan(df)
        self._c13_boxplot_sebaran(df)
        self._c14_korelasi_heatmap(df)
        self._c15_proxy_spasial_kabupaten(df)
        self._c16_diy_vs_jateng_bottom40(df)

    # ── 01. Distribusi pcexp (histogram density) ─────────────────────────────
    def _c01_distribusi_pcexp(self, df: pd.DataFrame) -> None:
        col  = "pcexp_raw" if "pcexp_raw" in df.columns else "pcexp"
        xlim = df[col].quantile(0.98) / 1e6
        # Gunakan batas bin yang sama untuk kedua kelompok agar lebar batang
        # histogram dapat dibandingkan secara langsung.
        common_bins = np.linspace(0, xlim, 121)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        for g in GROUP_LABELS:
            sub    = df.loc[df["kelompok"] == g, col] / 1e6
            med    = sub.median()
            ax.hist(sub, bins=common_bins, alpha=0.50, density=True,
                    color=GROUP_COLORS[g], label=GROUP_DISPLAY[g])
            ax.axvline(med, color=GROUP_COLORS[g], linestyle="--", linewidth=1.8,
                       label=f"Median {GROUP_DISPLAY[g]}  (Rp {med*1e6:,.0f})")

        ax.set_xlim(0, xlim)
        ax.set_xlabel("Pengeluaran per Kapita Sebulan (Juta Rupiah)")
        ax.set_ylabel("Kepadatan (Density)")
        ax.set_title(
            "Distribusi Pengeluaran per Kapita Berdasarkan Kelompok\n"
            "(40% Terbawah / 60% Teratas)", fontweight="bold", pad=12)
        ax.legend(fontsize=8.5)
        _source(fig)
        _save(fig, self.od, "chart_01_distribusi_pcexp.svg")

    # ── 02. Komposisi kelompok per tahun ─────────────────────────────────────
    def _c02_komposisi_kelompok(self, df: pd.DataFrame) -> None:
        years = sorted(df["year"].unique())
        vals  = {g: [(df[df["year"]==y]["kelompok"]==g).mean()*100 for y in years]
                 for g in GROUP_LABELS}

        fig, ax = plt.subplots(figsize=(7, 5))
        _grouped_bar_n(ax, [str(y) for y in years], vals, value_fmt="{:.1f}%")
        ax.set_ylim(0, 75)
        ax.set_xlabel("Tahun Susenas", fontweight="bold")
        ax.set_ylabel("Proporsi (%)", fontweight="bold")
        ax.set_title(
            "Komposisi Kelompok Kesejahteraan per Tahun\n"
            "(40% Terbawah / 60% Teratas)", fontweight="bold", pad=12)
        ax.legend(handles=LEGEND_HANDLES)
        ax.text(0.5, -0.20,
                "Catatan: proporsi ~40/60 adalah konsekuensi langsung dari cara\n"
                "kelompok dibentuk (relatif per kabupaten per tahun), bukan temuan substantif.",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
                style="italic", color="#555555")
        _source(fig)
        _save(fig, self.od, "chart_02_komposisi_kelompok.svg")

    # ── 03a. Rata-rata jumlah ART (COUNT murni) ──────────────────────────────
    def _c03a_jumlah_art(self, df: pd.DataFrame) -> None:
        col = "h_hhcount"
        if col not in df.columns:
            print("  [SKIP] chart_03a — h_hhcount tidak ada")
            return
        means = df.groupby("kelompok")[col].mean().reindex(GROUP_LABELS)
        fig, ax = plt.subplots(figsize=(6, 5))
        bar_width = 0.235
        bar_group_center = 0.0
        bar_x = np.array([
            bar_group_center - bar_width / 2,
            bar_group_center + bar_width / 2,
        ])
        bars = ax.bar(bar_x, means.values,
                      color=[GROUP_COLORS[g] for g in GROUP_LABELS],
                      edgecolor="#555555", linewidth=0.5, width=bar_width)
        ax.set_xticks(bar_x)
        ax.set_xticklabels([GROUP_DISPLAY[g] for g in GROUP_LABELS])
        ax.set_xlim(-0.55, 0.55)
        for bar, val in zip(bars, means.values):
            ax.annotate(f"{val:.2f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, fontweight="normal")
        ax.set_ylabel("Rata-rata Jumlah ART (Orang)")
        ax.set_title("Rata-rata Jumlah Anggota Rumah Tangga\nBerdasarkan Kelompok Kesejahteraan",
                     fontweight="bold", pad=12)
        ax.set_axisbelow(True)
        _source(fig)
        _save(fig, self.od, "chart_03a_jumlah_art.svg", crop_to_content=False)

    # ── 03a2. [v4] Komposisi gender — kembali ke rata-rata JUMLAH ART ────────
    def _c03a2_komposisi_gender(self, df: pd.DataFrame) -> None:
        cols_map = {"h_nmale": "Laki-laki", "h_nfemale": "Perempuan"}
        cols = [c for c in cols_map if c in df.columns]
        if not cols:
            print("  [SKIP] chart_03a2 — kolom gender tidak ada")
            return
        means = df.groupby("kelompok")[cols].mean().reindex(GROUP_LABELS)
        vals  = {g: [means.loc[g, c] for c in cols] for g in GROUP_LABELS}
        fig, ax = plt.subplots(figsize=(6, 5))
        _grouped_bar_n(ax, [cols_map[c] for c in cols], vals, value_fmt="{:.2f}")
        ax.set_xlim(-0.55, 1.55)
        ymax = max(v for vlist in vals.values() for v in vlist)
        ax.set_ylim(0, ymax * 1.3)
        ax.set_ylabel("Rata-rata Jumlah ART (Orang)")
        ax.set_title("Rata-rata Jumlah ART Berdasarkan Jenis Kelamin\ndan Kelompok Kesejahteraan",
                     fontweight="bold", pad=12)
        ax.legend(handles=LEGEND_HANDLES, title="Kelompok Kesejahteraan")
        _source(fig)
        _save(fig, self.od, "chart_03a2_komposisi_gender.svg", crop_to_content=False)

    # ── 03b. [v4] Struktur umur — kembali ke rata-rata JUMLAH ART ────────────
    def _c03b_struktur_umur(self, df: pd.DataFrame) -> None:
        cols_map = {
            "h_nage04": "Balita\n(0–4 Thn)", "h_nage519": "Anak/Remaja\n(5–19 Thn)",
            "h_nage2064": "Produktif\n(20–64 Thn)", "h_nage65up": "Lansia\n(65+ Thn)",
        }
        cols = [c for c in cols_map if c in df.columns]
        if not cols:
            print("  [SKIP] chart_03b — kolom struktur umur tidak ada")
            return
        means = df.groupby("kelompok")[cols].mean().reindex(GROUP_LABELS)
        vals  = {g: [means.loc[g, c] for c in cols] for g in GROUP_LABELS}
        fig, ax = plt.subplots(figsize=(9, 5))
        _grouped_bar_n(ax, [cols_map[c] for c in cols], vals, value_fmt="{:.2f}")
        ymax = max(v for vlist in vals.values() for v in vlist)
        ax.set_ylim(0, ymax * 1.3)
        ax.set_ylabel("Rata-rata Jumlah ART (Orang)")
        ax.set_title("Rata-rata Jumlah ART Berdasarkan Kelompok Umur\ndan Kelompok Kesejahteraan",
                     fontweight="bold", pad=12)
        ax.legend(handles=LEGEND_HANDLES, title="Kelompok Kesejahteraan")
        _source(fig)
        _save(fig, self.od, "chart_03b_struktur_umur.svg")

    # ── 03c. Stacked bar ukuran RT ────────────────────────────────────────────
    def _c03c_ukuran_rt_stacked(self, df: pd.DataFrame) -> None:
        if "h_hhcount" not in df.columns:
            print("  [SKIP] chart_03c — h_hhcount tidak ada")
            return
        df2 = df.copy()
        df2["kat_art"] = np.where(df2["h_hhcount"] <= 4, "≤ 4 ART", "> 4 ART")
        ct = (pd.crosstab(df2["kat_art"], df2["kelompok"], normalize="index") * 100)\
            .reindex(index=["≤ 4 ART", "> 4 ART"], columns=GROUP_LABELS)
        x_labels = ["≤ 4 ART", "> 4 ART"]
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        bottom = np.zeros(len(x_labels))
        for grp in GROUP_LABELS:
            vals = ct[grp].values
            bars = ax.bar(GROUP_X, vals, bottom=bottom, label=GROUP_DISPLAY[grp],
                          color=GROUP_COLORS[grp], edgecolor="#444444", linewidth=0.6, width=0.45)
            for bar, val, bot in zip(bars, vals, bottom):
                if val >= 3.0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2, f"{val:.1f}%",
                            ha="center", va="center", fontsize=10, fontweight="bold", color="black")
            bottom += vals
        ax.set_ylim(0, 100)
        _set_group_axis(ax)
        ax.set_ylabel("Persentase Rumah Tangga (%)")
        ax.set_xlabel("Kategori Jumlah Anggota Rumah Tangga")
        ax.set_title("Proporsi Kelompok Kesejahteraan Berdasarkan\nUkuran Rumah Tangga (Jumlah ART)",
                     fontweight="bold", pad=12)
        ax.legend(handles=LEGEND_HANDLES,
                  loc="lower center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=True, fontsize=9)
        ax.set_axisbelow(True)
        _save(fig, self.od, "chart_03c_ukuran_rt_stacked.svg")

    # ── 04. Status partisipasi sekolah (% dari ART yang punya status sekolah, stacked) ──
    def _c04_status_sekolah_proporsi(self, df: pd.DataFrame) -> None:
        cols_map = {"h_neverschool_pct": "Tidak/Belum\nPernah Sekolah",
                    "h_stillschool_pct": "Masih\nSekolah", "h_notschool_pct": "Tidak\nSekolah Lagi"}
        cols = [c for c in cols_map if c in df.columns]
        if not cols:
            print("  [SKIP] chart_04 — kolom status sekolah tidak ada")
            return
        means = df.groupby("kelompok")[cols].mean().reindex(GROUP_LABELS)  # sudah 0-100
        fig, ax = plt.subplots(figsize=(6, 5.5))
        _stacked_bar_pct(ax, means, cols, [cols_map[c].replace("\n", " ") for c in cols], C_SCHOOL[:len(cols)])
        ax.set_title("Status Partisipasi Sekolah ART (%)\nBerdasarkan Kelompok Kesejahteraan",
                     fontweight="bold", pad=12)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=1, frameon=True, fontsize=8.5)
        _source(fig)
        _save(fig, self.od, "chart_04_status_sekolah_proporsi.svg")

    # ── 05. Pendidikan PROPORSI ───────────────────────────────────────────────
    def _c05_pendidikan_proporsi(self, df: pd.DataFrame) -> None:
        cols_map = {"h_notgrad": "Tdk Tamat SD", "h_ngrad_sd": "Tamat SD",
                    "h_ngrad_smp": "Tamat SMP", "h_ngrad_sma": "Tamat SMA",
                    "h_ngrad_d": "Diploma", "h_ngrad_s": "Sarjana+"}
        cols = [c for c in cols_map if c in df.columns]
        if not cols:
            print("  [SKIP] chart_05 — kolom pendidikan tidak ada")
            return
        means = df.groupby("kelompok")[cols].mean().mul(100).reindex(GROUP_LABELS)
        vals  = {g: [means.loc[g, c] for c in cols] for g in GROUP_LABELS}
        fig, ax = plt.subplots(figsize=(10, 5.5))
        _grouped_bar_n(ax, [cols_map[c] for c in cols], vals, value_fmt="{:.1f}%")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Rata-rata Proporsi ART (%)")
        ax.set_title("Proporsi ART Berdasarkan Pendidikan Tertinggi yang Ditamatkan\nMenurut Kelompok Kesejahteraan",
                     fontweight="normal", pad=12)
        ax.legend(handles=LEGEND_HANDLES, title="Kelompok Kesejahteraan")
        _source(fig)
        _save(fig, self.od, "chart_05_pendidikan_proporsi.svg")

    # ── 06. Kepemilikan aset, diurutkan berdasarkan besar gap ────────────────
    def _c06_aset_dummy(self, df: pd.DataFrame) -> None:
        excluded_assets = {"h_asset_phone", "h_asset_boat", "h_asset_motorboat"}
        cols = [c for c in self.cfg.feat_asset
                if c in df.columns and c not in excluded_assets]
        if not cols:
            print("  [SKIP] chart_06 — kolom aset tidak ada")
            return
        means = df.groupby("kelompok")[cols].mean().mul(100).reindex(GROUP_LABELS)
        gaps = (means.loc["bottom40"] - means.loc["top60"]).abs()
        cols = gaps.sort_values(ascending=False).index.tolist()
        means = means[cols]
        labels = [_label(c) for c in cols]
        vals = {g: means.loc[g].tolist() for g in GROUP_LABELS}

        fig, ax = plt.subplots(figsize=(max(12, len(cols) * 1.15), 6))
        _grouped_bar_n(ax, labels, vals, value_fmt="{:.1f}%")
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 112)
        ax.set_ylabel("Persentase Rumah Tangga yang Memiliki (%)", fontweight="normal")
        ax.set_title("Kepemilikan Aset Rumah Tangga Berdasarkan Kelompok Kesejahteraan\n"
                     "Diurutkan menurut Gap Terbesar", fontweight="normal", pad=12)
        ax.legend(handles=LEGEND_HANDLES, title="Kelompok Kesejahteraan", loc="upper left")
        _source(fig)
        _save(fig, self.od, "chart_06_aset_gap.svg")

    # ── 07. Kualitas material ─────────────────────────────────────────────────
    def _c07_kualitas_material(self, df: pd.DataFrame) -> None:
        panels = [
            {"title": "Status Penguasaan Bangunan Tempat Tinggal", "cols": ["h_house1","h_house2","h_house3","h_house4"],
             "labels": ["Milik Sendiri","Bebas Sewa/Lainnya","Kontrak/Sewa","Dinas"],
             "colors": ["#A8D5BA","#B5D4E8","#F9E07F","#F4A96A"]},
            {"title": "Kualitas Material Lantai", "cols": ["h_floor1","h_floor2","h_floor3","h_floor4"],
             "labels": ["Marmer/Keramik/Vinil","Ubin/Tegel","Kayu/Semen","Lainnya"], "colors": ["#A8D5BA","#F9E07F","#F4A96A","#E8877A"]},
            {"title": "Kualitas Material Dinding", "cols": ["h_wall1","h_wall2","h_wall3","h_wall4"],
             "labels": ["Tembok","Plesteran/Kayu","Anyaman Bambu","Lainnya"], "colors": ["#A8D5BA","#F9E07F","#F4A96A","#E8877A"]},
            {"title": "Kualitas Material Atap", "cols": ["h_roof1","h_roof2","h_roof3","h_roof4"],
             "labels": ["Beton","Genteng","Seng/Asbes/Kayu","Lainnya"], "colors": ["#A8D5BA","#F9E07F","#F4A96A","#E8877A"]},
        ]
        x_labels = [GROUP_DISPLAY[g] for g in GROUP_LABELS]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 6), squeeze=False)
        axes = axes[0]
        for ax, p in zip(axes, panels):
            exist_cols   = [c for c in p["cols"]   if c in df.columns]
            exist_labels = [p["labels"][p["cols"].index(c)] for c in exist_cols]
            exist_colors = [p["colors"][p["cols"].index(c)] for c in exist_cols]
            if not exist_cols:
                ax.text(0.5, 0.5, "Data tidak tersedia", ha="center", transform=ax.transAxes)
                continue
            means  = df.groupby("kelompok")[exist_cols].mean().mul(100).reindex(GROUP_LABELS)
            bottom = np.zeros(len(GROUP_LABELS))
            for col, lbl, clr in zip(exist_cols, exist_labels, exist_colors):
                vals = means[col].values
                bars = ax.bar(GROUP_X, vals, bottom=bottom, label=lbl, color=clr,
                              edgecolor="#555555", linewidth=0.5, width=0.4)
                for bar, val, bot in zip(bars, vals, bottom):
                    if val > 5:
                        ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f"{val:.1f}%",
                                ha="center", va="center", fontsize=8, fontweight="bold", color="black")
                bottom += vals
            ax.set_ylim(0, 100)
            _set_group_axis(ax)
            ax.set_title(p["title"], fontsize=10.5, fontweight="bold", pad=8)
            ax.set_ylabel("Persentase Rumah Tangga (%)", fontweight="normal")
            ax.set_axisbelow(True)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8, ncol=2, frameon=True)
        fig.suptitle("Proporsi Kualitas Fisik Tempat Tinggal Berdasarkan Kelompok Kesejahteraan",
                     fontsize=12, fontweight="bold", y=1.02)
        _save(fig, self.od, "chart_07_kualitas_material.svg")

    # ── 08. Air & Sanitasi ─────────────────────────────────────────────────────
    def _c08_air_sanitasi(self, df: pd.DataFrame) -> None:
        panels = [
            {"title": "Sumber Air Minum", "cols": ["h_dwater1","h_dwater2","h_dwater3","h_dwater4","h_dwater5"],
             "labels": ["Kemasan Bermerek","Isi Ulang/Leding","Sumur Bor/Mata Air Terlindung","Mata Air Tak Terlindung","Lainnya"],
             "colors": ["#A8D5BA","#B5D4E8","#F9E07F","#F4A96A","#E8877A"]},
            {"title": "Jenis Toilet", "cols": ["h_toiltype1","h_toiltype2","h_toiltype3","h_toiltype4","h_toiltype5","h_toiltype6"],
             "labels": ["Ada & Leher Angsa","Ada & Tutup","MCK Umum & Leher Angsa","Ada & Tutup","MCK Komunal & Tutup","MCK Umum & Tutup"],
             "colors": ["#A8D5BA","#B5D4E8","#F9E07F","#F4A96A","#E8877A","#D4A5C9"]},
        ]
        x_labels = [GROUP_DISPLAY[g] for g in GROUP_LABELS]
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        for ax, p in zip(axes, panels):
            exist_cols   = [c for c in p["cols"]   if c in df.columns]
            exist_labels = [p["labels"][p["cols"].index(c)] for c in exist_cols]
            exist_colors = [p["colors"][p["cols"].index(c)] for c in exist_cols]
            if not exist_cols:
                continue
            means  = df.groupby("kelompok")[exist_cols].mean().mul(100).reindex(GROUP_LABELS)
            bottom = np.zeros(len(GROUP_LABELS))
            for col, lbl, clr in zip(exist_cols, exist_labels, exist_colors):
                vals = means[col].values
                bars = ax.bar(GROUP_X, vals, bottom=bottom, label=lbl, color=clr,
                              edgecolor="#555555", linewidth=0.5, width=0.4)
                for bar, val, bot in zip(bars, vals, bottom):
                    if val > 5:
                        ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f"{val:.1f}%",
                                ha="center", va="center", fontsize=7.5, fontweight="bold", color="black")
                bottom += vals
            ax.set_ylim(0, 100)
            _set_group_axis(ax)
            ax.set_title(p["title"], fontsize=11, fontweight="bold", pad=8)
            ax.set_ylabel("Persentase Rumah Tangga (%)", fontweight="normal")
            ax.set_axisbelow(True)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8, ncol=2, frameon=True)
        fig.suptitle("Akses Air Minum dan Sanitasi Berdasarkan Kelompok Kesejahteraan",
                     fontsize=12, fontweight="bold", y=1.02)
        _save(fig, self.od, "chart_08_air_sanitasi.svg")

    # ── 09. Energi ─────────────────────────────────────────────────────────────
    def _c09_energi(self, df: pd.DataFrame) -> None:
        panels = [
            {"title": "Sumber Penerangan", "cols": ["h_lighting1","h_lighting2","h_lighting3"],
             "labels": ["Listrik PLN","Listrik Non-PLN","Bukan Listrik"], "colors": ["#A8D5BA","#F9E07F","#E8877A"]},
             {"title": "Daya Listrik", "cols": ["h_epower1","h_epower2","h_epower3"],
             "labels": ["450 VA","900 VA","> 1300 VA"], "colors": ["#A8D5BA","#B5D4E8","#F9E07F","#E8877A"]},
            {"title": "Bahan Bakar Memasak", "cols": ["h_cookingfuel1","h_cookingfuel2","h_cookingfuel3","h_cookingfuel4","h_cookingfuel5"],
             "labels": ["Listrik","LPG 12 kg","LPG < 12 kg","Lainnya","Tidak Memasak"],
             "colors": ["#A8D5BA","#B5D4E8","#F9E07F","#F4A96A","#E8877A"]},
        ]
        x_labels = [GROUP_DISPLAY[g] for g in GROUP_LABELS]
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        for ax, p in zip(axes, panels):
            exist_cols   = [c for c in p["cols"]   if c in df.columns]
            exist_labels = [p["labels"][p["cols"].index(c)] for c in exist_cols]
            exist_colors = [p["colors"][p["cols"].index(c)] for c in exist_cols]
            if not exist_cols:
                continue
            means  = df.groupby("kelompok")[exist_cols].mean().mul(100).reindex(GROUP_LABELS)
            bottom = np.zeros(len(GROUP_LABELS))
            for col, lbl, clr in zip(exist_cols, exist_labels, exist_colors):
                vals = means[col].values
                bars = ax.bar(GROUP_X, vals, bottom=bottom, label=lbl, color=clr,
                              edgecolor="#555555", linewidth=0.5, width=0.4)
                for bar, val, bot in zip(bars, vals, bottom):
                    if val > 5:
                        ax.text(bar.get_x() + bar.get_width()/2, bot + val/2, f"{val:.1f}%",
                                ha="center", va="center", fontsize=8, fontweight="bold", color="black")
                bottom += vals
            ax.set_ylim(0, 100)
            _set_group_axis(ax)
            ax.set_title(p["title"], fontsize=11, fontweight="bold", pad=8)
            ax.set_ylabel("Persentase Rumah Tangga (%)", fontweight="normal")
            ax.set_axisbelow(True)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8.5, ncol=1, frameon=True)
        fig.suptitle("Akses Energi Berdasarkan Kelompok Kesejahteraan", fontsize=12, fontweight="bold", y=1.02)
        _save(fig, self.od, "chart_09_energi.svg")

    # ── 10. Luas lantai ────────────────────────────────────────────────────────
    def _c10_luas_lantai(self, df: pd.DataFrame) -> None:
        col = "h_lnluaslantai"
        if col not in df.columns:
            print("  [SKIP] chart_10 — h_lnluaslantai tidak ada")
            return
        means = df.groupby("kelompok")[col].mean().reindex(GROUP_LABELS)
        means_m2 = np.exp(means)
        fig, axes = plt.subplots(1, 2, figsize=(9, 5))
        ax = axes[0]
        colors = [GROUP_COLORS[g] for g in GROUP_LABELS]
        bars = ax.bar(GROUP_X, means.values,
                      color=colors, edgecolor="#555555", linewidth=0.5, width=0.4)
        _set_group_axis(ax)
        for bar, val in zip(bars, means.values):
            ax.annotate(f"{val:.3f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                        xytext=(0,4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")
        ax.set_ylabel("Rata-rata ln(Luas Lantai)", fontweight="bold")
        ax.set_title("Rata-rata Log Luas Lantai\n(h_lnluaslantai)", fontweight="bold")
        ax.set_axisbelow(True)
        ax2 = axes[1]
        bars2 = ax2.bar(GROUP_X, means_m2.values,
                        color=colors, edgecolor="#555555", linewidth=0.5, width=0.4)
        _set_group_axis(ax2)
        for bar, val in zip(bars2, means_m2.values):
            ax2.annotate(f"{val:.1f} m²", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                         xytext=(0,4), textcoords="offset points", ha="center", va="bottom",
                         fontsize=9, fontweight="bold")
        ax2.set_ylabel("Luas Lantai (m²)", fontweight="normal")
        ax2.set_title("Luas Lantai (exp transform)\nMenurut Kelompok", fontweight="normal")
        ax2.set_axisbelow(True)
        fig.suptitle("Luas Lantai Rumah Berdasarkan Kelompok Kesejahteraan", fontsize=12, fontweight="bold", y=1.02)
        _source(fig)
        _save(fig, self.od, "chart_10_luas_lantai.svg")

    # ── 11. Ranking Top-15 variabel dengan gap terbesar ────────────
    def _c11_ranking_gap(self, df: pd.DataFrame) -> None:
        cols = [c for c in self.cfg.gap_ranking_vars if c in df.columns]
        if not cols:
            print("  [SKIP] chart_11 — tidak ada variabel gap_ranking_vars")
            return
        rows = []
        for c in cols:
            b40 = df.loc[df["kelompok"] == "bottom40", c].mean()
            top = df.loc[df["kelompok"] == "top60", c].mean()
            is_pct = c != "h_lnluaslantai"
            gap = (b40 - top) * 100 if is_pct else (b40 - top)
            rows.append((c, gap))
        rows.sort(key=lambda x: abs(x[1]), reverse=True)
        top15 = rows[:15]
        labels = [_label(c) for c, _ in top15]
        gaps   = [g for _, g in top15]
        colors = [C_BOT if g < 0 else C_TOP for g in gaps]

        fig, ax = plt.subplots(figsize=(9, 7))
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, gaps, color=colors, edgecolor="#444444", linewidth=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_xlabel("Gap (40% Terbawah − 60% Teratas)\nNegatif = 60% Teratas lebih tinggi | satuan: poin persen (kecuali ln luas lantai)",
                     fontweight="bold", fontsize=9)
        ax.set_title("Top 15 Variabel dengan Kesenjangan Terbesar\nantara 40% Terbawah dan 60% Teratas",
                     fontweight="bold", pad=12)
        for y, g in zip(y_pos, gaps):
            ax.text(g + (0.5 if g >= 0 else -0.5), y, f"{g:+.1f}", va="center",
                    ha="left" if g >= 0 else "right", fontsize=8, fontweight="bold")
        ax.legend(handles=[mpatches.Patch(color=C_TOP, label="60% Teratas lebih tinggi"),
                           mpatches.Patch(color=C_BOT, label="40% Terbawah lebih tinggi")],
                  loc="lower right", fontsize=8.5)
        _save(fig, self.od, "chart_11_ranking_gap.svg")

    # ── 12. Komposisi kelompok per provinsi (DIY vs Jateng) ────
    def _c12_provinsi_perbandingan(self, df: pd.DataFrame) -> None:
        if "kode_prov_new" not in df.columns:
            print("  [SKIP] chart_12 — kode_prov_new tidak ada")
            return
        provs = sorted(df["kode_prov_new"].unique())
        provs = [p for p in provs if p in PROV_NAMES]
        if len(provs) < 1:
            print("  [SKIP] chart_12 — tidak ada provinsi dikenal (33/34)")
            return

        fig, axes = plt.subplots(1, len(provs), figsize=(4.5*len(provs), 5), squeeze=False)
        axes = axes[0]
        for ax, p in zip(axes, provs):
            sub = df[df["kode_prov_new"] == p]
            dist = sub["kelompok"].value_counts(normalize=True).reindex(GROUP_LABELS) * 100
            bars = ax.bar(GROUP_X, dist.values,
                          color=[GROUP_COLORS[g] for g in GROUP_LABELS],
                          edgecolor="#555555", linewidth=0.5, width=0.45)
            _set_group_axis(ax)
            for bar, val in zip(bars, dist.values):
                ax.annotate(f"{val:.1f}%", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                           xytext=(0,4), textcoords="offset points", ha="center", va="bottom",
                           fontsize=9.5, fontweight="bold")
            ax.set_ylim(0, 75)
            ax.set_title(f"{PROV_NAMES[p]} (n={len(sub):,})", fontweight="bold", fontsize=10.5)
            ax.set_ylabel("Proporsi RT (%)", fontweight="bold")
            ax.set_axisbelow(True)
        fig.suptitle("Komposisi Kelompok Kesejahteraan per Provinsi", fontweight="bold", fontsize=12, y=1.03)
        _save(fig, self.od, "chart_12_provinsi_perbandingan.svg")

    # ── 13. Boxplot sebaran pcexp & luas lantai per kelompok ───────
    def _c13_boxplot_sebaran(self, df: pd.DataFrame) -> None:
        col_pcexp = "pcexp_raw" if "pcexp_raw" in df.columns else "pcexp"
        has_pcexp = col_pcexp in df.columns
        has_luas  = "h_lnluaslantai" in df.columns
        if not (has_pcexp or has_luas):
            print("  [SKIP] chart_13 — tidak ada pcexp maupun luas lantai")
            return

        n_panel = int(has_pcexp) + int(has_luas)
        fig, axes = plt.subplots(1, n_panel, figsize=(5*n_panel, 5.5))
        if n_panel == 1:
            axes = [axes]
        idx = 0

        if has_pcexp:
            ax = axes[idx]; idx += 1
            data = [df.loc[df["kelompok"] == g, col_pcexp] / 1e6 for g in GROUP_LABELS]
            bp = ax.boxplot(data, positions=GROUP_X,
                            patch_artist=True, showfliers=False, widths=0.4,
                            medianprops=dict(color="black", linewidth=1.6))
            _set_group_axis(ax)
            for patch, g in zip(bp["boxes"], GROUP_LABELS):
                patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.8)
            ax.set_ylabel("Pengeluaran per Kapita (Juta Rupiah)", fontweight="bold")
            ax.set_title("Sebaran Pengeluaran per Kapita\n(outlier ekstrem disembunyikan)", fontweight="bold")
            ax.set_axisbelow(True)

        if has_luas:
            ax = axes[idx]; idx += 1
            m2 = np.exp(df["h_lnluaslantai"])
            data = [m2[df["kelompok"] == g] for g in GROUP_LABELS]
            bp = ax.boxplot(data, positions=GROUP_X,
                            patch_artist=True, showfliers=False, widths=0.4,
                            medianprops=dict(color="black", linewidth=1.6))
            _set_group_axis(ax)
            for patch, g in zip(bp["boxes"], GROUP_LABELS):
                patch.set_facecolor(GROUP_COLORS[g]); patch.set_alpha(0.8)
            ax.set_ylabel("Luas Lantai (m²)", fontweight="normal")
            ax.set_title("Sebaran Luas Lantai Rumah\n(outlier ekstrem disembunyikan)", fontweight="normal")
            ax.set_axisbelow(True)

        fig.suptitle("Sebaran (Bukan Hanya Rata-rata) Berdasarkan Kelompok Kesejahteraan",
                     fontweight="bold", fontsize=12, y=1.02)
        _save(fig, self.od, "chart_13_boxplot_sebaran.svg")

    # ── 14. Heatmap korelasi antar karakteristik kunci ─────────────
    def _c14_korelasi_heatmap(self, df: pd.DataFrame) -> None:
        col_pcexp = "pcexp_raw" if "pcexp_raw" in df.columns else "pcexp"
        candidate_cols = {
            col_pcexp: "Pengeluaran/Kapita",
            "h_lnluaslantai": "ln(Luas Lantai)",
            "h_asset_smartphone": "Smartphone",
            "h_asset_car": "Mobil",
            "h_asset_motorcycle": "Motor",
            "h_asset_ac": "AC",
            "h_asset_computer": "Komputer/Laptop",
            "h_ngrad_sma": "Tamat SMA",
            "h_ngrad_s": "Tamat Sarjana+",
            "h_hhcount": "Jumlah ART",
        }
        cols = {c: lbl for c, lbl in candidate_cols.items() if c in df.columns}
        if len(cols) < 3:
            print("  [SKIP] chart_14 — variabel untuk korelasi tidak cukup")
            return

        corr_df = df[list(cols.keys())].rename(columns=cols)
        corr = corr_df.corr(method="spearman")

        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns, fontsize=9)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                val = corr.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if abs(val) > 0.55 else "black")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Korelasi Spearman", fontweight="bold")
        ax.set_title("Korelasi antar Karakteristik Kunci Rumah Tangga\n(Seluruh RT, bukan per kelompok)",
                     fontweight="bold", pad=12)
        _save(fig, self.od, "chart_14_korelasi_heatmap.svg")

    # ── 15. Proxy spasial: % 40% Terbawah per kabupaten (bar, bukan peta) ─
    def _c15_proxy_spasial_kabupaten(self, df: pd.DataFrame) -> None:
        if "kode_kab_new" not in df.columns:
            print("  [SKIP] chart_15 — kode_kab_new tidak ada")
            return
        pct_b40 = (df.groupby("kode_kab_new")["kelompok"]
                   .apply(lambda s: (s == "bottom40").mean() * 100)
                   .sort_values(ascending=False))
        med_pcexp = None
        col_pcexp = "pcexp_raw" if "pcexp_raw" in df.columns else "pcexp"
        if col_pcexp in df.columns:
            med_pcexp = df.groupby("kode_kab_new")[col_pcexp].median().reindex(pct_b40.index) / 1e6

        fig, axes = plt.subplots(1, 2 if med_pcexp is not None else 1,
                                 figsize=(16 if med_pcexp is not None else 9, 6))
        if med_pcexp is None:
            axes = [axes]

        ax = axes[0]
        colors = plt.cm.RdYlGn_r((pct_b40.values - pct_b40.min()) / (pct_b40.max() - pct_b40.min() + 1e-9))
        ax.bar(range(len(pct_b40)), pct_b40.values, color=colors, width=0.8)
        ax.axhline(40, color="black", linestyle="--", linewidth=1, label="Referensi 40%")
        ax.set_xticks(range(len(pct_b40)))
        ax.set_xticklabels(pct_b40.index.astype(str), rotation=90, fontsize=6)
        ax.set_ylabel("% RT 40% Terbawah (dalam sampel kabupaten)", fontweight="bold")
        ax.set_title("Proxy Spasial: Sebaran %40% Terbawah per Kabupaten\n(Catatan: relatif per kabupaten, lihat metodologi)",
                     fontweight="bold", fontsize=10.5)
        ax.legend(fontsize=8)
        ax.set_axisbelow(True)

        if med_pcexp is not None:
            ax2 = axes[1]
            colors2 = plt.cm.RdYlGn(
                (med_pcexp.values - np.nanmin(med_pcexp.values)) /
                (np.nanmax(med_pcexp.values) - np.nanmin(med_pcexp.values) + 1e-9))
            ax2.bar(range(len(med_pcexp)), med_pcexp.values, color=colors2, width=0.8)
            ax2.set_xticks(range(len(med_pcexp)))
            ax2.set_xticklabels(med_pcexp.index.astype(str), rotation=90, fontsize=6)
            ax2.set_ylabel("Median Pengeluaran per Kapita (Juta Rp)", fontweight="bold")
            ax2.set_title("Proxy Spasial: Median Pengeluaran per Kabupaten\n(variabel absolut, lebih informatif secara spasial)",
                          fontweight="bold", fontsize=10.5)
            ax2.set_axisbelow(True)

        fig.suptitle("Variasi Antar-Kabupaten (33 & 34)\nUntuk peta koroplet sesungguhnya, perlu file shapefile batas kabupaten",
                     fontweight="bold", fontsize=11, y=1.03)
        _save(fig, self.od, "chart_15_proxy_spasial_kabupaten.svg")

    # ── 16. Deep-dive 40% Terbawah: DIY vs Jateng ───────────────────────
    def _c16_diy_vs_jateng_bottom40(self, df: pd.DataFrame) -> None:
        if "kode_prov_new" not in df.columns or "h_lnluaslantai" not in df.columns:
            print("  [SKIP] chart_16 — kolom provinsi/luas lantai tidak ada")
            return
        provs = [p for p in [33, 34] if p in df["kode_prov_new"].unique()]
        if len(provs) < 2:
            print("  [SKIP] chart_16 — data tidak mencakup kedua provinsi (33 & 34)")
            return

        sub = df[(df["kode_prov_new"].isin(provs)) & (df["kelompok"] == "bottom40")].copy()
        metrics = {
            "h_lnluaslantai": ("Rata-rata Luas Lantai (m²)", lambda s: np.exp(s.mean())),
            "h_asset_smartphone": ("% Punya Smartphone", lambda s: s.mean()*100),
            "h_asset_motorcycle": ("% Punya Motor", lambda s: s.mean()*100),
            "h_ngrad_sma": ("% Tamat SMA", lambda s: s.mean()*100),
        }
        metrics = {c: v for c, v in metrics.items() if c in sub.columns}
        if not metrics:
            print("  [SKIP] chart_16 — tidak ada metrik yang tersedia")
            return

        labels, vals_by_prov = list(metrics.keys()), {p: [] for p in provs}
        for c in labels:
            _, fn = metrics[c]
            for p in provs:
                vals_by_prov[p].append(fn(sub.loc[sub["kode_prov_new"] == p, c]))

        display_labels = [metrics[c][0] for c in labels]
        x = np.arange(len(labels)); width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for i, p in enumerate(provs):
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals_by_prov[p], width, label=PROV_NAMES.get(p, str(p)),
                          color=PROV_COLORS.get(p, "#888888"), edgecolor="#444444", linewidth=0.6)
            for bar, val in zip(bars, vals_by_prov[p]):
                ax.annotate(f"{val:.1f}", xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                           xytext=(0,3), textcoords="offset points", ha="center", va="bottom",
                           fontsize=8, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(display_labels, fontsize=9)
        ax.set_title("Perbandingan Karakteristik RT 40% Terbawah:\nDI Yogyakarta vs Jawa Tengah",
                     fontweight="bold", pad=12)
        ax.legend()
        ax.set_axisbelow(True)
        _save(fig, self.od, "chart_16_diy_vs_jateng_bottom40.svg")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class EDAKemiskinanPipeline:
    def __init__(self, config: Config, output_dir: str) -> None:
        self.cfg        = config
        self.output_dir = output_dir
        self.loader     = DataLoader(config)
        self.labeler    = PovertyLabeler(config)
        self.analyzer   = SummaryAnalyzer(config, output_dir)
        self.visualizer = Visualizer(config, output_dir)

    def run(self, data_map: Dict[int, str], gk_path: str,
            prov_filter: Optional[List[int]] = None) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        print("[1/6] Memuat Garis Kemiskinan...")
        gk_data = self.loader.load_gk(gk_path)

        print("[2/6] Memuat data Susenas...")
        pool: List[pd.DataFrame] = []
        for year, path in sorted(data_map.items()):
            if not os.path.exists(path):
                print(f"  [WARN] File tidak ditemukan: {path}")
                continue
            df = self.loader.load_year(path, year)
            df = self.labeler.filter_raw_provinces(df, prov_filter or [])
            df = self.labeler.attach_label(df, gk_data, year)
            for col in self.cfg.predictors:
                if col not in df.columns:
                    df[col] = 0
            pool.append(df)

        if not pool:
            print("[ERROR] Tidak ada data valid.")
            return

        full = pd.concat(pool, ignore_index=True)
        del pool; gc.collect()

        full = self.labeler.apply_province_filter(full, prov_filter or [])
        print(f"\nTotal data teranalisis: {len(full):,}\n")

        print("[3/6] Membangun summary & tabel CSV...")
        summary = self.analyzer.build_national_summary(full)
        self.analyzer.build_geo_summaries(full)
        self.analyzer.export_all_tables(full)

        self.visualizer.render_all(full)

        print(f"\n[SELESAI] Output tersimpan di: {self.output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EDA Susenas — 2 Kelompok Kesejahteraan (40% Terbawah / 60% Teratas, ala World Bank) v4")
    p.add_argument("--data",     nargs="+", required=True, metavar="YEAR:FILE")
    p.add_argument("--gk",       required=True)
    p.add_argument("--provinsi", nargs="*", type=int, default=[])
    p.add_argument("--output",   default="./output_eda")
    return p


def main() -> None:
    args     = _build_parser().parse_args()
    config   = Config()
    pipeline = EDAKemiskinanPipeline(config=config, output_dir=args.output)
    pipeline.run(
        data_map    = DataLoader.parse_data_args(args.data),
        gk_path     = args.gk,
        prov_filter = args.provinsi,
    )


if __name__ == "__main__":
    main()
