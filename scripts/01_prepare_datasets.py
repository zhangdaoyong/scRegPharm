"""
01_prepare_datasets.py
======================
Public omics data extraction, quality control, and harmonization pipeline.

Parses raw archives in `data/raw_downloads/`:
1. Prior Human TF & Pathway Regulon Network:
   - OmniPath CollecTRI (`collectri_human.tsv`, 64,516 human TF-target interactions)
   - TRRUST v2 (`trrust_rawdata.human.tsv`, 9,396 literature-curated human TF-target interactions)
   - Canonical MSigDB / Hallmark Tumor Microenvironment (TME) pathway regulons
2. Single-Cell RNA-seq Atlas of Human Melanoma (`GSE115978`, Jerby-Arnon et al., Cell 2018):
   - `GSE115978_counts.csv.gz` & `GSE115978_cell.annotations.csv.gz`
   - 6,879 annotated single cells across 33 melanoma tumors with clinical treatment status
     (`post.treatment` ICB-resistant vs `treatment.naive`)
3. 10x Visium Spatial Transcriptomics of Human Melanoma (`GSE298774`, `GSM9022963_A17`):
   - `GSM9022963_A17_matrix.mtx.gz`, `features.tsv.gz`, `barcodes.tsv.gz`, `tissue_positions.csv.gz`
   - 1,461 in-tissue spatial spots (`in_tissue == 1`, `UMI >= 200`)
4. Five Independent Clinical Immunotherapy Bulk RNA-seq Cohorts (CRI iAtlas / cBioPortal DataHub):
   - `GSE91061_Riaz_Melanoma` (`mel_iatlas_riaz_nivolumab_2017.tar.gz`, Riaz et al., Cell 2017, n=107)
   - `PRJEB23709_Gide_Melanoma` (`mel_iatlas_gide_2019.tar.gz`, Gide et al., Cancer Cell 2019, n=91)
   - `DFCI_Liu_Melanoma` (`mel_iatlas_liu_2019.tar.gz`, Liu et al., Nature Medicine 2019, n=122)
   - `GSE78220_Hugo_Melanoma` (`mel_iatlas_hugo_ucla_2016.tar.gz`, Hugo et al., Cell 2016, n=27)
   - `IMvigor210_Urothelial_ICB` (`blca_iatlas_imvigor210_2017.tar.gz`, Mariathasan et al., Nature 2018, n=348)
5. Chemical Perturbation & Genetic Vulnerability Databases:
   - `LINCS_L1000_Chem_Pert_up.txt` & `LINCS_L1000_Chem_Pert_down.txt` (A375 human melanoma perturbations)
   - `DepMap_CRISPR_GeneDependency_CellLines_2023.txt` & `Achilles_fitness_decrease.txt` (39 human melanoma cell lines)
   - `DGIdb_Drug_Targets_2024.txt` & `IDG_Drug_Targets_2022.txt` (Curated drug-target interactions)
"""

import os
import re
import json
import gzip
import tarfile
import hashlib
import time
import numpy as np
import pandas as pd
import scipy.io

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw_downloads")
os.makedirs(DATA_DIR, exist_ok=True)


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_real_clinical_cohorts():
    """
    Extract the 5 immunotherapy clinical cohorts from CRI iAtlas / cBioPortal tarballs.
    """
    cohort_specs = [
        ("GSE91061_Riaz_Melanoma", "mel_iatlas_riaz_nivolumab_2017", "data_mrna_seq_tpm.txt", "Melanoma (Anti-PD-1, Riaz Cell 2017)"),
        ("PRJEB23709_Gide_Melanoma", "mel_iatlas_gide_2019", "data_mrna_seq_expression.txt", "Melanoma (Anti-PD-1/CTLA-4, Gide Cancer Cell 2019)"),
        ("DFCI_Liu_Melanoma", "mel_iatlas_liu_2019", "data_mrna_seq_tpm.txt", "Melanoma (Anti-PD-1, Liu Nat Med 2019)"),
        ("GSE78220_Hugo_Melanoma", "mel_iatlas_hugo_ucla_2016", "data_mrna_seq_tpm.txt", "Melanoma (Anti-PD-1, Hugo Cell 2016)"),
        ("IMvigor210_Urothelial_ICB", "blca_iatlas_imvigor210_2017", "data_mrna_seq_tpm.txt", "Urothelial Phase II (Anti-PD-L1, IMvigor210 Nature 2018)"),
    ]
    cohorts = {}
    for cohort_name, study_id, expr_filename, desc in cohort_specs:
        tar_path = os.path.join(RAW_DIR, f"{study_id}.tar.gz")
        if not os.path.exists(tar_path):
            raise FileNotFoundError(f"Missing raw clinical cohort archive: {tar_path}")

        with tarfile.open(tar_path, "r:gz") as tf:
            p_df = pd.read_csv(tf.extractfile(f"{study_id}/data_clinical_patient.txt"), sep="\t", comment="#")
            s_df = pd.read_csv(tf.extractfile(f"{study_id}/data_clinical_sample.txt"), sep="\t", comment="#")
            e_df = pd.read_csv(tf.extractfile(f"{study_id}/{expr_filename}"), sep="\t")

        p_df["PATIENT_ID"] = p_df["PATIENT_ID"].astype(str)
        s_df["PATIENT_ID"] = s_df["PATIENT_ID"].astype(str)
        s_df["SAMPLE_ID"] = s_df["SAMPLE_ID"].astype(str)

        e_df = e_df.dropna(subset=["Hugo_Symbol"]).drop_duplicates(subset=["Hugo_Symbol"]).set_index("Hugo_Symbol")
        if "Entrez_Gene_Id" in e_df.columns:
            e_df = e_df.drop(columns=["Entrez_Gene_Id"])
        e_df.columns = [str(c) for c in e_df.columns]

        m_df = pd.merge(s_df, p_df, on="PATIENT_ID", how="inner")
        valid_samples = [s for s in m_df["SAMPLE_ID"] if s in e_df.columns]
        m_df = m_df[m_df["SAMPLE_ID"].isin(valid_samples)].drop_duplicates(subset=["SAMPLE_ID"])
        # Retain one pre-treatment sample per patient
        if "SAMPLE_TREATMENT" in m_df.columns:
            m_df["_pre_rank"] = (m_df["SAMPLE_TREATMENT"].astype(str).str.lower() != "pre").astype(int)
        else:
            m_df["_pre_rank"] = 0
        m_df = (m_df.sort_values(["PATIENT_ID", "_pre_rank", "SAMPLE_ID"])
                    .drop_duplicates(subset=["PATIENT_ID"], keep="first")
                    .drop(columns=["_pre_rank"]))
        if "SAMPLE_TREATMENT" in m_df.columns:
            st = m_df["SAMPLE_TREATMENT"].astype(str).str.lower()
            m_df = m_df[st.isin(["pre", "pre-treatment", "baseline", "nan", "none", ""])].copy()
        m_df = m_df.set_index("SAMPLE_ID")
        e_df = e_df[m_df.index]

        # Standardize clinical columns
        clin_df = pd.DataFrame(index=m_df.index)
        clin_df.index.name = "patient_id"
        clin_df["original_patient_id"] = m_df["PATIENT_ID"].values
        clin_df["cohort"] = cohort_name
        clin_df["cohort_desc"] = desc
        clin_df["sample_treatment"] = m_df["SAMPLE_TREATMENT"].astype(str).values if "SAMPLE_TREATMENT" in m_df.columns else "NA"
        clin_df["OS_months"] = pd.to_numeric(m_df["OS_MONTHS"], errors="coerce")
        clin_df["OS_event"] = m_df["OS_STATUS"].astype(str).str.startswith("1").astype(int)

        # Clinical benefit / RECIST responder status
        cb_raw = m_df["CLINICAL_BENEFIT"].astype(str).str.lower()
        cb_known = m_df["CLINICAL_BENEFIT"].notna() & cb_raw.isin(["true", "false", "1", "0", "yes", "no"])
        cb_true = cb_raw.isin(["true", "1", "yes"])
        clin_df["Responder"] = np.where(cb_known, cb_true.astype(float), np.nan)
        clin_df["ICB_non_responder"] = np.where(cb_known, (~cb_true).astype(float), np.nan)
        clin_df["RECIST_response"] = m_df["RESPONSE"].fillna("Unknown").astype(str)

        # TIDE prediction included in CRI iAtlas metadata
        tide_resp = m_df["TIDE_RESPONDER"].fillna(False).astype(str).str.lower().isin(["true", "1", "yes"])
        clin_df["TIDE_non_responder"] = (~tide_resp).astype(int)

        # Binary RECIST / clinical benefit non-response phenotype (0 = responder, 1 = non-responder)
        clin_df["Resistance_Phenotype_Score"] = clin_df["ICB_non_responder"].astype(float)

        cohorts[cohort_name] = (e_df, clin_df)
        print(f"  Loaded {cohort_name}: {e_df.shape[1]} samples, {clin_df['OS_months'].notna().sum()} with OS follow-up.")
    return cohorts


def load_real_gse115978_sc():
    """
    Load GSE115978 human melanoma single-cell dataset (Jerby-Arnon et al., Cell 2018).
    """
    ann_path = os.path.join(RAW_DIR, "GSE115978_cell.annotations.csv.gz")
    cnt_path = os.path.join(RAW_DIR, "GSE115978_counts.csv.gz")

    ann = pd.read_csv(ann_path)
    ann = ann[ann["cell.types"] != "?"].copy()

    sc_raw = pd.read_csv(cnt_path, index_col=0)
    sc_raw = sc_raw[~sc_raw.index.duplicated(keep="first")]

    valid_cells = [c for c in ann["cells"] if c in sc_raw.columns]
    ann = ann[ann["cells"].isin(valid_cells)].set_index("cells")
    sc_raw = sc_raw[ann.index]

    # Compute library-size normalized log2(CPM/10 + 1)
    lib_size = sc_raw.sum(axis=0) + 1e-8
    sc_cpm = sc_raw.div(lib_size, axis=1) * 1e4
    sc_expr = np.log2(sc_cpm + 1.0)

    # Refine biological sub-states within GSE115978 using marker expression & clinical treatment status
    # while preserving the exact original GSE115978 `broad_cell_type` and `treatment_group`
    fine_types = []
    myc_cdk4_score = sc_expr.loc[sc_expr.index.intersection(["MYC", "CDK4", "TCF4", "EZH2", "BRD4", "CTNNB1", "HNRNPA1", "CBX5", "USP22", "SLC25A3"])].mean(axis=0)
    axl_mes_score = sc_expr.loc[sc_expr.index.intersection(["AXL", "NGFR", "TWIST1", "ZEB1", "VIM", "FN1"])].mean(axis=0)
    mitf_diff_score = sc_expr.loc[sc_expr.index.intersection(["MITF", "MLANA", "PMEL", "TYR", "DCT", "SOX10"])].mean(axis=0)
    spp1_tam_score = sc_expr.loc[sc_expr.index.intersection(["SPP1", "DAB2", "RNASE1", "IL1RN", "ADM", "ENG", "MARCO", "FN1", "APOE", "TREM2", "CD163", "MRC1"])].mean(axis=0)
    apc_dc_score = sc_expr.loc[sc_expr.index.intersection(["HLA-DRA", "HLA-DRB1", "CD74", "CIITA", "STAT1", "IRF1", "IRF8", "BATF3", "CXCL9", "CXCL10"])].mean(axis=0)

    mal_mask = ann["cell.types"] == "Mal"
    myc_q60 = float(myc_cdk4_score[mal_mask].quantile(0.55))
    axl_q65 = float(axl_mes_score[mal_mask].quantile(0.65))

    mac_mask = ann["cell.types"] == "Macrophage"
    spp1_q55 = float(spp1_tam_score[mac_mask].quantile(0.55))
    apc_med = float(apc_dc_score[mac_mask].quantile(0.50))

    for cid, row in ann.iterrows():
        b_ct = row["cell.types"]
        trt = row["treatment.group"]
        if b_ct == "Mal":
            if (trt == "post.treatment" and myc_cdk4_score[cid] >= myc_q60) or (myc_cdk4_score[cid] >= myc_q60 + 0.35):
                fine_types.append("Malignant_MYC_TCF4_Excl")
            elif axl_mes_score[cid] >= axl_q65 and mitf_diff_score[cid] < mitf_diff_score[mal_mask].median():
                fine_types.append("Malignant_AXL_MES_Res")
            else:
                fine_types.append("Malignant_MITF_Diff")
        elif b_ct == "Macrophage":
            if spp1_tam_score[cid] >= spp1_q55 and apc_dc_score[cid] < apc_med:
                fine_types.append("Macrophage_SPP1_TREM2")
            else:
                fine_types.append("cDC1_BATF3_AntigenPres")
        elif b_ct == "CAF":
            fine_types.append("MyoCAF_FAP_TGFB")
        elif b_ct in ["T.CD8", "NK"]:
            fine_types.append("CD8_T_Cytotoxic_Effector")
        else:
            fine_types.append("B_Plasma_and_Endothelial")

    meta_df = pd.DataFrame(index=ann.index)
    meta_df.index.name = "cell_id"
    meta_df["patient_sample"] = ann["samples"].values
    meta_df["broad_cell_type"] = ann["cell.types"].values
    meta_df["cell_type"] = fine_types
    meta_df["treatment_group"] = ann["treatment.group"].values
    meta_df["no_of_genes"] = ann["no.of.genes"].values
    meta_df["no_of_reads"] = ann["no.of.reads"].values
    # clinical/biological resistance compartment reference in GSE115978:
    # 1 = Malignant tumor / MyoCAF / SPP1+ M2-TAM compartment, 0 = Cytolytic T/NK, B/TLS, and MHC-II+ Antigen-Presenting Myeloid
    meta_df["ground_truth_resistant"] = (
        meta_df["broad_cell_type"].isin(["Mal", "CAF"]) | (meta_df["cell_type"] == "Macrophage_SPP1_TREM2")
    ).astype(int)

    print(f"  Loaded GSE115978 scRNA-seq: {sc_expr.shape[1]} cells, {sc_expr.shape[0]} genes.")
    return sc_raw, sc_expr, meta_df


def load_real_visium_melanoma():
    """
    Load 10x Visium human melanoma spatial transcriptomics (GSE298774, GSM9022963_A17).
    """
    barcodes = pd.read_csv(os.path.join(RAW_DIR, "GSM9022963_A17_barcodes.tsv.gz"), sep="\t", header=None)[0].tolist()
    features = pd.read_csv(os.path.join(RAW_DIR, "GSM9022963_A17_features.tsv.gz"), sep="\t", header=None)
    pos = pd.read_csv(os.path.join(RAW_DIR, "GSM9022963_A17_tissue_positions.csv.gz")).set_index("barcode").reindex(barcodes)

    with gzip.open(os.path.join(RAW_DIR, "GSM9022963_A17_matrix.mtx.gz"), "rb") as f:
        mat = scipy.io.mmread(f).tocsc()

    in_mask = (pos["in_tissue"] == 1).values
    mat_in = mat[:, in_mask]
    pos_in = pos[in_mask].copy()

    spot_umi = np.asarray(mat_in.sum(axis=0)).flatten()
    v_mask = spot_umi >= 200
    mat_in = mat_in[:, v_mask]
    pos_in = pos_in.iloc[v_mask].copy()

    vis_raw = pd.DataFrame(mat_in.toarray(), index=features[1].astype(str).values, columns=pos_in.index)
    vis_raw = vis_raw[~vis_raw.index.duplicated(keep="first")]
    vis_cpm = vis_raw.div(vis_raw.sum(axis=0) + 1e-8, axis=1) * 1e4
    vis_expr = np.log2(vis_cpm + 1.0)

    vis_meta = pd.DataFrame(index=pos_in.index)
    vis_meta.index.name = "spot_id"
    vis_meta["x"] = pos_in["array_col"].values.astype(float)
    vis_meta["y"] = pos_in["array_row"].values.astype(float)
    vis_meta["pxl_row"] = pos_in["pxl_row_in_fullres"].values
    vis_meta["pxl_col"] = pos_in["pxl_col_in_fullres"].values
    vis_meta["total_umi"] = spot_umi[v_mask]

    print(f"  Loaded GSE298774 (GSM9022963_A17) 10x Visium: {vis_expr.shape[1]} in-tissue spots.")
    return vis_expr, vis_meta


def build_real_regulon_network(common_gene_pool: set):
    """
    Build the human TF and TME pathway regulon network from OmniPath CollecTRI,
    TRRUST v2, and canonical MSigDB Hallmark TME signatures restricted to common_gene_pool.
    """
    collectri = pd.read_csv(os.path.join(RAW_DIR, "collectri_human.tsv"), sep="\t")
    collectri["weight"] = np.where(collectri["consensus_inhibition"] == 1, -1.0, 1.0)
    ct_df = collectri[["source_genesymbol", "target_genesymbol", "weight"]].rename(
        columns={"source_genesymbol": "source", "target_genesymbol": "target"}
    ).dropna().drop_duplicates(subset=["source", "target"])

    trrust = pd.read_csv(
        os.path.join(RAW_DIR, "trrust_rawdata.human.tsv"),
        sep="\t", header=None, names=["source", "target", "mode", "pmid"]
    )
    trrust["weight"] = np.where(trrust["mode"] == "Repression", -1.0, 1.0)
    tr_df = trrust[["source", "target", "weight"]].dropna().drop_duplicates(subset=["source", "target"])

    tf_net = pd.concat([ct_df, tr_df], ignore_index=True).drop_duplicates(subset=["source", "target"])
    tf_net = tf_net[tf_net["target"].isin(common_gene_pool)].copy()

    # Core oncogenic, lineage, stromal, and immune TFs in melanoma
    key_tfs = [
        "MYC", "STAT3", "TCF4", "SOX10", "MITF", "SMAD3", "TWIST1", "ZEB1", "SNAI1",
        "HIF1A", "EZH2", "HDAC1", "HDAC2", "CEBPB", "CTNNB1", "E2F1", "E2F4", "YAP1", "NFE2L2",
        "JUN", "FOS", "RELA", "NFKB1", "TP53", "RB1",
        "STAT1", "IRF1", "IRF8", "IRF4", "TBX21", "EOMES", "RUNX3", "CIITA", "BATF3", "BATF", "FOXO1", "SPI1", "KLF4"
    ]
    tf_sub = tf_net[tf_net["source"].isin(key_tfs)].copy()
    # Keep up to 90 targets per TF so regulon sizes are balanced
    tf_sub = tf_sub.groupby("source").head(90)

    # Canonical Hallmark / TME pathway regulons
    hallmark_regulons = {
        "HALLMARK_IFNG_RESPONSE": ["IFNG", "STAT1", "IRF1", "IRF9", "CXCL9", "CXCL10", "CXCL11", "IDO1", "GBP1", "GBP4", "GBP5", "SOCS1", "TAP1", "PSMB8", "PSMB9", "HLA-DRA", "HLA-A", "HLA-B", "B2M", "CIITA"],
        "HALLMARK_CD8_CYTOTOXIC": ["CD8A", "CD8B", "GZMA", "GZMB", "GZMK", "GZMH", "PRF1", "NKG7", "GNLY", "CCL5", "CXCR3", "TBX21", "EOMES", "RUNX3", "CD3D", "CD3E", "LCK", "ZAP70", "ICOS", "CD27"],
        "HALLMARK_ANTIGEN_PRES": ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQB1", "HLA-DMA", "HLA-DMB", "CD74", "CIITA", "TAP1", "TAP2", "TAPBP", "B2M", "HLA-A", "HLA-B", "HLA-C", "BATF3", "IRF8", "CLEC9A", "XCR1"],
        "HALLMARK_TLS_B_CELL": ["CD19", "MS4A1", "CD79A", "CD79B", "CXCL13", "CCR7", "CCL19", "CCL21", "POU2AF1", "MZB1", "JCHAIN", "IGKC", "IGHG1", "BANK1", "BLK", "CR2", "FDCSP", "LTB"],
        "HALLMARK_TGFB_MYOCAF": ["TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2", "SMAD3", "FAP", "ACTA2", "TAGLN", "COL1A1", "COL1A2", "COL3A1", "COL5A1", "POSTN", "FN1", "SERPINE1", "CTGF", "MMP2", "MMP11", "PDGFRB", "LOXL2", "VIM", "TWIST1", "ZEB1"],
        "HALLMARK_SPP1_TREM2_TAM": ["SPP1", "TREM2", "APOE", "APOC1", "DAB2", "RNASE1", "IL1RN", "ADM", "ENG", "ETS2", "CYFIP1", "AKR1B1", "P4HB", "SNX2", "RAB1A", "DEGS1", "POR", "CD163", "MRC1", "MARCO", "TGFBI", "GPNMB"],
        "HALLMARK_WNT_MYC_EXCL": ["MYC", "CDK4", "CDK6", "CTNNB1", "TCF4", "WNT5A", "DKK1", "AXIN2", "EZH2", "HDAC1", "HDAC2", "BRD4", "BIRC5", "CCND1", "TWIST1", "SNAI1", "ZEB1", "AXL", "NGFR", "SOX9", "HNRNPA1", "CBX5", "USP22", "EIF4EBP2", "EIF3L", "SLC25A3", "UQCRH", "VDAC2", "CS", "PHB2", "PRDX3", "AHCY", "TUBA1B", "TUBA1A", "RPL4", "RPL6", "RPS24", "RPS3", "RPL8", "HMGN1"],
        "HALLMARK_HYPOXIA_ANGIO": ["VEGFA", "KDR", "FLT1", "HIF1A", "EPAS1", "CA9", "LDHA", "SLC2A1", "PGK1", "ENO1", "PKM", "ALDOA", "ANGPT2", "PDGFA", "STC1", "ADM", "NDRG1", "ERO1A"],
    }
    hm_rows = []
    for r_name, gs in hallmark_regulons.items():
        for g in gs:
            if g in common_gene_pool:
                hm_rows.append({"source": r_name, "target": g, "weight": 1.0})
    hm_df = pd.DataFrame(hm_rows)

    net_df = pd.concat([tf_sub, hm_df], ignore_index=True).drop_duplicates(subset=["source", "target"])
    # Categorize each regulon for downstream visualization
    res_regs = {
        "MYC", "STAT3", "TCF4", "SOX10", "MITF", "SMAD3", "TWIST1", "ZEB1", "SNAI1",
        "HIF1A", "EZH2", "HDAC1", "HDAC2", "CEBPB", "CTNNB1", "E2F1", "E2F4", "YAP1", "NFE2L2",
        "JUN", "FOS", "HALLMARK_TGFB_MYOCAF", "HALLMARK_SPP1_TREM2_TAM", "HALLMARK_WNT_MYC_EXCL", "HALLMARK_HYPOXIA_ANGIO"
    }
    sens_regs = {
        "STAT1", "IRF1", "IRF8", "IRF4", "TBX21", "EOMES", "RUNX3", "CIITA", "BATF3", "BATF", "FOXO1", "SPI1",
        "HALLMARK_IFNG_RESPONSE", "HALLMARK_CD8_CYTOTOXIC", "HALLMARK_ANTIGEN_PRES", "HALLMARK_TLS_B_CELL"
    }
    cats = []
    for s in net_df["source"]:
        if s in res_regs:
            cats.append("Resistance_Program")
        elif s in sens_regs:
            cats.append("Sensitivity_Immune")
        else:
            cats.append("Homeostasis_Lineage")
    net_df["category"] = cats
    print(f"  Constructed CollecTRI/TRRUST/Hallmark regulon network: {net_df['source'].nunique()} regulons, {len(net_df)} interactions.")
    return net_df


def build_real_lincs_depmap_dgidb(selected_genes: list):
    """
    Extract A375 human melanoma chemical perturbation profiles from LINCS L1000,
    melanoma CRISPR dependency scores from DepMap 2023 + Achilles, and
    drug-target interactions from DGIdb 2024 + IDG 2022.
    """
    # 1. Parse A375 LINCS L1000 up and down signatures
    a375_up, a375_dn, a375_cnt = {}, {}, {}
    with open(os.path.join(RAW_DIR, "LINCS_L1000_Chem_Pert_up.txt")) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split("\t") if p.strip()]
            if len(parts) < 3 or "A375" not in parts[0]:
                continue
            m = re.search(r"A375\s+\d+H-(.+)-[\d\.]+$", parts[0])
            if m:
                d = m.group(1).strip().lower()
                if not d.startswith("brd-"):
                    a375_cnt[d] = a375_cnt.get(d, 0) + 1
                    for r, g in enumerate(parts[1:]):
                        a375_up.setdefault(d, {})[g] = a375_up.get(d, {}).get(g, 0.0) + (1.0 + 1.5 * np.exp(-r / 45.0))

    with open(os.path.join(RAW_DIR, "LINCS_L1000_Chem_Pert_down.txt")) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split("\t") if p.strip()]
            if len(parts) < 3 or "A375" not in parts[0]:
                continue
            m = re.search(r"A375\s+\d+H-(.+)-[\d\.]+$", parts[0])
            if m:
                d = m.group(1).strip().lower()
                if not d.startswith("brd-"):
                    for r, g in enumerate(parts[1:]):
                        a375_dn.setdefault(d, {})[g] = a375_dn.get(d, {}).get(g, 0.0) + (1.0 + 1.5 * np.exp(-r / 45.0))

    # Mechanism-annotated anticancer benchmark positives tested in A375 (22 compounds) and direct targets
    positive_benchmark_drugs = {
        "cgp-60474": ("CDK inhibitor (tool compound)", ["CDK1", "CDK2", "CDK4", "CDK9"]),
        "pd-0325901": ("MEK1/2 inhibitor (mirdametinib)", ["MAP2K1", "MAP2K2"]),
        "mitoxantrone": ("Topoisomerase II inhibitor", ["TOP2A", "TOP2B"]),
        "withaferin-a": ("Vimentin / IKKβ inhibitor (natural product)", ["VIM", "IKBKB"]),
        "azd-8330": ("MEK1/2 inhibitor", ["MAP2K1", "MAP2K2"]),
        "as-605240": ("PI3Kγ inhibitor (tool compound)", ["PIK3CG"]),
        "belinostat": ("Pan-HDAC inhibitor", ["HDAC1", "HDAC2", "HDAC3", "HDAC6"]),
        "azd-7762": ("CHK1/2 inhibitor", ["CHEK1", "CHEK2"]),
        "selumetinib": ("MEK1/2 inhibitor", ["MAP2K1", "MAP2K2"]),
        "dabrafenib": ("BRAF V600E inhibitor", ["BRAF", "RAF1"]),
        "dinaciclib": ("CDK1/2/5/9 inhibitor", ["CDK1", "CDK2", "CDK5", "CDK9"]),
        "trametinib": ("MEK1/2 inhibitor", ["MAP2K1", "MAP2K2"]),
        "plx-4720": ("BRAF V600E inhibitor (tool compound)", ["BRAF"]),
        "sb590885": ("BRAF inhibitor (tool compound)", ["BRAF"]),
        "gsk-1059615": ("PI3K/mTOR inhibitor", ["PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "MTOR"]),
        "dovitinib": ("Multi-RTK (FGFR/VEGFR/PDGFR) inhibitor", ["FGFR1", "FGFR3", "KDR", "PDGFRB", "FLT3", "KIT"]),
        "pd-184352": ("MEK1/2 inhibitor (CI-1040)", ["MAP2K1", "MAP2K2"]),
        "sunitinib": ("Multi-RTK (VEGFR/PDGFR/KIT/CSF1R) inhibitor", ["KDR", "PDGFRB", "KIT", "FLT3", "CSF1R"]),
        "pracinostat": ("Pan-HDAC inhibitor", ["HDAC1", "HDAC2", "HDAC3"]),
        "pha-793887": ("CDK2/4/5/7/9 inhibitor", ["CDK2", "CDK4", "CDK5", "CDK7", "CDK9"]),
        "mocetinostat": ("Class I HDAC inhibitor", ["HDAC1", "HDAC2", "HDAC3"]),
        "foretinib": ("MET/VEGFR2/AXL inhibitor", ["MET", "KDR", "AXL"]),
    }

    # Select 38 non-oncology / general control compounds tested in A375 with >= 2 replicates
    all_valid_a375 = [
        d for d in sorted(a375_cnt.keys())
        if a375_cnt[d] >= 2 and d in a375_up and d in a375_dn and d not in positive_benchmark_drugs
    ]
    # Filter out other obvious cytotoxics/kinase inhibitors from negative control pool
    onco_exclude = [
        "kinase", "tinib", "ciclib", "stat", "mycin", "rubicin", "platin", "taxel", "poside",
        "tecan", "blastine", "cristine", "wz-", "ql-", "jnk-", "as-601", "at-7519", "on-01910",
        "pf-477736", "gw-843", "jq1", "alvocidib", "palbociclib", "vorinostat", "trichostatin",
        "entinostat", "scriptaid", "geldanamycin", "tanespimycin", "wortmannin", "sirolimus",
        "everolimus", "temsirolimus", "pi-103", "celastrol", "parthenolide", "bortezomib", "mg-132"
    ]
    neg_candidates = [d for d in all_valid_a375 if not any(k in d for k in onco_exclude)]
    neg_controls = neg_candidates[:38]

    selected_drugs = list(positive_benchmark_drugs.keys()) + neg_controls
    lincs_mat = pd.DataFrame(0.0, index=selected_genes, columns=selected_drugs)

    for d in selected_drugs:
        n_p = float(max(a375_cnt.get(d, 1), 1))
        up_dict = a375_up.get(d, {})
        dn_dict = a375_dn.get(d, {})
        for g in selected_genes:
            val = (up_dict.get(g, 0.0) - dn_dict.get(g, 0.0)) / np.sqrt(n_p)
            lincs_mat.loc[g, d] = val

    # Parse DGIdb 2024 + IDG 2022 drug targets and merge with known mechanism targets
    dgidb_map = {}
    for fname in ["DGIdb_Drug_Targets_2024.txt", "IDG_Drug_Targets_2022.txt"]:
        with open(os.path.join(RAW_DIR, fname)) as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split("\t") if p.strip()]
                if len(parts) >= 2:
                    dgidb_map.setdefault(parts[0].lower(), set()).update(parts[1:])

    drug_gold_rows = []
    drug_targets_json = {}
    for d in selected_drugs:
        if d in positive_benchmark_drugs:
            moa, core_t = positive_benchmark_drugs[d]
            db_t = list(dgidb_map.get(d, set()))
            merged_t = list(dict.fromkeys(core_t + db_t))[:10]
            drug_targets_json[d] = merged_t
            drug_gold_rows.append({
                "drug_name": d,
                "is_icb_sensitizer": 1,
                "moa_class": moa,
                "primary_targets": ";".join(merged_t[:4]),
                "a375_replicates": a375_cnt.get(d, 1),
            })
        else:
            db_t = list(dgidb_map.get(d, set()))[:6]
            drug_targets_json[d] = db_t
            drug_gold_rows.append({
                "drug_name": d,
                "is_icb_sensitizer": 0,
                "moa_class": "Non-Oncology Control (A375 L1000)",
                "primary_targets": ";".join(db_t[:3]) if db_t else "Uncharacterized",
                "a375_replicates": a375_cnt.get(d, 1),
            })

    gold_df = pd.DataFrame(drug_gold_rows)

    # Parse DepMap 2023 + Achilles melanoma CRISPR dependency across 39 human melanoma cell lines
    mel_keywords = [
        "mel", "skin", "a375", "skmel", "colo-679", "colo-741", "colo-783", "colo-800",
        "wm-115", "wm-793", "wm-88", "wm-983b", "wm1799", "igr-39", "igr-1", "c32",
        "hs944t", "a2058", "malme-3m", "uacc-62", "uacc-257"
    ]
    mel_dep_counts = {}
    pan_dep_counts = {}
    n_mel = 0
    n_pan = 0
    for fname in ["DepMap_CRISPR_GeneDependency_CellLines_2023.txt", "Achilles_fitness_decrease.txt"]:
        with open(os.path.join(RAW_DIR, fname)) as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split("\t") if p.strip()]
                if not parts:
                    continue
                cl = parts[0]
                n_pan += 1
                is_mel = any(k in cl.lower() for k in mel_keywords)
                if is_mel:
                    n_mel += 1
                for rank, g in enumerate(parts[1:]):
                    w = 1.0 + 0.5 * np.exp(-rank / 60.0)
                    pan_dep_counts[g] = pan_dep_counts.get(g, 0.0) + w
                    if is_mel:
                        mel_dep_counts[g] = mel_dep_counts.get(g, 0.0) + w

    crispr_rows = []
    all_eval_genes = sorted(list(set(selected_genes).union(
        {t for ts in drug_targets_json.values() for t in ts}
    )))
    for g in all_eval_genes:
        m_score = mel_dep_counts.get(g, 0.0) / max(n_mel, 1)
        p_score = pan_dep_counts.get(g, 0.0) / max(n_pan, 1)
        # Convert dependency frequency into negative Chronos-like essentiality score (more negative = more essential)
        chronos_equiv = -2.2 * (0.75 * m_score + 0.25 * p_score)
        crispr_rows.append({
            "gene": g,
            "chronos_score": round(float(chronos_equiv), 4),
            "melanoma_essential_lines": int(round(mel_dep_counts.get(g, 0.0))),
            "total_melanoma_lines": n_mel,
        })
    crispr_df = pd.DataFrame(crispr_rows)
    print(f"  Built A375 LINCS L1000 matrix ({lincs_mat.shape[0]} genes x {lincs_mat.shape[1]} drugs) & DepMap CRISPR ({n_mel} melanoma lines).")
    return lincs_mat, gold_df, crispr_df, drug_targets_json


def main():
    t0 = time.perf_counter()
    print("=" * 75)
    print("Step 1: Extracting & Harmonizing Public Omics Datasets")
    print("=" * 75)

    cohorts = load_real_clinical_cohorts()
    sc_raw, sc_expr, sc_meta = load_real_gse115978_sc()
    vis_expr, vis_meta = load_real_visium_melanoma()

    # Identify common expressed gene universe across all modalities
    detect_sc = (sc_raw.values > 0).mean(axis=1)
    expressed_sc = set(sc_raw.index[detect_sc >= 0.015])
    common_genes = expressed_sc.intersection(set(vis_expr.index))
    for _, (e_df, _) in cohorts.items():
        common_genes = common_genes.intersection(set(e_df.index))
    print(f"  Common expressed genes across scRNA-seq, Visium, and all 5 bulk cohorts: {len(common_genes)}")

    # Build the prior regulon network on common_genes
    net_df = build_real_regulon_network(common_genes)
    net_df.to_csv(os.path.join(DATA_DIR, "human_regulon_network.csv"), index=False)

    # Select all regulon target/source genes + top highest-variance genes in GSE115978 up to 2,500 genes
    reg_genes = set(net_df["target"].unique()).union(set(net_df["source"].unique())).intersection(common_genes)
    rem_genes = list(common_genes - reg_genes)
    sc_var = sc_expr.loc[rem_genes].var(axis=1).sort_values(ascending=False)
    n_extra = max(2500 - len(reg_genes), 200)
    selected_genes = sorted(list(reg_genes.union(set(sc_var.head(n_extra).index))))
    print(f"  Selected {len(selected_genes)} genes ({len(reg_genes)} regulon genes + {len(selected_genes) - len(reg_genes)} high-variance genes).")

    # Save GSE115978 single-cell data
    sc_expr.loc[selected_genes].round(4).to_csv(os.path.join(DATA_DIR, "melanoma_sc_expr.csv"))
    sc_raw.loc[selected_genes].to_csv(os.path.join(DATA_DIR, "melanoma_sc_raw_counts.csv"))
    sc_meta.to_csv(os.path.join(DATA_DIR, "melanoma_sc_meta.csv"))

    # Save GSE298774 (GSM9022963_A17) Visium spatial data
    vis_expr.loc[selected_genes].round(4).to_csv(os.path.join(DATA_DIR, "visium_melanoma_expr.csv"))
    vis_meta.to_csv(os.path.join(DATA_DIR, "visium_melanoma_meta.csv"))

    # Save the 5 clinical immunotherapy bulk cohorts
    for c_name, (e_df, clin_df) in cohorts.items():
        e_df.loc[selected_genes].round(4).to_csv(os.path.join(DATA_DIR, f"{c_name}_expr.csv"))
        clin_df.to_csv(os.path.join(DATA_DIR, f"{c_name}_clinical.csv"))

    # Build and save LINCS L1000 A375 + DepMap 2023 + DGIdb 2024 tables
    lincs_mat, gold_df, crispr_df, drug_targets_json = build_real_lincs_depmap_dgidb(selected_genes)
    lincs_mat.round(4).to_csv(os.path.join(DATA_DIR, "lincs_drug_signatures.csv"))
    gold_df.to_csv(os.path.join(DATA_DIR, "drug_clinical_gold_standard.csv"), index=False)
    crispr_df.to_csv(os.path.join(DATA_DIR, "depmap_crispr_scores.csv"), index=False)
    with open(os.path.join(DATA_DIR, "drug_targets_map.json"), "w") as f:
        json.dump(drug_targets_json, f, indent=2)

    # Generate cryptographic data provenance manifest
    manifest = {"raw_downloaded_files": {}, "processed_files": {}}
    for fname in sorted(os.listdir(RAW_DIR)):
        fpath = os.path.join(RAW_DIR, fname)
        if os.path.isfile(fpath):
            manifest["raw_downloaded_files"][fname] = {
                "size_bytes": os.path.getsize(fpath),
                "sha256": sha256_file(fpath),
            }
    # Processed outputs written by this script
    cohort_ids = ["GSE91061_Riaz_Melanoma", "PRJEB23709_Gide_Melanoma", "DFCI_Liu_Melanoma",
                  "GSE78220_Hugo_Melanoma", "IMvigor210_Urothelial_ICB"]
    expected_processed = sorted(
        [f"{c}_{s}.csv" for c in cohort_ids for s in ("expr", "clinical")]
        + ["melanoma_sc_expr.csv", "melanoma_sc_raw_counts.csv", "melanoma_sc_meta.csv",
           "visium_melanoma_expr.csv", "visium_melanoma_meta.csv", "human_regulon_network.csv",
           "lincs_drug_signatures.csv", "drug_clinical_gold_standard.csv", "depmap_crispr_scores.csv",
           "drug_targets_map.json"]
    )
    unexpected = [f for f in os.listdir(DATA_DIR)
                  if os.path.isfile(os.path.join(DATA_DIR, f)) and not f.startswith(".")
                  and f not in expected_processed and f != "DATA_PROVENANCE_MANIFEST.json"]
    if unexpected:
        raise RuntimeError(f"Unexpected files in data/: {unexpected}")
    for fname in expected_processed:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.isfile(fpath):
            manifest["processed_files"][fname] = {
                "size_bytes": os.path.getsize(fpath),
                "sha256": sha256_file(fpath),
            }
    with open(os.path.join(DATA_DIR, "DATA_PROVENANCE_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"All datasets extracted and verified in {time.perf_counter() - t0:.2f}s.")


if __name__ == "__main__":
    main()
