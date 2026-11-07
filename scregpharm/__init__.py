"""
scRegPharm: Single-Cell Regulon-Guided Phenotype Mapping, Spatial Niche Analysis,
and Pharmacological Reversal Framework.
"""

__version__ = "1.0.0"

from .core import (
    compute_regulon_activity,
    scregpharm_map,
    run_scissor_baseline,
    run_scpas_baseline,
    run_degas_corr_baseline,
    run_lp_sgl_baseline,
    scregpharm_spatial_score,
    scregpharm_drug_prioritization,
    run_drugresc_baseline,
    run_bulk_cmap_baseline,
    run_asgard_baseline,
)
