# Reproducibility Code

## Synthetic Experiments

- `run_exp1.py`: 1D sanity check reported in Appendix C.
- `run_exp2.py`: Muller-Brown proposal-limited rare-event benchmark reported in Appendix C.
- `run_exp2_take2_asymmetric.py`: asymmetric score diagnostic for the Take 2 criterion.
- `run_exp3.py`: 2D conditional-generation benchmark reported in Appendix C.
- `run_exp4.py`: 6D double-well benchmark reported in the main text.
- `run_exp8_nonmarkov_route.py`: non-Markovian route-history benchmark reported in the main text and appendix.
- `diagnostics.py`: variance-decomposition diagnostic used for Appendix C.

## Molecular Experiments

- `run_exp5_alanine.py`: alanine dipeptide endpoint and midpoint transition benchmark.
- `run_exp5_alanine_rarity_sweep.py`: alanine harder-task sweep summarized in Appendix C.
- `run_exp6_tetrapeptide_tps.py`: tetrapeptide transfer TPS benchmark.
- `summarize_exp6_transfer.py`: aggregates the three reported tetrapeptide seeds.

## Shared Modules

- `common.py`
- `fixed_twist.py`
- `learned_scores.py`
- `double_well_core.py`
- `muller_brown_core.py`
- `alanine.py`
- `tetrapeptide_tps.py`
- `prep_tetrapeptide_data.py`

## Data

- Alanine uses `alanine-dipeptide-3x250ns-backbone-dihedrals.npz`.
- Tetrapeptide uses preprocessed torsion files under `../data/tetrapeptide_npz_4AA_test_50/` and the split CSV `4AA_test.csv`.


