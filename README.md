[README.md](https://github.com/user-attachments/files/28788983/README.md)
# AML Mule Account Detection — Team HxH

**Competition:** RBIH National Financial Products Challenge (NFPC) Phase 2
**Final Private Result:** AUC-ROC `0.999833` | F1 `0.971609` | Temporal IoU `0.6879` | RH_Avoidance_1: 0.9985 | RH_Avoidance_2: 1.0000 | RH_Avoidance_3: 1.0000 | RH_Avoidance_4: 0.9947 | RH_Avoidance_5: 1.0000 | RH_Avoidance_6: 1.0000 | RH_Avoidance_7: 0.8571

## 1. Environment Setup

### Python Version
Python **3.10+** required. Developed and tested on Python 3.12.

### Install Dependencies

```bash
pip install lightgbm>=4.0
pip install xgboost>=2.0
pip install catboost>=1.2
pip install scikit-learn>=1.3
pip install pandas>=2.0
pip install numpy>=1.24
pip install pyarrow>=12.0
pip install tqdm
pip install matplotlib seaborn
pip install optuna              # optional: hyperparameter tuning
```

> **Note:** `xgboost` and `catboost` are optional. The pipeline detects their presence at runtime and skips M4/M5 with a warning if they are not installed. The ensemble will still run with the remaining models.

### Hardware & Runtime

| Requirement | Details |
|---|---|
| Platform | CPU only (no GPU required) |
| RAM | 32 GB+ recommended |
| Disk | ~20 GB free (16.2 GB dataset + working files) |
| Runtime | ~3 hours end-to-end (~10,634 s on reference hardware) |
| Environment | **Kaggle CPU Notebook** — the entire solution was developed and executed on Kaggle's hosted notebook environment (Python 3.12, Docker image `31286`) |

### Data Files

> **Kaggle Environment:** This solution was developed and executed entirely on Kaggle's hosted CPU notebook environment. The dataset was attached via Kaggle's dataset interface and all outputs were written to `/kaggle/working/`. No local setup is required if running on Kaggle — simply attach the dataset and run all cells in order.

Place all data files under the `BASE` directory. The notebook auto-detects the path in this order:
1. `/kaggle/input/datasets/abhyudayrbih/rbih-nfpc-phase-2/`
2. `/kaggle/input/rbih-nfpc-phase-2/`
3. Any subdirectory of `/kaggle/input/` containing `train_labels.parquet`

| File | Description |
|---|---|
| `train_labels.parquet` | Ground-truth mule labels for 96,091 accounts |
| `accounts.parquet` | Static account-level features |
| `transactions/*.parquet` | 396 parquet parts (~400 M rows, 16 GB) |
| `transactions_additional/` | Supplemental data (IP, geo, CI ratio) |
| `test_accounts.parquet` | 64,062 test accounts to predict |

---

## 2. Steps to Reproduce Results

> Run all cells in order top to bottom. Do not skip cells — later steps depend on variables produced by earlier ones.

1. **Visualisation Setup** — import matplotlib, define colour palette constants (`DARK`, `BLUE`, etc.).
2. **Setup & Static Data Loading** — resolve `BASE` path, load `accounts.parquet` and `train_labels.parquet`.
3. **Feature Engineering** (Steps 1–11) — seven streaming passes over 400 M transaction rows producing ~206 features. Longest step (~1.5 h).
4. **Graph Feature Assembly** — build transaction graph (3.25 M edges), compute CV-encoded 1-hop / 2-hop / 3-hop contamination scores, ring detection, and branch collusion features.
5. **Adversarial Validation** (Step 8) — univariate AUC check + LightGBM adversarial model to detect and exclude red-herring features. Outputs `feat_clean`, `final_rh`, `adv_aucs`, `weights`.
6. **Ensemble Training** — train six base models (M1–M6) with 5-fold stratified CV, Borda rank aggregation, LightGBM meta-learner, and 5,000-sample Dirichlet blend search.
7. **Calibration & Thresholding** — isotonic regression on OOF predictions; F1-optimal threshold search.
8. **Temporal Window Detection** — per-account daily suspicious density scores; cumulative-fraction burst window (10–90% of mass) for temporal IoU.
9. **Visualisations (VIZ A–F)** — optional diagnostic plots saved to `/kaggle/working/`. Each cell is independently guarded and skips gracefully if prerequisite variables are missing.
10. **Submission Generation** — produces `submission.csv` with `account_id`, `is_mule`, `suspicious_start`, `suspicious_end`.

### Expected Output Files

| File | Description |
|---|---|
| `submission.csv` | Final predictions |
| `viz_A_*.png` | Transaction feature distributions |
| `viz_B_graph_features.png` | Graph feature distributions |
| `viz_C_adversarial_noise.png` | Adversarial validation & label noise |
| `viz_D_*.png` | Model performance & ensemble analysis |
| `viz_E_temporal_deep.png` | Temporal window analysis |
| `viz_06_final_summary.png` | Final dashboard with KPI cards |

### Reproducibility Notes

- All random seeds are fixed: `SEED = 42`, `SEED_ENSEMBLE = [42, 7, 13, 21, 99]`
- 5-fold stratified CV is seeded; fold assignments are deterministic
- Graph CV encoding is deterministic given the same fold splits
- Optuna is optional and skipped if not installed; the notebook uses hard-coded best parameters
- The blend search evaluates 5,000 fixed Dirichlet samples — results may differ slightly across NumPy versions due to RNG changes

---

## 3. Approach Description

### 3.1 Problem Framing

Binary classification over 64,062 test bank accounts (mule vs legitimate), given 96,091 labelled training accounts with 400 M transactions spanning five years. Mule prevalence in training data: 2.8%.

Evaluation is multi-criteria: **40% feature ingenuity**, **20% AUC-ROC + F1**, **15% red herring avoidance**, **15% temporal IoU**, **10% report quality**.

### 3.2 Feature Engineering — 206 Features

#### Behavioural Transaction Features (80+)

Seven streaming passes over the full transaction history extracted:

- **Structuring signals:** round-amount proportions at ₹1K / 5K / 10K / 25K / 50K thresholds
- **Pass-through velocity:** fast pass-through count (large debit within 24 h of large credit); same-day credit+debit pairing — *98.9% prevalence among predicted mules*
- **Fan-in / Fan-out asymmetry:** unique credit vs debit counterparty counts; `fan_in_out_ratio` ranked **#9** by importance
- **MCC-amount anomaly:** global MCC mean/std across all 400 M rows; per-account max z-score — **#1 feature overall** (importance 8,668)
- **Temporal signals:** night transactions (23:00–06:00), time-of-day entropy, burst ratio (txn rate last 30d vs long-run average), dormancy detection
- **Channel decomposition:** NEFT rate (**#7** importance), ATM rate (**#15**), UPI credit/debit rates
- **Post-mobile-change spike:** days from last mobile number update to last transaction — *11.1% mule prevalence*

#### Graph Features (39)

Transaction graph of **3.25 M edges**. All mule contamination scores use 5-fold CV to prevent target leakage.

- **1-hop CV contamination** (`g_mcm`, `g_mcs`, `g_wmsn`, `g_pexcl`): mean/max/sum mule rate across direct counterparties
- **2-hop contamination** (`g2_mcm`, `g2_mcx`, `g2_wmn`): mule rate propagated two hops with edge-weight discounting
- **CP reuse entropy** (`g_cp_norm_entropy`): **#2 feature** (importance 6,326). Mules concentrate traffic on 1–5 CPs; legitimate accounts use 20–100+
- **Ring detection** (`g_ring_max_shared`, `g_ring_n2plus`, `g_ring_n5plus`): accounts sharing ≥2/≥5 counterparties — *99.5% prevalence among predicted mules*
- **Branch collusion** (`bc_max/mean/sum_shared_cp`): shared counterparties within the same branch — *95.2% prevalence*

#### Static & Account Features (87)

- Account age (`age_d`, rank **#4**) and KYC recency (`kyc_d`, rank **#6**) — protected by `NEVER_EXCLUDE`
- Balance ratios (`bmd`, rank **#8**): monthly vs daily average balance
- IP diversity, geo spread, CI ratio from `transactions_additional`
- Interaction features: `g_mcm × pass-through ratio`; ring membership × direct contamination

### 3.3 Adversarial Validation & Red Herring Avoidance

Two-stage adversarial framework applied at every solution version:

- **Stage 1 — Univariate AUC:** features with train/test discriminative AUC > 0.62 are flagged
- **Stage 2 — LightGBM adversarial model:** top-5 importances added to exclusion list
- **Graph protection:** all `g_`, `g2_`, `g3_`, `bc_` prefixed features are exempt — they legitimately differ between train and test because contamination scores are computed from training labels via CV
- **`NEVER_EXCLUDE` list:** `age_d`, `kyc_d`, `rel_y`, `span`, `bmd` and all graph features are hard-protected after v3 wrongly excluded them, causing a −0.003 AUC regression

**Result:** 10 red herrings excluded out of 203 features; 193 clean features used for training.

### 3.4 Six-Model Ensemble

| Model | Algorithm | Feature Set | OOF AUC |
|---|---|---|---|
| M1 | LightGBM ×5 seeds | Clean (no red herrings) | 0.9519 |
| M2 | LightGBM DART | All 203 features | 0.9577 ← best single model |
| M3 | LightGBM GBDT | Graph + branch only | 0.9410 |
| M4 | XGBoost | All 203 features | 0.9479 |
| M5 | CatBoost | All 203 features | 0.9467 |
| M6 | MLP 256→128→64 | Clean features (scaled) | 0.9379 |
| **Borda** | Rank aggregation | All 6 models | **0.9564** |

Predictions are combined via **Borda rank aggregation**, then a **5,000-sample Dirichlet random blend search**. The random blend (OOF AUC 0.9539) outperformed the LightGBM meta-stacker (0.9507) and was selected as the final prediction.

### 3.5 Calibration & Threshold

- Isotonic regression on OOF predictions → calibrated OOF AUC **0.9562**
- F1-optimal decision threshold: **0.320**
- Label noise: 362 accounts (0.377%) assigned weight 0.05 via cross-validated agreement score

### 3.6 Temporal Window Detection

For each predicted mule, a suspicious activity window is estimated using:

- Per-account **daily suspicious density**: round amounts × weight + night txns × 1.5 + large credits × 1.5 + same-day pass-through × 2
- **Cumulative-fraction burst detection:** window spans the 10th–90th percentile of cumulative suspicious mass — finds the concentrated laundering period rather than full account lifespan
- Three tiers by mule score: ≥0.65 → 10–90%, ≥0.35 → 5–95%, ≥0.15 → 60-day rolling peak

2,280 of 1,948 predicted mules received usable windows. **Private leaderboard IoU: 0.6879.**

### 3.7 Final Results

| Metric | Value |
|---|---|
| Private AUC-ROC | **0.999833** |
| Private F1 Score | **0.971609** |
| Private Temporal IoU | **0.6879** |
| Predicted mules | 1,948 / 64,062 test accounts (3.0%) |
| Total runtime | ~10,634 s (~3 hours, CPU only) |
| Features used | 193 clean of 206 engineered |

---

*Team HxH — RBIH NFPC Phase 2 — Solution v4*
