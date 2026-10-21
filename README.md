# FED4FED — Privacy-Preserving Federated Learning & Statistical Fairness Evaluation Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.56.0-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/pytest-26%20passed-brightgreen.svg)]()

FED4FED is a research-grade platform implementing the privacy-preserving federated statistical testing framework introduced in IEEE Transactions on Dependable and Secure Computing. It evaluates cross-client performance fairness of global federated learning models without leaking decentralized private client data.

---

## 1. Architectural Overview

The application couples two interconnected phases:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PHASE 1: FEDERATED LEARNING                              │
│  [Dataset] ──> [Partition Clients] ──> [Local Training] ──> [FedAvg] ──> [Export Model]│
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Artifact Handoff (Model + Private Test Sets)
┌───────────────────────────────────────────▼────────────────────────────────────────────┐
│                       PHASE 2: FED4FED STATISTICAL EVALUATION                          │
│  [Load Output] ──> [Bootstrap Resampling] ──> [Alg 1 (T1) & Alg 2 (F)] ──> [Alg 3 CI] │
│                                     │                                                  │
│                                     └───> [Final Research Report PDF]                  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1 — Federated Learning
- **Datasets**: Unified dataset registry supporting MNIST, Fashion-MNIST, CIFAR-10, and SST.
- **Partitioning**:
  - Fair Scenario: Uniform random IID allocation across clients.
  - Unfair Scenario: Clustered label distribution (MNIST), category proportion skew (Fashion-MNIST), Dirichlet label distribution with $\alpha \in \{0.001, 0.01, 0.1, 10.0\}$ (CIFAR-10), and label skew (SST).
- **Local Training & FedAvg**: Layer-by-layer parameter aggregation weighted by participating client training sample counts:
  $$\mathbf{w}_{\text{global}}^{(t+1)} = \sum_{k \in S_t} \frac{n_k}{\sum_{j \in S_t} n_j} \mathbf{w}_k^{(t+1)}$$
- **Export Handoff**: Verified checkpoint (`global_model.pt`), per-client private test sets (`client_test_sets/*.pt`), manifest, and configuration.

### Phase 2 — FED4FED Statistical Evaluation
- **Decentralized Bootstrap**: Computes non-parametric bootstrap replications ($B$) locally on client test observations to estimate client sample means $\bar{X}_i$ and variances $S_i^2$.
- **Privacy Preservation**: Clients only exchange encrypted/masked statistical summary parameters $(\bar{X}_i, S_i^2, n_i)$; raw data and individual predictions remain strictly private.
- **Algorithm 1 (Graybill-Deal / Chi-Square $T_1$ Test)**:
  $$T_1 = \sum_{i=1}^K \frac{(\bar{X}_i - \hat{\mu}_{GD})^2}{S_i^2} \sim \chi^2(K - 1), \quad \hat{\mu}_{GD} = \frac{\sum w_i \bar{X}_i}{\sum w_i}$$
- **Algorithm 2 (Welch-Type $F$ Test under Heteroscedasticity)**:
  $$F_W = \frac{\frac{1}{K-1} \sum_{i=1}^K w_i (\bar{X}_i - \bar{X}^*)^2}{1 + \frac{2(K-2)}{K^2-1}\Lambda} \sim F(\nu_1, \nu_2)$$
- **Algorithm 3 (Confidence Interval under Decision Rules)**:
  Pooled Graybill-Deal confidence interval is computed **only when the fairness test condition is satisfied** ($p \ge \alpha$). If disparity is statistically detected ($p < \alpha$), a single common performance interval is declared unavailable and individual client intervals are reported.
- **Reporting**: ReportLab automated PDF report generation, CSV tables, and JSON outputs.

---

## 2. Directory Structure

```
fed4fed-project/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── config.yaml
├── app.py
├── test_model_selection.py          # NEW: Verify model selection
│
├── configs/
│   ├── mnist.yaml
│   ├── fashion_mnist.yaml
│   ├── cifar10.yaml                 # Updated training defaults
│   └── sst.yaml
│
├── pages/
│   ├── 0_Home_Overview.py
│   │
│   ├── phase1_fl/
│   │   ├── 1_Select_Dataset.py
│   │   ├── 2_Partition_Clients.py
│   │   ├── 3_FL_Training.py
│   │   └── 4_Export_Model.py
│   │
│   └── phase2_fed4fed/
│       ├── 5_Load_Phase1_Output.py
│       ├── 6_Bootstrap.py
│       ├── 7_Fairness_Test.py
│       ├── 8_Confidence_Interval.py
│       └── 9_Final_Report.py
│
├── ui/
│   ├── theme.py
│   ├── navigation.py
│   ├── state.py
│   ├── guards.py
│   ├── sidebar.py
│   ├── components.py
│   ├── charts_phase1.py
│   └── charts_phase2.py
│
├── src/
│   │
│   ├── data/
│   │   ├── registry.py
│   │   ├── preprocessing.py
│   │   ├── mnist.py
│   │   ├── fashion_mnist.py
│   │   ├── cifar10.py
│   │   ├── sst.py
│   │   └── partition/
│   │       ├── uniform.py
│   │       ├── clustered.py
│   │       ├── category.py
│   │       ├── dirichlet.py
│   │       └── label_skew.py
│   │
│   ├── models/
│   │   ├── registry.py              # UPDATED: Selects the right model
│   │   ├── mlp_mnist.py              # Unchanged
│   │   ├── cnn.py                    # UPDATED: Contains two CNNs
│   │   │                              #   FedCNN
│   │   │                              #   CIFAR10CNN
│   │   └── lstm_attention.py         # Unchanged
│   │
│   ├── phase1_fl/
│   │   ├── client.py
│   │   ├── server.py
│   │   ├── fedavg.py                 # Checked; no change required
│   │   ├── simulation.py
│   │   └── export.py
│   │
│   ├── phase2_fed4fed/
│   │   ├── bootstrap.py
│   │   ├── normality_check.py
│   │   ├── stats_client.py
│   │   ├── stats_server.py
│   │   ├── encryption.py
│   │   ├── alg1_graybill_deal.py
│   │   ├── alg2_welch.py
│   │   ├── alg3_confidence.py
│   │   └── fairness.py
│   │
│   ├── metrics/
│   │   └── metrics.py
│   │
│   ├── pipeline/
│   │   ├── run_phase1.py
│   │   └── run_phase2.py
│   │
│   ├── reports/
│   │   ├── pdf_report.py
│   │   └── result_tables.py
│   │
│   └── utils/
│       ├── config.py
│       ├── seed.py
│       ├── logger.py
│       └── artifact_manager.py
│
├── outputs/
│   ├── phase1/
│   └── phase2/
│
├── scripts/
│   ├── download_data.py
│   ├── run_phase1.py
│   ├── run_phase2.py
│   ├── run_all.py
│   └── run_alpha_sweep.py
│
├── notebooks/
│   ├── explore_datasets.ipynb
│   └── validate_statistics.ipynb
│
└── tests/
    ├── test_datasets.py
    ├── test_partitions.py
    ├── test_models.py
    ├── test_fedavg.py
    ├── test_export.py
    ├── test_bootstrap.py
    ├── test_fairness_tests.py
    └── test_reports.py
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.13 on Windows, Linux, or macOS.

### Install Dependencies
```bash
cd fed4fed-project
pip install -r requirements.txt
```

### Run Automated Test Suite
```bash
python -m pytest -v
```
All 26 automated unit tests cover mathematical formulas, FedAvg aggregation, model forward passes, partition completeness, bootstrap reproducibility, and PDF generation.

---

## 4. Running the Research Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The application will launch on `http://localhost:8501` featuring:
1. **Home Overview**: Live status badges, workflow map, recent experiment list, and executive fairness cards.
2. **Phase 1 (FL)**: Dataset loading, non-IID client partition analysis, real-time round training curves, and artifact export.
3. **Phase 2 (FED4FED)**: Handoff loading, client-side bootstrap distributions (violin/histogram/Q-Q), Algorithm 1 & 2 p-values with significance threshold $\alpha$, Algorithm 3 decision rules, forest plots, and downloadable PDF reports.

---

## 5. Command-Line Batch Execution

### Run Phase 1 FL Experiment
```bash
# Fair MNIST run with 50 clients over 5 rounds
python scripts/run_phase1.py --dataset MNIST --scenario Fair --clients 50 --rounds 5

# Unfair CIFAR-10 run with Dirichlet alpha=0.1
python scripts/run_phase1.py --dataset CIFAR-10 --scenario Unfair --clients 50 --rounds 5 --alpha 0.1
```

### Run Phase 2 Statistical Evaluation
```bash
python scripts/run_phase2.py --exp_dir outputs/phase1/mnist/fair/<experiment_id> --resamples 200 --alpha 0.05
```

### Run Dirichlet Alpha Sweep
```bash
python scripts/run_alpha_sweep.py
```

---

## 6. Scientific Methodology & Decision Rules

1. **Null Hypothesis ($H_0$)**: $\mu_1 = \mu_2 = \dots = \mu_K = \mu$. The global model exhibits equal expected performance across all decentralized client distributions.
2. **Alternative Hypothesis ($H_1$)**: There exist $i \ne j$ such that $\mu_i \ne \mu_j$.
3. **Significance Level ($\alpha$)**: Default 0.05. If $p < \alpha$, we reject $H_0$ (statistical evidence of cross-client disparity).
4. **Algorithm 3 Decision Rule**: The pooled confidence interval for common performance mean is strictly enabled **only if $H_0$ is not rejected**. If $H_0$ is rejected, the common mean assumption is statistically invalid; individual client intervals are displayed instead.
