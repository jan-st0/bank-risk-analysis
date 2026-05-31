# Credit Risk Simulation Framework

This repository contains a credit risk simulation engine developed to explore the intersection of software architecture, applied machine learning, and numerical computation in Python. The framework processes historical loan performance data, estimates default probabilities using gradient boosting, and models portfolio risk through Monte Carlo simulations. The system is structured using the Ports and Adapters paradigm to systematically decouple domain logic from external libraries and data persistence layers.

## Data Source
The framework is built to process the Fannie Mae Single-Family Loan Performance Dataset. The pipeline is specifically configured to analyze the two most recent available quarters (e.g., 2025Q3 for model training and 2025Q4 for forward-looking simulation). This dataset provides realistic, loan-level financial attributes including Unpaid Principal Balance (UPB), Loan-to-Value (LTV) ratios, borrower credit scores, and monthly delinquency statuses.
For now the etl requires 2 csv files with Single Family dataset format.
   - csv for training
   - csv for simulation

You can download 2025Q3 and Q4 from Frannie Mae official site or [My drive](https://drive.google.com/drive/folders/1bboB3lDKmR_dQ2-Bomi-LVMhcEeBHZZ5?usp=sharing)
Make sure you change `config/secrets/conn_sample.yaml` to `conn.yaml` with approprate connection string. And `config/app/pipeline.yaml` to include all your csv files.

## System Architecture
The codebase separates operations into three distinct modules:

* **ETL & Persistence:** Extracts raw CSV records, applies temporal aggregation and feature cleansing, and loads the standardized data into a PostgreSQL database.
* **Machine Learning Pipeline:** Trains an XGBoost classifier to predict the Probability of Default (PD). The pipeline applies Isotonic Calibration via cross-validation to ensure outputs represent empirical probabilities.
* **Simulation Engine:** Implements a multi-factor structural default model.

## Installation
The framework requires Python 3.10+ and a functional PostgreSQL database instance.
Configuration variables and schema mappings are managed via Hydra in the `config/` directory.
Put your db connection string in `config/secrets/conn.yaml`

```bash
git clone https://github.com/jan-st0/bank-risk-analysis
cd bank-risk-analysis
pip install -r requirements.txt
```


## How To Run The Project

Critical Execution Requirement: All commands must be executed from the repository root with the PYTHONPATH environment variable directed to the src directory to ensure proper internal module resolution.

### 1. Processing data

Initializes the database schema and processes the raw analytical files.

```bash
PYTHONPATH=src python main.py etl
```
### 2. Model Training & Probability Calibration

Executes the classification pipeline, applies calibration, and serializes the model artifacts.

```bash
PYTHONPATH=src python main.py train
```

### 3. Portfolio Risk Simulation

Executes the Monte Carlo integrations across parallel worker processes to compute the loss distribution.

```bash
PYTHONPATH=src python main.py simulate --simulations 1000 --workers 4
```
### 4. Run Pytests
```bash
PYTHONPATH=src pytest src/tests -v
```
## Mathematical Core & Formulas

The framework utilizes a highly optimized, vectorized simulation approach to calculate portfolio risk based strictly on machine learning outputs, bypassing theoretical macro-factor assumptions in favor of computational efficiency.

### 1. Independent Bernoulli Trials
Each loan defaults independently based purely on its calibrated Probability of Default (PD). 

Mathematically, the default indicator $D_i$ for a single Monte Carlo path is a random variable drawn from a Bernoulli distribution:

$$D_i \sim \text{Bernoulli}(\text{PD}_i)$$

In the execution engine, this is fully vectorized by comparing a matrix of continuous uniform random variables $\mathcal{U}(0,1)$ directly against the probability vector to extract the boolean default triggers across millions of simulated paths.

### 2. Deterministic Loss Given Default (LGD)
The framework maps the Loss Given Default ($\text{LGD}_i$) using a deterministic, piecewise step function based on the loan's original Loan-to-Value (LTV) ratio. This models the expected recovery friction without introducing arbitrary uniform noise:

* LTV $\leq$ 60%: LGD = 0.20
* 60% < LTV $\leq$ 80%: LGD = 0.35
* 80% < LTV $\leq$ 90%: LGD = 0.50
* LTV > 90%: LGD = 0.70

### 3. Portfolio Loss Accumulation
The total accumulated portfolio loss $L$ for a given macroeconomic scenario path is the summation of individual loan exposures across all active default indicators:

$$L = \sum_{i=1}^{N} \text{UPB}_i \times \text{LGD}_i \times D_i$$

*(where $\text{UPB}_i$ is the Unpaid Principal Balance).*
