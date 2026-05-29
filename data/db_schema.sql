DROP TABLE IF EXISTS simulation_loan_data CASCADE;
DROP TABLE IF EXISTS training_loan_data CASCADE;
DROP TABLE IF EXISTS sectors CASCADE;

CREATE TABLE IF NOT EXISTS sectors (
    sector_id INT PRIMARY KEY,
    property_type VARCHAR(2) UNIQUE,
    asset_correlation NUMERIC,
    beta_economy NUMERIC
);

CREATE TABLE IF NOT EXISTS simulation_loan_data (
    loan_identifier BIGINT PRIMARY KEY,
    current_actual_upb DOUBLE PRECISION,
    loan_report_date INT,
    current_loan_delinquency_status INT,
    zero_balance_code INT,
    borrower_credit_score INT,
    ltv_ratio INT,
    debt_to_income INT,
    occupancy_status VARCHAR(1),
    loan_age INT,
    sector_id INT,
    FOREIGN KEY (sector_id) REFERENCES sectors(sector_id)
);

CREATE TABLE IF NOT EXISTS training_loan_data (
    loan_identifier BIGINT PRIMARY KEY,
    current_actual_upb DOUBLE PRECISION,
    loan_report_date INT,
    current_loan_delinquency_status INT,
    zero_balance_code INT,
    borrower_credit_score INT,
    ltv_ratio INT,
    debt_to_income INT,
    occupancy_status VARCHAR(1),
    loan_age INT,
    sector_id INT,
    FOREIGN KEY (sector_id) REFERENCES sectors(sector_id)
);