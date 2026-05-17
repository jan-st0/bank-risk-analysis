CREATE TABLE IF NOT EXISTS sectors (
    sector_id INT PRIMARY KEY,
    property_type VARCHAR(2) UNIQUE,
    asset_correlation NUMERIC,
    beta_economy NUMERIC
);

CREATE TABLE IF NOT EXISTS client_loan_data (
    loan_identifier BIGINT PRIMARY KEY,
    current_actual_upb NUMERIC,
    loan_report_date VARCHAR(6),
    current_loan_delinquency_status VARCHAR(2),
    zero_balance_code VARCHAR(3),
    borrower_credit_score INT,
    ltv_ratio NUMERIC,
    debt_to_income NUMERIC,
    occupancy_status VARCHAR(1),
    loan_age VARCHAR(6),
    sector_id INT,
    FOREIGN KEY (sector_id) REFERENCES sectors(sector_id)
);
