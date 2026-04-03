-- Setup script: creates Snowflake objects for the PySpark migration showcase.
--
-- Before running, replace the placeholders:
--   MY_DATABASE   -> your Snowflake database
--   MY_WAREHOUSE  -> your warehouse name
--   MY_ROLE       -> the role that should own/access the tables
--   @...DATA/     -> your internal stage path where CSVs are uploaded
--
-- Usage:
--   snow sql -f setup.sql -c <your_connection_name>

USE DATABASE MY_DATABASE;
CREATE SCHEMA IF NOT EXISTS RISK_SCORING_MODEL;
USE SCHEMA RISK_SCORING_MODEL;

CREATE FILE FORMAT IF NOT EXISTS CSV_FORMAT
    TYPE = 'CSV'
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"';

-- Optional: resize warehouse for faster bulk load
-- ALTER WAREHOUSE MY_WAREHOUSE SET WAREHOUSE_SIZE = 'XXLARGE';

CREATE OR REPLACE TABLE RAW_TRANSACTIONS AS
SELECT $1 AS transaction_id, $2 AS customer_id, $3 AS card_number, $4 AS timestamp,
       $5 AS merchant_category, $6 AS merchant_type, $7 AS merchant, $8 AS amount,
       $9 AS currency, $10 AS country, $11 AS city, $12 AS city_size, $13 AS card_type,
       $14 AS card_present, $15 AS device, $16 AS channel, $17 AS device_fingerprint,
       $18 AS ip_address, $19 AS distance_from_home, $20 AS high_risk_merchant,
       $21 AS transaction_hour, $22 AS weekend_transaction, $23 AS velocity_last_hour,
       $24 AS is_fraud
FROM @DATA/synthetic_fraud_data.csv
    (FILE_FORMAT => 'CSV_FORMAT');

CREATE OR REPLACE TABLE RAW_TRANSACTIONS_MEDIUM AS
SELECT $1 AS transaction_id, $2 AS customer_id, $3 AS card_number, $4 AS timestamp,
       $5 AS merchant_category, $6 AS merchant_type, $7 AS merchant, $8 AS amount,
       $9 AS currency, $10 AS country, $11 AS city, $12 AS city_size, $13 AS card_type,
       $14 AS card_present, $15 AS device, $16 AS channel, $17 AS device_fingerprint,
       $18 AS ip_address, $19 AS distance_from_home, $20 AS high_risk_merchant,
       $21 AS transaction_hour, $22 AS weekend_transaction, $23 AS velocity_last_hour,
       $24 AS is_fraud
FROM @DATA/synthetic_fraud_data_medium.csv
    (FILE_FORMAT => 'CSV_FORMAT');

CREATE OR REPLACE TABLE RAW_TRANSACTIONS_500K AS
SELECT $1 AS transaction_id, $2 AS customer_id, $3 AS card_number, $4 AS timestamp,
       $5 AS merchant_category, $6 AS merchant_type, $7 AS merchant, $8 AS amount,
       $9 AS currency, $10 AS country, $11 AS city, $12 AS city_size, $13 AS card_type,
       $14 AS card_present, $15 AS device, $16 AS channel, $17 AS device_fingerprint,
       $18 AS ip_address, $19 AS distance_from_home, $20 AS high_risk_merchant,
       $21 AS transaction_hour, $22 AS weekend_transaction, $23 AS velocity_last_hour,
       $24 AS is_fraud
FROM @DATA/synthetic_fraud_data_500k.csv
    (FILE_FORMAT => 'CSV_FORMAT');

CREATE OR REPLACE TABLE RAW_TRANSACTIONS_1MIO AS
SELECT $1 AS transaction_id, $2 AS customer_id, $3 AS card_number, $4 AS timestamp,
       $5 AS merchant_category, $6 AS merchant_type, $7 AS merchant, $8 AS amount,
       $9 AS currency, $10 AS country, $11 AS city, $12 AS city_size, $13 AS card_type,
       $14 AS card_present, $15 AS device, $16 AS channel, $17 AS device_fingerprint,
       $18 AS ip_address, $19 AS distance_from_home, $20 AS high_risk_merchant,
       $21 AS transaction_hour, $22 AS weekend_transaction, $23 AS velocity_last_hour,
       $24 AS is_fraud
FROM @DATA/synthetic_fraud_data_1mio.csv
    (FILE_FORMAT => 'CSV_FORMAT');

CREATE OR REPLACE TABLE RAW_TRANSACTIONS_SMALL AS
SELECT $1 AS transaction_id, $2 AS customer_id, $3 AS card_number, $4 AS timestamp,
       $5 AS merchant_category, $6 AS merchant_type, $7 AS merchant, $8 AS amount,
       $9 AS currency, $10 AS country, $11 AS city, $12 AS city_size, $13 AS card_type,
       $14 AS card_present, $15 AS device, $16 AS channel, $17 AS device_fingerprint,
       $18 AS ip_address, $19 AS distance_from_home, $20 AS high_risk_merchant,
       $21 AS transaction_hour, $22 AS weekend_transaction, $23 AS velocity_last_hour,
       $24 AS is_fraud
FROM @DATA/synthetic_fraud_data_small.csv
    (FILE_FORMAT => 'CSV_FORMAT');

-- Optional: resize warehouse back
-- ALTER WAREHOUSE MY_WAREHOUSE SET WAREHOUSE_SIZE = 'XSMALL';

-- Grant access to your role (adjust MY_ROLE as needed)
-- GRANT SELECT ON ALL TABLES IN SCHEMA RISK_SCORING_MODEL TO ROLE MY_ROLE;
-- GRANT SELECT ON FUTURE TABLES IN SCHEMA RISK_SCORING_MODEL TO ROLE MY_ROLE;
-- GRANT CREATE TABLE ON SCHEMA RISK_SCORING_MODEL TO ROLE MY_ROLE;
