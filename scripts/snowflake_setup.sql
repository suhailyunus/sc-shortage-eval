-- Snowflake account bootstrap for supply-chain-stress-prediction.
--
-- Run this once, as ACCOUNTADMIN (or SYSADMIN + SECURITYADMIN), right
-- after creating the trial account. Creates a dedicated, cost-capped
-- warehouse and a role scoped only to this project's own database --
-- not broad ACCOUNTADMIN access for day-to-day dbt runs.
--
-- Usage:
--   snowsql -a <account_identifier> -u <your_admin_user> -f scripts/snowflake_setup.sql
-- or paste into a Snowsight worksheet and run top to bottom.

-- 1. Warehouse: XSMALL, auto-suspends after 60s idle, auto-resumes on
--    query. This is the single biggest cost lever on a trial account --
--    an always-on warehouse burns credits even when nothing is running.
CREATE WAREHOUSE IF NOT EXISTS SUPPLY_CHAIN_WH
    WAREHOUSE_SIZE = 'XSMALL'
    AUTO_SUSPEND = 60
    AUTO_RESUME = TRUE
    INITIALLY_SUSPENDED = TRUE
    COMMENT = 'supply-chain-stress-prediction: dbt + ad hoc query warehouse';

-- 2. Database and schema. RAW holds untransformed loads from the M5
--    CSVs; dbt models build STAGING/MARTS schemas on top of this one.
CREATE DATABASE IF NOT EXISTS SUPPLY_CHAIN_DB
    COMMENT = 'supply-chain-stress-prediction';

CREATE SCHEMA IF NOT EXISTS SUPPLY_CHAIN_DB.RAW
    COMMENT = 'Untransformed M5 source tables, loaded via COPY INTO';

-- 3. Role: scoped only to this project's warehouse + database, not a
--   broad account role. This is the role dbt will actually run as.
CREATE ROLE IF NOT EXISTS SUPPLY_CHAIN_ROLE
    COMMENT = 'Least-privilege role for supply-chain-stress-prediction dbt runs';

GRANT USAGE ON WAREHOUSE SUPPLY_CHAIN_WH TO ROLE SUPPLY_CHAIN_ROLE;
GRANT OPERATE ON WAREHOUSE SUPPLY_CHAIN_WH TO ROLE SUPPLY_CHAIN_ROLE;

GRANT USAGE ON DATABASE SUPPLY_CHAIN_DB TO ROLE SUPPLY_CHAIN_ROLE;
GRANT CREATE SCHEMA ON DATABASE SUPPLY_CHAIN_DB TO ROLE SUPPLY_CHAIN_ROLE;

GRANT USAGE ON SCHEMA SUPPLY_CHAIN_DB.RAW TO ROLE SUPPLY_CHAIN_ROLE;
GRANT CREATE TABLE ON SCHEMA SUPPLY_CHAIN_DB.RAW TO ROLE SUPPLY_CHAIN_ROLE;
GRANT CREATE STAGE ON SCHEMA SUPPLY_CHAIN_DB.RAW TO ROLE SUPPLY_CHAIN_ROLE;
GRANT CREATE FILE FORMAT ON SCHEMA SUPPLY_CHAIN_DB.RAW TO ROLE SUPPLY_CHAIN_ROLE;

-- Future schemas dbt creates (STAGING, MARTS, etc.) inherit these grants
-- automatically via this future-grant rule, so you don't have to re-run
-- GRANT statements every time dbt materializes a new schema.
GRANT ALL PRIVILEGES ON FUTURE SCHEMAS IN DATABASE SUPPLY_CHAIN_DB TO ROLE SUPPLY_CHAIN_ROLE;
GRANT ALL PRIVILEGES ON FUTURE TABLES IN DATABASE SUPPLY_CHAIN_DB TO ROLE SUPPLY_CHAIN_ROLE;
GRANT ALL PRIVILEGES ON FUTURE VIEWS IN DATABASE SUPPLY_CHAIN_DB TO ROLE SUPPLY_CHAIN_ROLE;

-- 4. Dedicated service user for dbt -- not your personal login. Replace
--    the password below before running, then rotate it immediately
--    after (this file will end up in git history even though it's
--    gitignored going forward, so treat any password typed here as
--    already-compromised once you've run it once).
CREATE USER IF NOT EXISTS SUPPLY_CHAIN_DBT_USER
    PASSWORD = 'MySupplyChainProject8593'
    DEFAULT_ROLE = SUPPLY_CHAIN_ROLE
    DEFAULT_WAREHOUSE = SUPPLY_CHAIN_WH
    DEFAULT_NAMESPACE = 'SUPPLY_CHAIN_DB.RAW'
    MUST_CHANGE_PASSWORD = FALSE
    COMMENT = 'Service account for dbt runs -- not a personal login';

GRANT ROLE SUPPLY_CHAIN_ROLE TO USER SUPPLY_CHAIN_DBT_USER;

-- Also grant the role to your own admin user so you can inspect/debug
-- from Snowsight without switching accounts. Replace with your actual
-- Snowflake username.
-- GRANT ROLE SUPPLY_CHAIN_ROLE TO USER <your_admin_username>;

-- 5. Verify: should show SUPPLY_CHAIN_WH, SUPPLY_CHAIN_DB, and the RAW
--    schema all present.
SHOW WAREHOUSES LIKE 'SUPPLY_CHAIN_WH';
SHOW DATABASES LIKE 'SUPPLY_CHAIN_DB';
SHOW SCHEMAS IN DATABASE SUPPLY_CHAIN_DB;
