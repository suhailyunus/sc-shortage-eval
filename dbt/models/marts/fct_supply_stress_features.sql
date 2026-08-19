-- Gold: final model-ready feature table.
-- Mirrors src/features.py's create_model_features(), feature for feature.
--
-- Leakage guard: rolling_mean_7 / rolling_std_7 use ROWS BETWEEN 7
-- PRECEDING AND 1 PRECEDING -- i.e. the 7 days strictly BEFORE the
-- current row, never including the current day's own sales. This is
-- the SQL equivalent of the pandas .shift(1).rolling(7) pattern that
-- was originally a leakage bug fixed early in this project (see
-- paper_draft/case_study.md, "The bug that was hiding in the
-- validation itself").
--
-- Location encoding uses a FIXED, known category list (all 10 stores,
-- all 3 states) via explicit CASE WHEN, rather than any form of
-- dynamically-inferred one-hot encoding. This sidesteps the exact bug
-- found during full-catalog validation, where chunking by store caused
-- one-hot encoding to silently produce zero columns for a single-
-- category chunk -- SQL sees the whole table at once, so that specific
-- failure mode can't occur here, but the fixed-category approach is
-- used anyway for parity with scripts/build_full_catalog_chunks.py's
-- create_model_features_fixed_location().

with target as (
    select * from {{ ref('int_stress_target') }}
),

windowed as (
    select
        *,

        lag(sales, 1) over (
            partition by item_id, store_id order by day_num
        ) as sales_lag_1,

        lag(sales, 7) over (
            partition by item_id, store_id order by day_num
        ) as sales_lag_7,

        avg(sales) over (
            partition by item_id, store_id order by day_num
            rows between 7 preceding and 1 preceding
        ) as rolling_mean_7,

        stddev(sales) over (
            partition by item_id, store_id order by day_num
            rows between 7 preceding and 1 preceding
        ) as rolling_std_7,

        lag(sell_price, 1) over (
            partition by item_id, store_id order by day_num
        ) as price_lag_1,

        case when weekday in ('Saturday', 'Sunday') then 1 else 0 end as is_weekend,
        case when event_name_1 is not null then 1 else 0 end as is_event_day

    from target
),

final as (
    select
        id,
        item_id,
        dept_id,
        cat_id,
        store_id,
        state_id,
        day_num,
        sales,
        stress_threshold,
        stress_event,

        sales_lag_1,
        sales_lag_7,
        rolling_mean_7,
        rolling_std_7,
        is_weekend,
        is_event_day,
        snap_ca,
        snap_tx,
        snap_wi,
        sell_price,
        (sell_price - price_lag_1) as price_change_1,

        -- Fixed-category location dummies (drop_first convention: CA
        -- and CA_1 are the reference categories, matching
        -- pl.to_dummies(drop_first=True)'s alphabetical-first-dropped
        -- behaviour used throughout the rest of this project).
        case when state_id = 'TX' then 1 else 0 end as state_id_tx,
        case when state_id = 'WI' then 1 else 0 end as state_id_wi,
        case when store_id = 'CA_2' then 1 else 0 end as store_id_ca_2,
        case when store_id = 'CA_3' then 1 else 0 end as store_id_ca_3,
        case when store_id = 'CA_4' then 1 else 0 end as store_id_ca_4,
        case when store_id = 'TX_1' then 1 else 0 end as store_id_tx_1,
        case when store_id = 'TX_2' then 1 else 0 end as store_id_tx_2,
        case when store_id = 'TX_3' then 1 else 0 end as store_id_tx_3,
        case when store_id = 'WI_1' then 1 else 0 end as store_id_wi_1,
        case when store_id = 'WI_2' then 1 else 0 end as store_id_wi_2,
        case when store_id = 'WI_3' then 1 else 0 end as store_id_wi_3

    from windowed
    -- Drop warmup rows with incomplete lag/rolling history, matching
    -- prepare_model_input()'s dropna(subset=feature_names) behaviour.
    where sales_lag_1 is not null
      and sales_lag_7 is not null
      and rolling_mean_7 is not null
      and rolling_std_7 is not null
      and price_change_1 is not null
)

select * from final
