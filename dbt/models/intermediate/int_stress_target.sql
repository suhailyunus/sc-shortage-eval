-- Silver: stress-event target.
--
-- Mirrors src/preprocess.py's add_stress_target(), with the CORRECTED
-- grouping: (item_id, store_id), NOT item_id alone. Grouping by item_id
-- alone pools sales across all 10 stores per item, which lets
-- high-volume stores cross the pooled threshold more often simply by
-- selling more -- a volume confound, not genuine demand stress.
-- Verified during the validation-at-scale work: store-rate/volume
-- correlation dropped from Pearson r=0.85 (item-only grouping) to
-- r=0.03 (item+store grouping). See paper_draft/case_study.md.
--
-- Leakage guard: the threshold is computed ONLY from day_num <=
-- {{ var('split_day') }} (the training period), then applied to every
-- row for that (item_id, store_id) pair -- including holdout rows.
-- Estimating the threshold over the full history (including holdout)
-- would leak future sales into a label applied to the training period.

with base as (
    select * from {{ ref('int_analytical_base') }}
),

train_period as (
    select * from base
    where day_num <= {{ var('split_day') }}
),

-- Per-(item, store) threshold from train-period history only.
group_thresholds as (
    select
        item_id,
        store_id,
        percentile_cont({{ var('stress_quantile') }})
            within group (order by sales) as stress_threshold
    from train_period
    group by item_id, store_id
),

-- Fallback for any (item, store) pair with no train-period history
-- (shouldn't occur in practice on the full M5 catalog, since every
-- series has data from day 1, but kept for parity with the Python
-- implementation's fallback behaviour).
global_fallback as (
    select
        percentile_cont({{ var('stress_quantile') }})
            within group (order by sales) as fallback_threshold
    from train_period
)

select
    base.*,
    coalesce(group_thresholds.stress_threshold, global_fallback.fallback_threshold) as stress_threshold,
    case
        when base.sales > coalesce(group_thresholds.stress_threshold, global_fallback.fallback_threshold)
        then 1
        else 0
    end as stress_event
from base
left join group_thresholds
    on base.item_id = group_thresholds.item_id
    and base.store_id = group_thresholds.store_id
cross join global_fallback
