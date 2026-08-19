-- Silver: sales joined with calendar and price context.
-- Mirrors src/preprocess.py's merge_calendar() + merge_prices(), same
-- join keys and same left-join semantics (a sale should never be
-- dropped for lacking a price or calendar row).

with sales as (
    select * from {{ ref('stg_sales_long') }}
),

calendar as (
    select * from {{ ref('stg_calendar') }}
),

prices as (
    select * from {{ ref('stg_sell_prices') }}
),

joined as (
    select
        sales.id,
        sales.item_id,
        sales.dept_id,
        sales.cat_id,
        sales.store_id,
        sales.state_id,
        sales.day,
        sales.sales,
        sales.day_num,
        calendar.calendar_date,
        calendar.wm_yr_wk,
        calendar.weekday,
        calendar.wday,
        calendar.month,
        calendar.year,
        calendar.event_name_1,
        calendar.event_type_1,
        calendar.event_name_2,
        calendar.event_type_2,
        calendar.snap_ca,
        calendar.snap_tx,
        calendar.snap_wi,
        prices.sell_price
    from sales
    left join calendar
        on sales.day = calendar.d
    left join prices
        on sales.store_id = prices.store_id
        and sales.item_id = prices.item_id
        and calendar.wm_yr_wk = prices.wm_yr_wk
)

select * from joined
