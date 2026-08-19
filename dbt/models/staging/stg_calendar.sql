-- Bronze: calendar, typed and column-named consistently.

select
    d,
    cast(date as date) as calendar_date,
    cast(wm_yr_wk as integer) as wm_yr_wk,
    weekday,
    cast(wday as integer) as wday,
    cast(month as integer) as month,
    cast(year as integer) as year,
    event_name_1,
    event_type_1,
    event_name_2,
    event_type_2,
    cast(snap_ca as integer) as snap_ca,
    cast(snap_tx as integer) as snap_tx,
    cast(snap_wi as integer) as snap_wi
from {{ source('raw', 'calendar') }}
