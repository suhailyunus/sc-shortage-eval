-- Bronze: sales, typed and column-named consistently. No business logic
-- here -- just "this table can be trusted to have the right types."

select
    id,
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    day,
    cast(sales as integer) as sales,
    cast(day_num as integer) as day_num
from {{ source('raw', 'sales_long') }}
