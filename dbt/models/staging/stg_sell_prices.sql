-- Bronze: sell prices, typed and column-named consistently.

select
    store_id,
    item_id,
    cast(wm_yr_wk as integer) as wm_yr_wk,
    cast(sell_price as float) as sell_price
from {{ source('raw', 'sell_prices') }}
