# dbt / Snowflake pipeline (Phase 3)

This is a second, parallel implementation of the same feature-engineering
logic that lives in `src/preprocess.py` and `src/features.py` — same
target definition, same leakage guards, same features — but running as
SQL inside Snowflake instead of pandas on a laptop. Nothing about the
*model* changes; this is a data-engineering exercise in translating an
already-correct pipeline into a warehouse-native one.

## Why the layers are split this way

- **Python loader** (`scripts/load_raw_to_snowflake.py`) reshapes the M5
  sales CSV from wide (`d_1`...`d_1913` columns) to long format using
  the exact same `reshape_sales_long()` already tested in
  `tests/test_leakage.py`, then pushes three raw tables into Snowflake.
  This one step stays in Python deliberately — unpivoting ~1,900 columns
  is awkward in raw SQL (Snowflake's `UNPIVOT` wants an explicit column
  list), while pandas already does it correctly.

- **Bronze** (`models/staging/`): typed, renamed views over the raw
  tables. No business logic — just "these types can be trusted."

- **Silver** (`models/intermediate/`): the actual transformation logic.
  `int_analytical_base.sql` joins sales with calendar and price context
  (mirrors `merge_calendar`/`merge_prices`). `int_stress_target.sql`
  computes the stress-event target — **grouped by `(item_id, store_id)`,
  the corrected definition, not the flawed `item_id`-only pooling** — with
  the threshold computed only from train-period history (`day_num <=
  {{ var('split_day') }}`), same leakage guard as the Python version.

- **Gold** (`models/marts/`): `fct_supply_stress_features.sql`, the final
  model-ready table. Lag/rolling features use `ROWS BETWEEN 7 PRECEDING
  AND 1 PRECEDING` — the SQL equivalent of `.shift(1).rolling(7)` — so
  today's sales can never leak into today's own rolling average, same
  guard as the original leakage bug fix. Location is one-hot encoded
  against a **fixed, known category list** (all 10 stores, all 3
  states), not inferred from what's present in the data — the same
  fix applied in `scripts/build_full_catalog_chunks.py` after the
  silent-zero-columns bug was found there.

## split_day is a hardcoded variable, not computed in SQL

`dbt_project.yml` sets `split_day: 1531`. This is intentional, not a
shortcut: every M5 series shares the identical day range (1..1913), so
the 80th-percentile split day is the same regardless of how many
items/stores are loaded — verified to match `find_split_day()`'s output
exactly during the validation-at-scale work. If the underlying data ever
changes, recompute it with `src/preprocess.find_split_day()` and update
this variable.

## Running it

1. Make sure `.env` is filled in (see repo root `.env.example`) and
   you've run `scripts/snowflake_setup.sql` against your Snowflake
   trial account.

2. Install dependencies:
   ```bash
   pip install -r requirements-dbt.txt
   ```

3. Load the raw data into Snowflake (start small while testing):
   ```bash
   python scripts/load_raw_to_snowflake.py --max-items 100
   ```
   Drop `--max-items` once you're ready to load the full catalog.

4. Set up dbt's connection profile (one-time, outside this repo):
   ```bash
   mkdir -p ~/.dbt
   cp dbt/profiles.yml.example ~/.dbt/profiles.yml
   ```
   dbt reads Snowflake credentials from the same `.env` variables via
   `env_var()` — make sure they're exported into your shell environment
   before running dbt (e.g. `export $(cat .env | xargs)` on Mac/Linux,
   or use a tool like `direnv`).

5. Install the dbt package dependencies (dbt_utils, used by the
   uniqueness test):
   ```bash
   cd dbt
   dbt deps
   ```

6. Run it:
   ```bash
   dbt run
   dbt test
   ```

7. Verify in Snowsight:
   ```sql
   SELECT COUNT(*) FROM SUPPLY_CHAIN_DB.MARTS.FCT_SUPPLY_STRESS_FEATURES;
   SELECT AVG(stress_event) FROM SUPPLY_CHAIN_DB.MARTS.FCT_SUPPLY_STRESS_FEATURES;
   ```
   That second query is a quick sanity check — with the corrected
   target definition, this should land somewhere around 6-9%, not
   spread widely by store (see `paper_draft/case_study.md` for why that
   number matters).

## What I could and couldn't verify before handing this off

I don't have network access to Snowflake's package hub or to any real
Snowflake account from my end, so I could not run `dbt deps`, `dbt run`,
or `dbt test` against live infrastructure. What I did verify:

- `dbt parse` succeeds cleanly against this exact project (6 models, 3
  sources, 3 data tests, no errors or warnings).
- The Python loader's reshape step was tested against your actual M5
  CSV data and produces the expected output shape.
- The `snowflake-connector-python` and `python-dotenv` imports resolve
  correctly.

What's unverified and will surface only when you actually run this: the
live Snowflake connection itself, whether `write_pandas` handles the
full 58M-row long-format table within Snowflake's default upload limits
(start with `--max-items 100` first for exactly this reason), and
whether the compiled SQL executes without a Snowflake-specific syntax
issue I couldn't catch via `parse` alone (e.g. `PERCENTILE_CONT` syntax
details, `STDDEV` vs `STDDEV_SAMP` semantics). Report back anything
that errors and we'll debug it together, same as the rest of this
project.
