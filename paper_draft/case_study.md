# Building a Supply Chain Stress Predictor: What Actually Happened

*A case study in proxy targets, a leakage bug that inflated results, and the
engineering work that came after the model.*

## The problem

Retail companies want to know when demand is about to outstrip supply
*before* it happens. The M5 Forecasting dataset — 1,500+ item-store series
of daily retail sales, calendar events, and pricing — is a natural testbed
for this, but it has one honest limitation: **it contains no inventory
levels and no verified stockout events.** There is no ground truth to
predict.

So the first real decision in this project wasn't a modeling choice, it was
a labeling one: what should count as "stress" when the data doesn't say?

The target adopted was a transparent proxy — **a stress event is any day
where an item's sales exceed its own historical 90th-percentile demand.**
It's not a claim about actual shortages. It's a reproducible, auditable
stand-in that the README states plainly as a caveat rather than burying in
a footnote. That framing choice — say what the target actually measures,
not what it sounds like it measures — carried through the rest of the
project.

## The experimental journey

The model didn't start as XGBoost. It started with the simplest thing that
could fail informatively:

**Logistic regression, unweighted.** 94% accuracy, ~0% recall on stress
events. The model learned to always predict "no stress" and got rewarded
for it — a direct demonstration that accuracy is the wrong metric here
before any tuning began.

**Logistic regression, class-balanced.** Accuracy dropped to 78%, recall
rose to 30%. The tradeoff was now visible and explicit, not hidden inside
an aggregate score.

**Random Forest, balanced.** Recall jumped to 87%, precision fell to 8%.
This is where the project's central business framing crystallized: in a
supply chain, a **false negative** (a missed stress event) is usually more
expensive than a **false positive** (an unnecessary inventory check).
Recall, not accuracy, became the metric that mattered.

From there, feature engineering proceeded as a series of falsifiable
questions, each with a measured answer:

| Question | Finding |
|---|---|
| Does recent demand trend matter more than one-day spikes? | Yes — `rolling_mean_7` dominated feature importance (69%) over `sales_lag_1` (19%) |
| Does short-term volatility add signal beyond the mean? | Yes — `rolling_std_7` came in nearly as important as the rolling mean (41% vs. 48%) |
| Do calendar effects matter? | Weekends mattered (12% importance); formal holiday events and SNAP benefit days barely moved the needle |
| Does price matter? | It became the single strongest predictor — lower prices associated with higher stress, plausibly because discounts coincide with demand surges rather than causing stress directly |
| Does geography matter? | Substantially — stress rates ranged from 3% to over 10% across stores, and several store indicators outranked traditional lag features |

Each addition was evaluated on the same holdout, and each finding was
cross-checked against SHAP explanations later in the project — the SHAP
summary confirmed the same directional relationships (lower price → higher
stress, weekends → higher stress, high-stress stores → higher stress) that
the earlier exploratory analysis had already surfaced. That agreement
between two independent methods is a reasonable, if not conclusive, signal
that the model was learning real structure rather than memorizing noise.

XGBoost with `scale_pos_weight` for class balancing ultimately outperformed
the tuned Random Forest on the metric that mattered (minority-class F1:
0.23 vs. 0.22) and became the production model.

## The bug that was hiding in the validation itself

Here's the part that isn't in the original project notes, because it was
found after they were written.

The validation was designed to be chronological — train on early days,
test on later days, so the model is evaluated the way it would actually be
used. That's the right idea. The implementation had two bugs that quietly
broke it:

**Bug 1 — the split wasn't actually chronological.** The analytical table
was sorted item-major (all of one item's history, then the next item's),
not time-major. A positional train/test split on that ordering doesn't
divide *early days from late days* — it divides *some items from other
items*, with time mixed randomly within each side. The split had the
right name and the wrong effect.

**Bug 2 — the label leaked the future.** The 90th-percentile stress
threshold was computed over the *entire* dataset, including the holdout
period. That means each label was partly defined using sales figures that,
at prediction time, hadn't happened yet.

Both bugs make a model look better than it is, and neither throws an
error — they just quietly inflate the reported numbers. That's exactly why
they're dangerous: a broken train/test split doesn't fail loudly, it fails
by flattering you.

**The fix:** a single `split_day` boundary now threads through the entire
pipeline — `build_analytical_table`, `chronological_split`, and the
training pipeline all reference the same cutoff, so "before" and "after"
mean the same thing everywhere. The stress threshold is now estimated from
training-period data only.

**The proof it's fixed, not just patched:** 14 regression tests in
`tests/test_leakage.py` that don't just check the split logic once — they
perturb *future* values in the raw data and assert that *past* features and
labels are completely unchanged. If a future leakage bug is ever
reintroduced, these tests catch it by construction, not by someone
remembering to re-check manually.

**The honest result:** average precision moved from ~0.19 (the old, leaky
split) to **0.2225** (the corrected split) against a genuine 8.85% holdout
base rate — a 2.51x lift over random ranking. The corrected number is
lower than what an unvalidated version of this project might have
reported, and it's the one that's actually true. The README states the
base rate and the lift together, deliberately, so a reader can judge the
result without accuracy hiding the picture:

| Metric | Value | Reference |
|---|---:|---|
| Average precision | 0.2225 | 0.0885 if ranked at random (2.51x lift) |
| Precision @ top 100 alerts | 0.520 | 5.9x base rate |

And it beats the obvious heuristic — flag an item whenever yesterday's
sales already crossed the stress threshold — at matched alert volume: 33%
fewer false alarms, 39% more caught events, for the same analyst workload.
That comparison, not the raw precision number, is the project's actual
headline result.

## Engineering hardening: making the honesty durable

A correct result on one run is a data point. A correct result that stays
correct as the code changes is an engineering property, and that's the
gap this phase closed.

**Test coverage.** Three modules — `pipeline.py`, `predict.py`,
`evaluate.py` — had zero direct test coverage despite being the code paths
that wire training together and serve live predictions. `predict.py`,
specifically, is what the deployed FastAPI endpoint calls on every scoring
request; a bug there would have been live in production with nothing
catching it. Closing that gap added coverage for the missing-artifact
path, the pickle-fallback path, threshold behavior, and an end-to-end
integration test that runs the full pipeline against synthetic
M5-shaped data and asserts the chronological split is still valid — the
same guarantee `test_leakage.py` proves at the unit level, now proven
again at the pipeline level.

**Reproducible builds.** The Docker setup was already well hardened —
multi-stage builds, non-root users, `read_only` filesystems,
`cap_drop: ALL`, `no-new-privileges`, healthcheck-gated startup — but the
dependencies feeding it weren't pinned. `requirements-api.txt` used open
ranges (`fastapi>=0.115,<1.0`), which means the image built today could
resolve different transitive dependencies than the image built in three
months, with no error and no warning. Compiled lockfiles now pin every
dependency, verified to install cleanly and pass the full test suite in a
completely fresh environment, with a CI check that fails the build if the
lockfiles ever drift from their source files.

**A live, real dependency incompatibility, caught and fixed.** Adding that
CI check surfaced a genuine upstream problem: `pip-tools` 7.6.0 (the
current release) doesn't yet support `pip` 26.x — it imports a private pip
internal that pip 26 removed. This wasn't a bug in this project's code; it
was two "latest" packages that don't currently work together. The fix —
pin `pip<26` for that one CI step — was found by reproducing the exact CI
failure locally, confirming the root cause, and verifying the fix
end-to-end before pushing, rather than guessing and re-pushing blind.

**A second failure that looked identical but wasn't.** The same check
later failed again, on the same line of the same file, and the first
instinct was that it was a repeat of a known, already-fixed issue — a
transient PyPI timing blip, since re-running the job should self-correct a
true one-off. It didn't self-correct. That was the signal the first
diagnosis was wrong, and it led to the actual mechanism: `pip-compile`
deliberately preserves an existing pin if it still satisfies the
constraints file, rather than always jumping to the newest available
version. The CI check compiles to a brand-new filename with no prior pin
to prefer, so it always resolves the true latest — meaning a lockfile
regenerated "in place" on top of itself can silently stay one patch
version stale indefinitely, disagreeing with CI on every single run, not
just once. The fix was to regenerate from a clean slate rather than in
place, verified by reproducing five consecutive fresh resolutions before
trusting the result.

## Scaling past 100 items — and a critique that held up

Everything above was validated on 100 items — about 3.3% of the M5
catalog's 30,490 item-store series, not the 0.3% an early draft of this
project claimed. That correction itself is worth naming: "100 items"
sounds like "100 of 30,490," but every item spans all 10 stores, so 100
items is actually 1,000 series. Small arithmetic error, but the kind
that quietly understates a limitation instead of stating it — worth
catching and fixing before a reader catches it first.

The real question it raised wasn't "was the arithmetic right," it was
"does any of this hold up past a 100-item sample." That took three
separate pushes, not one.

**Push one: a stratified 5,000-series sample.** Random sampling would
have mostly reproduced the catalog's natural mix — 44% of items are
intermittent demand, the hardest and most business-relevant segment, and
a random draw would represent them at roughly that rate. Instead, 500
items were sampled stratified by department and demand class (the
Syntetos-Boylan ADI/CV² classification — smooth, erratic, intermittent,
lumpy), with intermittent and lumpy oversampled 1.5x relative to their
natural share. Every item spans all 10 stores, so 500 items is exactly
5,000 series with zero store-representation bias, for free.

Precision improved at every threshold over the 100-item run — a real
effect of more, more diverse data, not an artifact. But running the same
sampling design across 3 random seeds showed the *dollar* result wasn't
stable even though the *direction* was: ROI-optimal net ranged from
+$37,850 to +$85,500 depending purely on which 500 items got drawn. One
seed's number would have been a reasonable-looking, wrong headline. Three
seeds turned an implied point estimate into an honest range.

**Push two: pandas couldn't hold this much data on the reference
hardware (a 3.9GB-RAM, 1-core container), so the pipeline was ported to
Polars.** The port surfaced its own lesson before producing any result:
the first three attempts died to out-of-memory kills, not because Polars
is slow but because the defaults are expensive at this row count. Plain
string columns (`item_id`, `weekday`, event names) cost ~1.5GB
uncompressed at 9.5M rows before any join; `Categorical` dictionary
encoding fixed that. Plain Int64 columns holding values that fit in a
single byte (`wday`, `month`, SNAP flags) cost 8x more than necessary;
explicit downcasting fixed that too. Same lesson already learned once
with pandas dtypes, now paid for again in a different library — worth
remembering it doesn't transfer automatically.

**Push three: the complete 30,490-series catalog**, which doesn't fit
this hardware in one pass regardless of dtype tightening. Processed
store-by-store — 10 chunks of ~5.8M rows each, each one checkpointed to
disk before starting the next, so a mid-run failure loses one chunk, not
the whole job. Trained on a 20% row-subsample of the training period (a
real compute constraint, stated as one); evaluated on the complete,
non-subsampled 11.6M-row holdout, because the evaluation is the number
that actually gets trusted and shouldn't be the one that's approximated.

That chunking approach hid a bug that a metrics dashboard would never
have surfaced: `store_id`/`state_id` one-hot encoding, run one store at a
time, saw only a single category per chunk — and one-hot-encoding a
single-category column with `drop_first=True` produces *zero* columns.
No error. No warning. Just a joint model silently trained without
location features, on a result that still looked entirely plausible.
Caught by checking that every chunk produced the same column set before
trusting anything downstream of it — not by anything failing loudly.

**Then an external reviewer (Gemini) read the README's own stated
limitations and found the one that had been named but never checked.**
The stress threshold grouped by `item_id` alone, pooling sales across all
10 stores per item. A high-volume store would cross that pooled 90th
percentile more often simply by selling more — a volume artifact wearing
the costume of a demand signal. The README had already written this down
as an open question ("this has not been evaluated") months earlier and
then moved on to other work without closing it.

Verified before touching any code: store-level stress-event rate ranged
7.7%–19.9% across the 10 stores, correlated with store-level average
sales volume at **Pearson r = 0.85**. That's not a subtle effect. Fixed
by grouping the threshold on (item_id, store_id) instead of item_id alone — 
after the fix, the same correlation measured r = 0.034, and per-store 
rates tightened to 10.3%–16.4% (down from the original 7.7%–19.9% spread).

Recall dropped substantially at every threshold once the fix landed.
That's the correct outcome, not a regression: a real share of the
original recall was the model learning "which store-item combinations
sell a lot," not genuine demand stress, and losing that inflation is
what an honest number is supposed to do.

| Scale | Target definition | @0.80 net | ROI-optimal |
|---|---|---:|---|
| 1,000-series (original) | pooled by item (flawed) | -$50,414 | 0.95, +$26 |
| 5,000-series (3-seed avg) | pooled by item (flawed) | -$53,158 | ~0.88, +$57,877 |
| 5,000-series | corrected (item+store) | -$9,078 | 0.89, +$27,584 |
| 30,490-series (full) | corrected (item+store) | **-$209,794** | **0.90, +$47,148** |

The qualitative findings survived every correction: the shipped 0.80
threshold consistently loses money, and a real, smaller, more honestly
earned positive-ROI threshold exists near 0.90. What changed was
magnitude and cause, not conclusion — and a labeling artifact getting
caught by outside review, verified rather than just accepted, and fixed
with a regression test guarding against it coming back, is a more useful
thing to have happened to this project than a clean first draft would
have been.

## Honest limitations

This project is explicit about what it doesn't do, both in the README and
here:

- The target is a proxy for stress, not a verified stockout label. Every
  result should be read through that lens.
- Precision at any usable alert volume is modest — this is a
  **prioritization aid** for a review queue, not a detection system with
  confidence.
- Model scores are not calibrated probabilities; `scale_pos_weight` shifts
  them upward by construction.
- The default threshold (0.80) trades recall down to **0.03** on the full
  30,490-series catalog under the corrected, store-volume-confound-free
  target definition (was 0.12 under the original, flawed one — a more
  honest number, not a regression; see "Scaling past 100 items" above).
  Most stress events are still missed, and that number is stated without
  softening.
- The full-catalog result was trained on a 20% row-subsample of the
  training period, for compute reasons on the reference hardware — stated
  plainly rather than implied to be the complete training set. The
  holdout evaluation, which is the number that matters most, was not
  subsampled.

## What this project is actually demonstrating

Not "I can train a gradient-boosted model to 92% accuracy" — that number
was easy to get and was actively misleading. The real demonstration is
narrower and more defensible: **a proxy target stated honestly, a
validation methodology that was checked hard enough to find its own bugs,
a fix proven with tests rather than asserted, infrastructure hardened
enough that those guarantees hold up outside a notebook, and a real flaw
caught by outside review that got verified before being trusted and
fixed with a regression test instead of a shrug.**
