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

## Honest limitations

This project is explicit about what it doesn't do, both in the README and
here:

- The target is a proxy for stress, not a verified stockout label. Every
  result should be read through that lens.
- Precision at any usable alert volume is low (0.27–0.52 depending on
  threshold) — this is a **prioritization aid** for a review queue, not a
  detection system with confidence.
- Model scores are not calibrated probabilities; `scale_pos_weight` shifts
  them upward by construction.
- The default threshold (0.80) trades recall down to 0.12 — meaning most
  stress events are still missed. That number is stated in the README
  without softening.

## What this project is actually demonstrating

Not "I can train a gradient-boosted model to 92% accuracy" — that number
was easy to get and was actively misleading. The real demonstration is
narrower and more defensible: **a proxy target stated honestly, a
validation methodology that was checked hard enough to find its own bugs,
a fix proven with tests rather than asserted, and infrastructure hardened
enough that those guarantees hold up outside a notebook.**
