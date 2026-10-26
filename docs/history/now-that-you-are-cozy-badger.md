# Project Audit: Findings and Fix Plan

## Outcome (2026-06-11)

Implemented. A1-A3 and B1-B4 fixed; improvements filed as UltraSignup-e4o, -7yu,
-rl7, -do5, -0dy, -3sb; bugs closed as -d54, -rwy, -yzd, -618. Deviations from plan:
- Root repo has NO remote (the audit agent's "unreachable HTTPS remote" was wrong);
  creating a new GitHub repo was denied by the permission classifier, so root stays
  local-only and CLAUDE.md now documents per-repo pushes for repos with remotes.
- A2: *.nc was already gitignored; the real gap was an absolute-path traceplot
  symlink + untracked diagnostics PNGs — materialized and committed per convention.
- Stale caches from changed models quarantined with .stale-pre-audit-fix suffix
  (nb 2 reporting trace, nb 4 m5 trace, nb 3 map_full_dataset x3) — refits required.
- Fixes verified with synthetic-data compile tests (truncation term matches scipy
  betabinom analytically; nb 4 masked logp differs from double-counting model by
  exactly the removed terms; nb 1 MAP builds and optimizes). Full refits on real
  data are the next session's work.

## Context

Full audit of the UltraSignup workspace requested, with emphasis on the **PyMC modeling
itself**. Scope agreed with user: fix the things that are straight-up wrong; file the
rest as bd issues. 18 issues were already open in beads — everything below is either
new, or a correction/sharpening of a tracked issue.

## Part A — Straight-up wrong: workspace & hygiene (fix now)

**A1. Session-close protocol in CLAUDE.md cannot succeed as written.**
- Root repo remote is `https://github.com/justmytwospence/UltraSignup.git` — unreachable,
  and `main` has no upstream; bare `git push` fails. Root has never been pushed.
- Beads is embedded-dolt with no dolt remote; `bd dolt push` has nothing to push to
  (`bd prime` itself says "local-only").
- Fix: `git remote set-url origin git@github.com:justmytwospence/UltraSignup.git`
  (match scraper's working SSH form), `git push -u origin main`. Rewrite the
  Session Completion section of root `CLAUDE.md`: per-repo commit+push steps, drop
  `bd dolt push` (or document it as a no-op until a dolt remote exists).
- Push scraper's 5 unpushed commits (`git -C scraper push`); commit scraper's dirty
  `uv.lock`; commit root's staged `.beads/issues.jsonl`.

**A2. Analysis model artifacts not gitignored.**
- `analysis/notebooks/models/model_2/alpha3_beta629/` (4.4 MB of `.nc` traces) is
  untracked with no ignore rule. Add `notebooks/models/` to `analysis/.gitignore`.

**A3. Doc drift (root CLAUDE.md + nb 1).**
- Root `CLAUDE.md` describes `data/cache/` as "mostly stale shared caches" — `data/` is
  empty and `data/cache/` doesn't exist. Update the sentence.
- Stale comment in `analysis/notebooks/1_finish_times.py:123-126` (references a
  tune=1000 cache-invalidation rationale; config is now tune=2000). Delete/update.

## Part B — Modeling audit (the main ask)

Models reviewed in code: nb 1 MCMC (`1_finish_times.py:281`) + MAP (`:1019`),
nb 2 reporting (`2_dnf_reporting.py:179`), nb 3 joint models (`3_observed_dnfs.py:676`,
`:2659`), nb 4 model_m5 (`4_unobserved_dnfs.py:733`).

### Straight-up wrong (fix now)

**B1. nb 1 MAP likelihood misuses the log-scale noise as a minutes-scale sigma.**
`1_finish_times.py:1030`: MCMC was upgraded to `LogNormal(mu=log_time, sigma=noise)`
where `finish_time_noise` is dimensionless log-sd (~0.15). The MAP stage still uses
`Normal(mu=pace*d, sigma=noise*d)` with that same posterior median (extracted at `:917`).
Correct sd on the minutes scale would be ≈ time·0.15 ≈ pace·d·0.15 — the MAP likelihood
is ~pace (≈10×) too tight, so closure-course effects are drastically under-shrunk
(overfit on sparse courses).
- Fix (one cell): make the MAP likelihood identical to MCMC —
  `pm.LogNormal('finish_times', mu=_expected_log_pace + log(d), sigma=_finish_time_noise[g], observed=observed_times_full)`.
- Also update stale tracked issue **UltraSignup-618**: its notes still say nb 1 MCMC is
  "Normal-on-time"; MCMC is fixed, the residual bug is the MAP stage.

**B2. nb 2 reporting model has uncorrected zero-truncation selection bias.**
`2_dnf_reporting.py:173-187`: the DNF–distance relationship is fit only on races with
`has_dnf == 1`, but a genuinely-reporting race can have zero DNFs by chance — especially
short races and small fields. Conditioning on n_dnf ≥ 1 without truncating the likelihood
biases `mu_logit_dnf` upward and distorts `beta_distance_dnf` (bias is largest exactly
where the reporting question matters most). The derived `p_reports` inherits the bias.
- Fix: zero-truncated BetaBinomial — keep the BetaBinomial logp and add
  `pm.Potential('truncation', -pm.math.log1mexp(logp_zero).sum())` style correction,
  i.e. subtract `log(1 - P(Y=0 | n_i, alpha_i, beta_i))` per race (P(0) has closed form:
  `Beta(alpha, beta + n) / Beta(alpha, beta)`).

**B3. nb 4 double-counts the zero-DNF evidence.**
`4_unobserved_dnfs.py:756-765`: zero-DNF courses get the correct marginal mixture
`Potential: log(psi·(1-p)^n + (1-psi))` — but their results are *also* included in the
per-result `Bernoulli(p = psi·p_i)` likelihood. Two different marginalizations of the
same data are both in the joint; the zero-DNF evidence is counted twice (and the
per-result version wrongly treats the shared reporting indicator as independent).
- Fix: mask zero-DNF-course results out of the Bernoulli (`finish_likelihood` over
  results from reporting courses only, where reporting_prob = 1) and keep the Potential
  as the sole term for zero-DNF courses.
- Related (fold into tracked **UltraSignup-9sx** P5.4): the mixture's `(1-p)^n` uses the
  distance-only baseline p while the Bernoulli uses race-difficulty-adjusted p —
  inconsistent; and the `selection_correction` Potential (`:770-772`) keeps only the
  `-log P(finish)` denominator while omitting the pace-dependent numerator
  `P(finish | pace_i)`, so it cannot correct slope bias. Annotate the issue.

**B4. nb 3 MAP stage fixes finish_time_noise at the prior constant, not the posterior.**
`3_observed_dnfs.py:687-688`: in fixed mode `finish_time_noise = 0.1` (the prior scale,
`:487`) instead of the MCMC posterior median; `fixed_params` at `:2096` never passes it.
Mis-weights the LogNormal vs Bernoulli components and mis-shrinks course effects.
- Fix: add `finish_time_noise` to `fixed_params` (posterior median, like the others)
  and use it in fixed mode.

### Improvements — file as bd issues (don't implement now)

1. **nb 1 has no runner random effect** — course effects absorb field strength
   (destination races attract atypical fields); within-runner correlation also
   understates uncertainty. The k-core was built around runner×course density precisely
   to support runner effects; nb 3 model 2 has them, nb 1 doesn't. (Relates to tracked
   P5.7 but nb 1's omission is untracked.)
2. **Course identity keyed by raw `name` string** in nb 1/3 (`model_data['name']`) vs
   `course_id` in nb 4 — name collisions merge distinct courses; standardize on ids.
3. **nb 2 functional form inconsistency**: linear-in-km distance on logit vs
   log-distance(+quadratic) everywhere else.
4. **nb 4 informative quadratic prior** `beta_dist_dnf_quad ~ N(0.5, 0.3)` looks
   tuned-to-data; justify or widen.
5. **empirical_priors.py hygiene**: course-variation residuals computed on finishers
   only (biased for high-DNF courses); silent drop of courses with <5 obs; document
   miles assumption. (Extends tracked UltraSignup-nfw.)
6. **Scraper robustness** (from the scraper sweep; all non-urgent): unbounded
   db-writer queue (`core/concurrency.py:109`), no retry on elevation-extractor
   timeouts, parameterize the f-string LIMIT in `database/operations.py:259,298`
   (int-typed via click, so hygiene not vulnerability), redundant `time`/`formattime`
   duplicate column. Note: "SQL injection" and "committed API key" findings from the
   sweep were verified FALSE (typed int arg; .env never committed).

### bd issue updates during implementation

- Update UltraSignup-618 (stale notes; new MAP-stage scope = B1).
- Annotate UltraSignup-9sx with the B3-related approximation details.
- Create issues for B-improvements 1–6 (before writing code, per beads workflow).
- Close issues as fixes land.

## Files to modify

- Root: `CLAUDE.md` (A1 protocol, A3 data/cache), git remote config, pushes.
- `analysis/.gitignore` (A2).
- `analysis/notebooks/1_finish_times.py` (B1 MAP cell, A3 stale comment).
- `analysis/notebooks/2_dnf_reporting.py` (B2 truncation Potential).
- `analysis/notebooks/4_unobserved_dnfs.py` (B3 Bernoulli mask).
- `analysis/notebooks/3_observed_dnfs.py` (B4 fixed_params).

## Verification

- A1: `git status` in all three repos shows clean + "up to date with origin".
- B1: re-run nb 1 MAP only (fast; MCMC trace is cached) via marimo-pair per
  analysis/CLAUDE.md; closure-course effect spread should shrink markedly; PPC on
  closure rows should stop overfitting.
- B2: refit nb 2 (<30 min at production size); expect lower mu_logit_dnf and a
  steeper distance slope; sanity-check predicted P(zero DNF | reporting) by distance
  bin against observed zero shares.
- B3/B4: refit at dev k-core (3,629) per the scaling table; check r_hat, psi posterior
  shift, and that course-effect posteriors widen for zero-DNF courses.
- Models are fit through the running marimo kernels only (hard rule in
  analysis/CLAUDE.md) — never standalone scripts.
