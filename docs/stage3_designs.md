# Stage 3: designs that were written and not implemented

> **Status: a design. Nothing in this document was implemented, run or tested.** It was written on 2026-11-23 at the close of stage 3 (`docs/mlops.md` §13; `docs/final_report.md` §11). The project has no live traffic and no users, so none of it could be tried. Where the document needs a quantity that only a live system has, it says so and names the quantity; it does not invent a value.
> **Numbers.** Every number below is quoted from a record of the project (a result table, `docs/final_report.md` or `docs/mlops.md`) and says where it comes from. No sample size, duration or threshold is computed here, because the inputs to compute them do not exist in the data.

---

## 1. An online experiment for this system

### 1.1 The question

Does showing a customer the Top-12 of this system, instead of what is shown today, change what the customer buys? The offline results cannot answer this (§1.9). The experiment measures a difference between groups of customers who differ only in which recommender served them.

### 1.2 Unit of randomisation: the customer

- **A customer is assigned to one arm for the whole experiment**, by a hash of the customer id and a salt that belongs to this experiment. The assignment needs no stored table and gives the same answer in every service and every week.
- **Why not the request or the session.** The system produces one list per customer per week; a customer who sees two recommenders on two visits carries the effect of one into the other, and a purchase could not be attributed to an arm. The weekly cadence also means that the unit of observation is naturally "a customer in a week".
- **Customers without an id** (not logged in) are outside the experiment: the system can only give them the fallback list.
- A new salt for every experiment, so that the groups of one experiment are not the groups of the next.

### 1.3 Arms

| Arm | What the customer is served | Share | Purpose |
|---|---|---|---|
| Control | The recommender that is live today | most of the traffic at first | The comparison |
| Treatment | This system: six recall channels and the LambdaMART ranker, the weekly Top-12 | grows after the canary stage (§1.7) | The effect to measure |
| Long-term holdout | **The RRF baseline** of the same candidates (reciprocal-rank fusion of the six channels, no trained model) | a small fixed share, kept for months after the experiment ends | Two things. If revenue from the slot falls months later, the holdout says whether the trained ranker or something outside (season, assortment, prices) is the cause. And it measures the cumulative value of the ranker over a baseline that needs no training, which a short experiment cannot |

The RRF baseline is already computed by the pipeline every week for the closing step (`docs/final_report.md` §11.3), so serving it needs a second prediction set and a second pointer, not a second system.

### 1.4 Metrics

**Primary metric: revenue per randomised customer from the recommendation slot, over the experiment period.** Every customer assigned to an arm counts, whether or not they visited (intention to treat); a customer who bought nothing counts as zero. Revenue is net of returns once returns are known.

**Guardrail metrics** (the experiment may not make these worse beyond a limit fixed in advance):

| Guardrail | Why |
|---|---|
| p95 latency of the endpoint | The treatment must not slow the page down |
| Error rate of the endpoint, and the share of requests answered by the fallback | A broken or empty treatment would look like "no effect" |
| Return rate of articles bought through the slot | Revenue that comes back is not revenue |
| Total revenue per customer (all of the site) | The slot must not only move purchases from elsewhere on the page into the slot |

**Secondary, for reading the result, not for deciding it**: click-through and add-to-cart rate of the slot, number of distinct articles shown, share of shown articles among the most popular ones.

### 1.5 Sample size and duration: how they would be computed

For a difference in means between two arms, the customers needed per arm are

    n = 2 × (z(1 − α/2) + z(1 − β))² × σ² / δ²

- **σ²**: the variance, across customers, of the primary metric over the planned period. **It has to be observed on the live site before the experiment**; this dataset cannot give it, because it records purchases and not which purchases came through a recommendation slot.
- **δ**: the smallest difference worth detecting, in the unit of the metric. It is a business decision, fixed before the experiment.
- **α and β**: the usual error rates, fixed in advance, with a correction if more than one treatment arm is compared with the control.

What the records of this project say about the inputs, without giving them:

- **The metric is mostly zeros.** In the replayed weeks 68,984 to 82,802 of the 1,371,980 customers bought anything at all in a week (`results/m12_replay/comparison.csv`, quantity `queries`). A per-customer revenue metric over all randomised customers therefore has a large variance relative to its mean, which pushes n up; the pre-experiment period should be used to reduce it (a covariate adjustment with each customer's earlier spend).
- **Weeks differ a lot.** Single-week MAP@12 over V1–T4 ranges from 0.030112 to 0.042435, and the T weeks are 19.8% above the V weeks (`docs/final_report.md` §1, §5). An experiment must compare arms inside the same weeks, never one arm's weeks with another's.
- **Duration is counted in whole weeks**, because the recommendations change once a week and purchases follow a weekly cycle; it is at least the number of weeks that n requires at the available traffic, and not fewer than a few complete weeks so that one unusual week does not decide the result. The duration is fixed before the start.

### 1.6 Sample-ratio check

Every day, the number of customers actually observed in each arm is compared with the planned split by a chi-square test. A mismatch at a strict level (fixed in advance; the usual choice is far below 5%, because the check is repeated daily) means that assignment or logging is broken: **the analysis stops and no result is read** until the cause is found. This check comes before any look at the metrics.

### 1.7 Canary stage

Before the measured experiment, a very small share of customers is served by the treatment for a few days. **Only the guardrails are read** (latency, errors, fallback share, the alert lines of the service). The canary answers "does it work under real traffic", not "is it better". It ends by a rule written beforehand: no guardrail above its limit for the whole period. The service of this project has never seen concurrent traffic (`docs/final_report.md` §11.12 item 9), so the canary is also its first load test.

### 1.8 Stopping and rollback

- **The decision is made once, at the fixed end**, on the primary metric with its confidence interval. Looking earlier and stopping when the difference looks good is not allowed, unless a sequential design with its stopping boundaries was declared before the start.
- **Stopping early is allowed for harm only**: a guardrail beyond its limit, a failed sample-ratio check, or an incident. The limits are written before the start.
- **Rollback has two levels.** The arm is routed back to the control (the experiment's switch). Inside the treatment, a bad weekly prediction set is rolled back by moving the pointer to the previous set; in the drill the service followed in 41 seconds (`results/m14_serving/rollback_drill.csv`).
- **The outcome is recorded whatever it is**, including "no difference detected", with the interval.

### 1.9 What the offline results can and cannot predict

**They can say:**

- The ranker orders bought articles better than the untrained baseline of the same candidates: pooled MAP@12 on T1–T4 is 0.039194 against 0.021154 (`docs/final_report.md` §1). If the treatment does not beat the holdout online, something between the offline evaluation and the page is wrong.
- **Where to expect less.** For about half of the queries no channel recalls any bought article (48% to 52% per week), and customers without history gain almost nothing from the ranker (`docs/final_report.md` §7.2 items 11 and 12). The analysis should be split by whether a customer has history, decided before the start.
- **What is not worth testing first.** DeepFM and LambdaMART could not be told apart offline in three comparisons (`docs/final_report.md` §10.8); an online test between them would need far more traffic than a test of either against the control.

**They cannot say:**

- **Whether the recommendations cause purchases.** The purchases in the data were made under H&M's own recommender. An article this system "predicts" may have been bought anyway. MAP@12 rewards predicting what happens, not changing it.
- **Anything about clicks, position or exposure.** The data hold no impressions, so nothing is known about how the 12 positions are seen.
- **The size of the effect.** There is no known mapping from a MAP@12 difference to a revenue difference; the offline number ranks candidates for a test and does not forecast its result.
- **The level in another season.** The offline scores rise with the season (§1.5).

---

## 2. Online metrics a live system would watch

The pipeline's quality metrics need the purchases of the predicted week, so they arrive one tick late (`docs/final_report.md` §11.5). A live system adds signals that arrive within hours.

| Metric | Cadence | What it answers | Relation to the pipeline's weekly metrics |
|---|---|---|---|
| Requests, error rate, p95 latency of the endpoint | hourly | Is the service up and fast | The service already logs status and latency per request; alert policies on them were not built |
| Share of requests answered by the fallback | hourly | Are customers found in the prediction set | The `source` field of every request line. A jump after a publication means the new set lacks customers |
| Age of the served prediction set | hourly | Did the weekly tick publish | The anchor in every answer; older than a week plus a margin means a missed or stopped tick |
| Impressions and click-through rate of the slot | hourly, daily | Do customers look at and react to the list | No offline counterpart: the data hold no impressions. The earliest sign of a bad publication, days before any labelled metric |
| Add-to-cart rate and conversion of the slot | daily | Do clicks become purchases | The numerator of the weekly Recall@12 and MAP@12 counts the same purchases, but for every buyer whether or not they saw the list |
| Revenue from the slot, net of returns | daily, weekly | The primary metric of §1.4 | The hit value of the pipeline is its offline analogue: the summed prices of recommended articles that were bought, counted without knowing whether the list was seen. It is not revenue caused by the system |
| Distinct articles shown; share of shown articles among the 100 most popular | daily | Is the list collapsing onto bestsellers | The pipeline computes both at publication time as output signals and flags a change of more than 25% against the previous 4 weeks |
| Recall of each channel, union recall, MAP@12 of the ranker and of the RRF baseline | weekly, one week late | Did last week's list predict last week's purchases | The labelled metrics of the closing step; the only ones that separate the channels from the ranker |

**How to read them together.** The hourly signals say that the system is serving; the daily ones say that customers respond; the weekly labelled ones say why (a channel, the ranker, the inputs). A fall in click-through with unchanged labelled metrics points outside the model (the page, the assortment, the season); a fall in a channel's recall with unchanged click-through is an early warning that has not reached customers yet. The levels are not comparable across the two kinds: a MAP@12 of 0.04 and a click-through rate are different quantities.

---

## 3. A rule for re-selecting hyperparameters

The weekly pipeline refits the models with **locked** hyperparameters; it never searches (`docs/mlops.md` §3). Re-selecting them is a human decision. This is the proposed rule for when and how.

### 3.1 What triggers a re-selection

| Evidence | Triggers | Why |
|---|---|---|
| A quality alert (the same labelled flag in 3 consecutive weeks) that remains after the inputs and the pipeline were checked | **Yes** | A persistent change, not one unusual week |
| The ranker not above the RRF baseline in a closed week, after the same checks | **Yes** | The trained model no longer adds anything to its own candidates |
| A change of what the system is given: a new channel, new features, a new catalogue structure | **Yes** | The locked values were selected for another system |
| A scheduled review, at a fixed interval and at each change of season | **A review, not automatically a search** | The scores carry a seasonal component (`docs/final_report.md` §5), and the assumption that values selected in one season hold in the next was never tested (§7.1 item 10) |
| A single flag; one week's dip | No | The replay raised 5 flags in 8 weeks and none repeated (`docs/final_report.md` §11.5) |
| A result of an already used test block | **Never** | §3.2 |

### 3.2 Which weeks it may use

- **A new split of recent closed weeks**, in the order of the original one: weeks to select the recall configuration, weeks that only generate ranker training data, weeks to select the ranker, and **a test block that nobody has looked at**, evaluated once after everything is locked. If no untouched weeks exist yet, the test block is the coming weeks, and the decision waits for them.
- **Weeks that served as a final test before are never used to select again.** In this project that is T1–T4, whose labels have been read on recorded occasions (`docs/final_report.md` §7.2 item 5).
- **The selection rule is written and committed before any score is seen**, as `configs/selection_rule.yaml` was, with the same content: a candidate replaces the default only if the pooled score of the selection weeks improves by at least 1% relative and it is strictly better in at least 3 of the 4 weeks; among candidates that are about the same, the simpler and cheaper one is chosen (`docs/final_report.md` §3.3).
- Every prediction uses only data from before its anchor, through the same as-of functions.

### 3.3 Who approves, and what is recorded

- **A person who owns the system approves**, in writing, before the search starts (the trigger and the split) and again before the result is adopted (the comparison on the untouched block).
- **The result is a new set of lock files**; the old ones are not modified. The decision, the weeks, the rule and the result go into the decision log.
- **Adoption goes through the canary stage and, if traffic allows, the experiment of §1**, with the old configuration as the control. A new configuration that is better offline is a candidate, not a release.
- The pointer rollback stays available for every weekly set the new configuration publishes.

---

## 4. What a production version would change

Each item is a limitation recorded in `docs/final_report.md` §11.12 (or in `docs/mlops.md`) that has a known remedy. None was built.

| Limitation | Remedy |
|---|---|
| A week opened in `serve` mode cannot be closed, because the features of the buyers were removed with each shard | Build the closing step's inputs for the customers who bought when the week is closed: their candidates and features are recomputed as of the same anchor from the tick's own ingested data, which is still in the state. Or keep the features of every shard, at the cost of about 26 GB per week |
| The clock is simulated, and the query list of `eval` mode knows who will buy | In production only `serve` mode exists and the anchor is the scheduler's date; the weekly labelled metrics are computed for the buyers at closing time (the item above) |
| The platform's request log holds the customer id from the URL | Take the id from the request body or from the caller's authenticated session, or exclude the service's request log from logging |
| One region, one instance, no load test | A load test at the expected request rate; a minimum instance to remove the cold start of 6 seconds for the first request; a second region if the page depends on the endpoint |
| The service keeps both sets in memory during a switch | Measure a switch between two full sets and size the memory from it; or load the new set in a new revision and move traffic |
| Alerting covers four log lines and failed executions | Add policies on the service's error rate and p95 latency, and an alert for a tick that did **not** run or did not finish in its expected time (an absence, which no log line can report) |
| After a rollback the next tick scores the set that was published, not the one that was served | Score the set the pointer named during the week: the table of pointer moves already records which that was |
| MAP@12 of T4 was 0.74% and 0.80% below the record and was not investigated | Run T4 again twice with one input swapped each time (the recorded candidates with the replay's ranker; the replay's candidates with the recorded ranker) to see which side carries the difference |
| The pair overlap of candidates and the overlap with the Kaggle set were not measured in the cloud | Run the comparison as a job in the cloud, with the records uploaded next to the state, instead of on the laptop |
| R6 fine-tuning is not repeatable with the stage-1 settings | Use the deterministic setting on every platform, the laptop included, and keep treating a model file with its sha256 as the record, not the ability to train it again |
| MLflow is a local view | A hosted tracking server with a database, written by the pipeline at the end of each tick; the manifests stay the record |
| The job is far larger than it needs (24 GiB; the highest true peak of a process was 7.53 GB) | Lower the memory after a few weeks of measurements; the price follows |
| The state grows with every week (about 2.7 GiB per `eval` week) | A lifecycle rule that removes the large intermediates of weeks that are closed and older than the ranker's training window, keeping the records and the models |
| Infrastructure is created by scripts | Terraform, with its state stored and reviewed, once more than one person changes it |
| The deployer and the operator hold wide predefined roles | Custom roles with only the permissions the workflows and the runbook use |
| The fallback for unknown customers is one list | Use what a request does know (the page, the country, the device) to choose among several precomputed lists |
