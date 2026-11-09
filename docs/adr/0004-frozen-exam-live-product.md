# 4. Frozen exam, live product

**Decision.** Accuracy is measured on frozen past events whose causes the regulator has already published. The product runs on live data. Both use the same tables and tools.

**Why.** Live data has no answer key: if the data keeps moving, you can't tell whether the system improved or the data changed. A frozen exam is like a driving test on a closed course. Because the live and exam paths share every tool and the same schema, a score on the exam carries over to live behaviour.

**How live behaviour is still checked.** The Warden checks freshness, price sanity and gaps on the live window. `reconcile` closes the loop months later: each live investigation is compared with the AER's report on that quarter and proposed as a new exam row for a human to review.

**Limits.** The exam is small (20 events) and its labels are drafts until reviewed, so scores come with wide uncertainty. Archive retention on NEMWeb is limited; events older than about a year may need the MMSDM monthly archive, which is not implemented yet.
