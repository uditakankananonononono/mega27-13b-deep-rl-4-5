# Native solver record-integrity check

The full-integration merger previously verified only plan hash and step count.
It now rejects seed, policy, variant or source-commit mismatch, altered action
schedules, nonfinite or wrong-length reward vectors, and inconsistent saved
reward/beta/duty/cost summaries. Eleven deliberate corruption tests exercise
these checks. All nine original records are retained and the merged result
is unchanged. This is evidence-integrity hardening, not a new simulation,
independent-model validation or clinical benefit. Full suite: 70 passed.
