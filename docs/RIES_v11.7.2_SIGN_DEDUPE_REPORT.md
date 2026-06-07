# RIES v11.7.2 sign-equivalence dedupe report

## Summary

Pre-dedupe rows: 36,685
Post-dedupe rows: 36,443
Rows removed: 242
Mixed-sign absolute-value groups scanned: 742
Rows in mixed-sign groups: 1,484
Retained mixed-sign groups as distinct/non-proved: 500

## Mixed-sign groups by family before dedupe

- BETA_LOG_PLUS_MINUS/unit_beta_log_pm: 142 rows involved
- BINOMIAL_INVBINOM_SUM/normalized_central_binomial: 2 rows involved
- HYPERGEOM_EULER_INTEGRAL/gauss_euler_r: 1 rows involved
- POLYLOG_LERCH_SUM/lerch_basic: 12 rows involved
- POLYLOG_LERCH_SUM/shifted_rational_polylog: 4 rows involved
- RATIONAL_LOG_BETA/unit_rational_log_beta: 160 rows involved
- RATIONAL_TAIL_SUM/fast_rational_tail: 537 rows involved
- TRIG_BETA_LOG/trig_beta_log: 10 rows involved
- TRIG_RATIONAL_FOURIER/trig_rational_fourier: 616 rows involved

## Rows removed by rule

- trig-Fourier reflection x->pi-x: 170
- unit-beta/rational-log sign convention: 72

## Rows removed by family

- RATIONAL_LOG_BETA/unit_rational_log_beta: 72
- TRIG_RATIONAL_FOURIER/trig_rational_fourier: 170

## Sample removed cases

- abs(value16)=0.5332093694507152; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000402; removed=INTSUMDB_028442; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.4979592630272659; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000445; removed=INTSUMDB_028421; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.4893286504007088; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000464; removed=INTSUMDB_028409; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.4269973678628307; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000568; removed=INTSUMDB_028353; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.4201332497734422; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000584; removed=INTSUMDB_028346; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.4112147916753063; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000598; removed=INTSUMDB_028333; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.4095763949651731; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000602; removed=INTSUMDB_028332; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.4045618781385931; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000614; removed=INTSUMDB_028327; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.3783062168048461; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000672; removed=INTSUMDB_028298; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.3575106628186250; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000717; removed=INTSUMDB_028268; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.3512341651354769; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000730; removed=INTSUMDB_028262; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta
- abs(value16)=0.3492205657275813; rule=unit-beta/rational-log sign convention; kept=INTSUMDB_000735; removed=INTSUMDB_028258; families=BETA_LOG_PLUS_MINUS/unit_beta_log_pm; RATIONAL_LOG_BETA/unit_rational_log_beta

## Policy

Conservative sign-equivalence pass: remove only rows whose value is the negative of another row at value16 precision and whose family-specific canonical kernel proves they differ by a global sign convention. Nontrivial equal-absolute-value identities remain in the database.
