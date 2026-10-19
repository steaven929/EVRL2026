# SIP demand split and validation

The reviewer experiments use `od_demand_1.csv` through
`od_demand_30.csv` as 30 complete demand scenarios:

- training: scenarios 1–25;
- held-out testing: scenarios 26–30;
- each file contains 50 zones, 2,450 inter-zone OD pairs, and 26 time steps;
- every scenario and every time step has positive total demand;
- the 30 scenario totals are not identical.

The validated files were restored byte-for-byte from repository commit
`a95d0d5cf2950c62f037d3d13bb7e9e21f06b36e`. The `demand_original/q_*.csv`
files must not be treated as 30 complete days: some contain demand in only one
time step and `q_26.csv` through `q_29.csv` have zero total demand. The
conversion utility now rejects such incomplete sources before overwriting an
output file.

The repository does not contain external calendar-date provenance for these
30 files. The paper should therefore call them **demand scenarios** or
**synthetic daily demand realizations**, unless the original data source and
generation procedure are documented separately. Do not describe them as 30
observed calendar days without that evidence.

Regression checks are in `test_multiday_data_split.py` and
`test_sip_demand_conversion.py`.
