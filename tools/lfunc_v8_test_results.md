# RIES v8 L-function matcher test results

| case | input | expected | status | first matching row |
|---|---:|---|---|---|
| L2 rational low 8 digits: L/pi | `0.080800374` | rational / 11.2.1 | PASS | `L-rational #1: x = L(f,1)/π` |
| L2 rational high 25 digits: 3/2*L*pi^2 | `3.757978120626313012636993` | rational / 11.2.1 | PASS | `L-rational #1: x = 3/2·L(f,1)·π²` |
| L2 quadratic low 7 digits: sqrt(2)*L | `0.3589866` | quadratic / 11.2.1 | PASS | `L-quadratic #1: x = (√8)/2·L(f,1)` |
| L2 log-extra high 25 digits: log(2)*L | `0.1759497701603644365005967` | log / 11.2.1 | PASS | `L-log #1: x = L(f,1)·log(2)` |
| L4,1 quadratic high 25 digits: sqrt(3)*L | `0.2795016192693953841783954` | quadratic / 5.4.1 | PASS | `L-quadratic #1: x = (√12)/2·L(f,1)` |
| L4,2 log-extra high 25 digits: Gamma(1/3)*L | `1.103351183569710873591804` | log / 5.4.1 | PASS | `L-log #1: x = L(f,2)·Γ(1/3)` |

6/6 cases passed.