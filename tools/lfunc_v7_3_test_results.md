# RIES v7.3 L-function matcher test results

| case | input | expected | status | first matching row |
|---|---:|---|---|---|
| L2 rational low 8 digits: L/pi | `0.080800374` | rational / L(f2#1,1; N=11) | PASS | `L-rational #1: x = L(f2#1,1; N=11)/π` |
| L2 rational high 25 digits: 3/2*L*pi^2 | `3.757978120626313012636993` | rational / L(f2#1,1; N=11) | PASS | `L-rational #1: x = 3/2·L(f2#1,1; N=11)·π²` |
| L2 quadratic low 7 digits: sqrt(2)*L | `0.3589866` | quadratic / L(f2#1,1; N=11) | PASS | `L-quadratic #1: x = (√8)/2·L(f2#1,1; N=11)` |
| L2 log-extra high 25 digits: log(2)*L | `0.1759497701603644365005967` | log / L(f2#1,1; N=11) | PASS | `L-log #1: x = L(f2#1,1; N=11)·log(2)` |
| L4,1 quadratic high 25 digits: sqrt(3)*L | `0.2795016192693953841783954` | quadratic / L(f4#1,1; N=5) | PASS | `L-quadratic #1: x = (√12)/2·L(f4#1,1; N=5)` |
| L4,2 log-extra high 25 digits: Gamma(1/3)*L | `1.103351183569710873591804` | log / L(f4#1,2; N=5) | PASS | `L-log #1: x = L(f4#1,2; N=5)·Γ(1/3)` |

6/6 cases passed.