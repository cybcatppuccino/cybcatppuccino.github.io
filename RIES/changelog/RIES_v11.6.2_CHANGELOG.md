# RIES v11.6.2 Changelog

Focused linear-combination catalogue update from v11.6.1.

## Added

- Added ten level-5-only constants to the low-precision sparse linear-combination matcher:
  - `π√2`, `π√3`
  - `Sum[1/(n^2 + 1), {n,0,Infinity}]`
  - `Sum[1/(n^3 + 1), {n,0,Infinity}]`
  - `Sum[(-1)^n/(n^2 + 1), {n,0,Infinity}]`
  - `Sum[(-1)^n/Binomial[2 n,n], {n,0,Infinity}]`
  - `Sum[1/Binomial[3 n,n], {n,0,Infinity}]`
  - `Sum[1/(n^2+n+1), {n,0,Infinity}]`
  - `Sum[(-1)^n/(n^2+n+1), {n,0,Infinity}]`
  - `Sum[1/(1+2^n), {n,0,Infinity}]`

## Changed

- The linear-combination basis cache is now keyed by depth so level-gated constants do not leak into lower-depth searches.
- Depth 4 keeps the existing 107-item pruned basis; depth 5 and higher include the ten new constants, giving a 117-item pruned basis.
