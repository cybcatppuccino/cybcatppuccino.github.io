# RIES v11.6.4 Changelog

- Added four level-5-only low-precision linear-combination constants:
  - `Sum[(-1)^n Log[n]/n, {n,1,Infinity}]`
  - `Sum[(-1)^n Log[n]/n^2, {n,1,Infinity}]`
  - `Sum[Log[n]/n^2, {n,1,Infinity}]`
  - `Sum[Log[n]/n^3, {n,1,Infinity}]`
- Merged the duplicate level-5 value `Im PolyGamma[0, Exp[2 Pi I/4]]` with the existing `Sum[1/(n^2+1), {n,0,Infinity}]` constant, keeping one numeric basis row and exposing both labels.
- Bumped RIES page/cache version to v11.6.4.
- Added v11.6.4 packaging and level-5 linear-combo regression tests.
