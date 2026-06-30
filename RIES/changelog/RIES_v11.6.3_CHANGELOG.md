# RIES v11.6.3 Changelog

- Added 12 level-5-only low-precision linear-combination constants.
- Added real and imaginary parts of PolyGamma[0, Exp[2 Pi I/k]] for k = 8, 6, 4, 3.
- Added four alternating logarithmic NSum constants beginning at n = 2.
- Marked the new constants as level-5-only (`minLevel: 5`).
- Preserved the duplicate value Im PolyGamma[0, Exp[2 Pi I/4]] by allowing dependent level-5 additions to remain in the basis.
- Bumped RIES page/cache version to v11.6.3.
