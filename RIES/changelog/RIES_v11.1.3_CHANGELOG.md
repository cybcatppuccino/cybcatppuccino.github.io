# RIES v11.1.3 changelog

- Moved the low-precision algebraic relation search to the final module slot, after RIES equation, log relations, Möbius relations, and constant database matches.
- Added a cooperative asynchronous low-precision algebraic relation path (`relationCandidatesAsync` / `pslqAlgebraicRowsAsync`) so the UI can repaint between PSLQ/LLL batches and the SO(4) tesseract animation remains responsive.
- Kept the existing synchronous `relationCandidates()` path for high-precision algebraic reconstruction and other legacy callers.
