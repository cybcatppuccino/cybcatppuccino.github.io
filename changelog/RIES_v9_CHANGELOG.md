# RIES v9

- Keeps the v8.8 feature set and focuses on integer Continue stability.
- Adds safer rational-power verification so medium integer searches do not build enormous BigInt powers repeatedly.
- Keeps previously displayed integer results on screen during Continue and reuses cached factor/database/shortform rows.
- Adds tighter responsive slices and clearer progress labels for exact shortform and <=10^8 exhaustive passes.
- Removes visible release-note text from the RIES results header and keeps README concise.
