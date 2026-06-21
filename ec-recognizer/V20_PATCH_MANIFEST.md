# EC atlas v20 patch

Changed from v19:

- Removes the EC atlas brand/top title bar while preserving search, controls, and guide.
- Replaces full hover/detail/search database loading with id-based curve shards and hashed j/discriminant indexes.
- Keeps JS BigInt cubic recognition as the search engine.
- Adds mobile pointer/pinch handling and prevents browser page zoom inside the atlas.
- Adds a continuously rotating selected-star marker.
- Keeps v19 static tiles and all existing mathematical/detail functionality, but loads heavy C-isogeny data lazily after the main detail panel is visible.
