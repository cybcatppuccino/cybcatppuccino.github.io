# Review of `mcts-chess-master.zip`

## What it contains

The supplied project is a Stanford class project described as PPO on minichess. It contains Python game logic, Ray/RLlib training infrastructure, PyTorch model weights, old MCTS experiments, and implementations of several chess variants.

## Why it was not directly integrated

### Different Gardner rules

The supplied README and Gardner logic do not implement the same strict rules used by this web project. The archive discusses or implements:

- pawn double moves;
- en passant state;
- castling-related support;
- termination by capturing a king rather than standard legal check/checkmate;
- incremental rule addition;
- promotion behavior that is not the UI's explicit Q/R/B/N choice.

Its logged games and trained model therefore cannot be treated as valid ground truth for this engine.

### Runtime mismatch

The project relies on Python, NumPy, Ray/RLlib, and PyTorch. Shipping it would break the current dependency-free, local JavaScript/Web Worker architecture and would not provide a directly callable browser model.

### Licensing uncertainty

No clear licence file was present in the supplied archive. Its code and model weights were therefore treated as review-only material.

### No demonstrated strength gain

A policy trained under different legality and terminal conditions can recommend illegal or strategically meaningless moves under strict Gardner Chess. Integrating such a model merely because it exists would risk reducing strength and correctness.

## Ideas retained at the design level

No source code or model data was copied. Only general, independently implemented ideas were considered:

- always mask choices through the strict legal move generator;
- separate finite game-playing search from continuous analysis;
- create lower difficulty levels by controlled exploration among near-best legal lines rather than by deliberately corrupting evaluation;
- keep evaluation/testing data separate from the live engine.

These ideas are common algorithmic patterns and are implemented independently in Orion JS 5.0.
