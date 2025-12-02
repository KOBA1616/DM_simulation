POMDP Implementation Plan (Requirement 11)

Goal
----
Implement a maintainable POMDP inference component to support partial-observation reasoning for AI agents. Deliverables:
- A small, well-documented C++ API for belief/transition/observation models.
- Python bindings exposing the inference API for training and evaluation.
- Unit tests and a minimal example trace demonstrating belief updates.

High-level design
-----------------
1) Core concepts
   - Belief: a compact vector representing a probability distribution over hidden variables (e.g., opponent hand composition, unknown card instances).
   - Transition model: given an action and state, approximate how hidden state evolves.
   - Observation model: probability of observing visible events given a hidden state.
   - Inference/Update: Bayes-style update of the belief after each observation; optional particle-filter or parametric update.

2) Minimal API sketch
   - class `POMDPInference` (header-only stub exists as `src/ai/pomdp/pomdp.hpp`)
     - `void initialize(const std::map<uint16_t, CardDefinition>& card_db)`
     - `void update_belief(const GameState& observed_state)`
     - `std::vector<float> infer_action(const GameState& state)`
     - `std::vector<float> get_belief_vector() const`

3) Implementation plan (phased)
   Phase 0 — scaffolding (current)
     - Provide header-only stubs (done) and Python bindings (done).
     - Add pytest skeleton to ensure importability.
   Phase 1 — simple parametric belief
     - Represent belief as a set of per-card probabilities (expected counts in opponent hand/deck).
     - Implement a deterministic transition heuristic for draws/discards and an observation update for revealed cards.
     - Add tests verifying belief mass conservation and basic updates.
   Phase 2 — particle filter (optional)
     - Add a lightweight particle filter implementation to support richer dynamics and stochastic effects.
     - Provide seeding and resampling options; tests to validate convergence on small traces.
   Phase 3 — integration and performance
     - Expose batched inference APIs for use in self-play and MCTS.
     - Optimize representations and add benchmarks.

4) Tests and data
   - Unit tests for API surface and belief invariants.
   - Small example traces in `python/tests/data/` to drive integration tests.
   - CI updates will run the test suite (no build-system changes needed for header-only initial work).

Next immediate actions (what I'll do now)
---------------------------------------
- Add a short design doc (this file) — completed.
- Implement the next C++ header stubs for `Belief` and `Transition` (lightweight) and add a unit test skeleton (I will create stubs if you confirm).

Confirm how you'd like to proceed:
- "Implement Phase 1 (parametric belief)" — I will implement per-card probability belief, update rules, and tests.
- "Prepare docs + tests only" — I will expand docs and add more test skeletons before coding.
- "Merge current PR and start Phase 1 on main" — I will merge PR #3 and continue on `main`/`develop` as you prefer.
