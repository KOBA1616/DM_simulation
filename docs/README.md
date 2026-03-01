# Duel Masters AI Simulator Documentation

Welcome to the documentation for the Duel Masters AI Simulator project. This documentation is organized into the following sections:

## 📚 [Architecture](architecture/)
High-level system design and core concepts.

- [System Overview](architecture/system_overview.md) - Detailed system architecture (Japanese).
- [Core Concepts](architecture/core_concepts.md) - Game engine core concepts and refactor notes.
- [Project Goals](architecture/project_goals_jp.md) - Project objectives and high-level introduction (Japanese).
- [Intermediate Representation](architecture/mid_representation.md) - Design for the AI's intermediate state representation.

## ⚙️ [Systems](systems/)
Detailed documentation for specific engine subsystems.

- **[Game Flow](systems/flow/)**: Documentation on how game states transition and commands are generated.
- **[Commands](systems/commands/)**: The Command System infrastructure (CommandDef, Instructions).
- **[AI](systems/ai/)**: AI agent design, MCTS implementation, and requirements.
- **[Mechanics](systems/mechanics/)**: Core game mechanics (Battle System, Shield System, etc.).
    - [Spell Execution QA](systems/mechanics/spells/SPELL_EXECUTION_TIMING_QA.md) - QA on spell timing.
    - [Spell Replacement Quick Ref](systems/mechanics/spells/SPELL_REPLACEMENT_QUICK_REF.md)
- **[Rules](systems/rules/)**: Rule implementation details and restrictions.
- **[Native Bridge](systems/native_bridge/)**: Documentation for the C++ engine to Python binding (`dm_ai_module`).

## 📖 [Guides](guides/)
Developer guides, setup instructions, and testing.

- [Setup Guide](guides/setup.md) - Environment setup and installation.
- [Testing Guide](guides/testing/headless_setup.md) - Headless testing setup.
- [Planning](guides/planning.md) - Development planning documents.
- [Repository Policy](guides/repository_policy.md) - Guidelines for repository management.
- [Module Loading](guides/module_loading.md) - Explanation of how the native module is loaded.

## 🗂 [Reference](reference/)
API specifications and cheatsheets.

- **[Specs](reference/specs/)**: Detailed specifications for Game Engine, AI System, and Card Editor.
- **[Features](reference/features/)**: Specific feature documentation (e.g., Replace Card Move).
- [Spell Execution QA](systems/mechanics/spells/SPELL_EXECUTION_TIMING_QA.md) (See Mechanics)

## 🏛 [Archive](archive/)
Legacy documentation, old design docs, and project history.

- [Migration Guides](archive/migration/) - Historical migration plans (e.g., Action to Command system).
- [Troubleshooting](archive/troubleshooting/) - Legacy troubleshooting guides.
- [Legacy Engine Architecture](archive/legacy_engine_architecture.md) - Older English architecture overview.
- [Project History](archive/project_history/) - Changelogs, status updates, and archive notes.
- [Research](archive/research/) - Presentations and research materials.

## 🤖 [Agent Policy](AGENTS.md)
- Guidelines for AI Agents working on this repository.
