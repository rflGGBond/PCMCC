# MAPCMCC: Multi-Agent Progressive Community Merging Cooperative Coevolution

This package implements the MAPCMCC framework, refactoring the original PCMCC algorithm into a modular, agent-based architecture.

## Structure

### `core/`
Contains pure functional logic extracted from the original algorithm.
- `evaluator.py`: DPADV calculation, positive/negative score logic.
- `graph_ops.py`: Community detection and graph manipulation.
- `evolution.py`: Evolutionary operations (crossover, mutation) for subpopulations.
- `merger.py`: Logic for merging communities.

### `environment/`
Manages the state of the system.
- `env.py`: The `PCMCCEnvironment` class acting as the interface for Agents. Handles global state (Graph, SN) and execution steps.
- `community.py`: The `Community` class managing local state, history, and observation generation.

### `agents/`
Defines the Agent interfaces and implementations.
- `base.py`: Abstract base class for Agents.
- `community_agent.py`: Agent responsible for local parameter tuning and candidate seed generation.
- `meta_agent.py`: Global controller responsible for parameter baselines and merge suggestions.

### `utils/`
- `types.py`: Data structures (Dataclasses) for Observations and Actions.

## Usage

Run the main entry point:
```bash
python mapcmcc/main.py
```

## Integration with LLM
To connect real LLMs:
1. Modify `agents/community_agent.py` and `agents/meta_agent.py`.
2. Replace the rule-based logic in `get_action` with API calls to an LLM provider.
3. Serialize the `observation` object to JSON to pass as the prompt context.
