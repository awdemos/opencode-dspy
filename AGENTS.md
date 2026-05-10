# AGENTS.md - OpenCode DSPy Integration

## Project Overview

Dual-component repository:
1. **Session Logger Plugin** (TypeScript) - `.opencode/plugin/session-logger.ts` - captures OpenCode sessions for training data
2. **DSPy Training Pipeline** (Python) - `dspy-trainingv2/` - optimizes agent prompts using collected session data

## Repository Structure

```
opencode-dspy/
├── .opencode/
│   └── plugin/
│       └── session-logger.ts          # OpenCode plugin (auto-loads)
│   └── package.json                   # Plugin deps: @opencode-ai/plugin
├── .opencode-logs/                    # Generated training data (gitignored)
│   ├── dspy-*.json                    # Successful sessions only
│   ├── session-*.json                 # All sessions
│   └── plugin.log                     # Plugin activity log
├── dspy-trainingv2/                   # Python DSPy pipeline
│   ├── cli.py                         # Main CLI entry point
│   ├── test_pipeline.py               # Pipeline validation
│   ├── config/
│   │   ├── default.yaml               # Default config
│   │   ├── openai-example.yaml
│   │   └── openai-compatible-example.yaml
│   ├── requirements.txt
│   └── src/                           # Source modules
│       ├── data/session_parser.py
│       ├── data/example_builder.py
│       ├── dspy_modules/
│       ├── evaluation/metrics.py
│       ├── optimization/optimizer.py
│       └── export/opencode_exporter.py
├── README.md
└── DSPY_PLUGIN_DOCUMENTATION.md       # Plugin technical docs
```

## Development Commands

### Plugin (TypeScript)
- No build step - plugin auto-loads when OpenCode restarts
- Dependencies managed via `npm install` in `.opencode/`
- Check plugin status: `tail -f .opencode-logs/plugin.log`

### Training Pipeline (Python)
- **Install**: `cd dspy-trainingv2 && pip install -r requirements.txt`
- **Validate setup**: `python cli.py validate`
- **Test pipeline**: `python test_pipeline.py`
- **Run optimization**: `python cli.py train --experiment-name <name>`
- **Clear DSPy cache**: `python cli.py clear-cache`

### Config Overrides
- Copy example configs: `cp config/openai-example.yaml config/local.yaml`
- `config/local.yaml` and `config/secrets.yaml` are gitignored - use for local overrides

## Environment Setup

- API keys via `.env` file or environment variables
- Teacher model API key required (OpenAI: `OPENAI_API_KEY`, Anthropic: `ANTHROPIC_API_KEY`)
- Student model defaults to local Ollama (no API key needed)
- Ollama endpoint: `http://localhost:11434/v1`

## Key Constraints

- **No build system**: No Makefile, no pyproject.toml, no setup.py
- **No test suite**: Only `test_pipeline.py` for validation
- **No linting config**: black and ruff in requirements.txt but no config files
- **Plugin is OpenCode-specific**: Depends on `@opencode-ai/plugin` package
- **Session logs contain sensitive data**: `.opencode-logs/` and `data/*.json` are gitignored
- **DSPy cache**: `.dspy_cache/` is gitignored
- **Python 3.8+ required**, Node.js for plugin development

## Working with Session Data

- Successful sessions saved to `.opencode-logs/dspy-*.json`
- Copy session logs to `dspy-trainingv2/data/` for training
- Minimum 10 examples required (config: `min_examples: 10`)
- Quality thresholds: `min_correctness: 0.8`, `min_efficiency: 0.3`

## Plugin Architecture

- Hooks: `event`, `tool.execute.before`, `tool.execute.after`
- Captures: tool calls, project context, LSP diagnostics, git status, outcome metrics
- Auto-saves every 5 updates and on session idle
- Success criteria: no errors, ≥1 tool used, ≥2 messages, <5 min duration

## Useful References

- `DSPY_PLUGIN_DOCUMENTATION.md` - Complete plugin technical documentation
- `dspy-trainingv2/README.md` - Training pipeline documentation
- `dspy-trainingv2/TROUBLESHOOTING.md` - Common issues and solutions
- `dspy-trainingv2/QUICKSTART.md` - Quick start guide
