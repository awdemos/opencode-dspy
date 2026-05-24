# OpenCode DSPy Integration

A comprehensive system for capturing OpenCode coding sessions and training DSPy models to optimize agent prompts.

## Overview

This project consists of two main components:

1. **Session Logger Plugin** (TypeScript) - Captures high-quality training data from OpenCode sessions
2. **DSPy Training Pipeline** (Python) - Optimizes agent prompts using collected data

## Improvements Over Upstream

### Code Quality
- **Type Safety** — Replaced all `any` types in the TypeScript plugin with proper types (`Record<string, unknown>`, specific message types)
- **No Type Assertions** — Removed Python `as any` cast in metrics module
- **Clean Architecture** — Extracted cache bypass workaround into reusable helper functions (`_bypass_dspy_cache()`, `_restore_temperature()`)

### Testing
- **109 Unit Tests** — Comprehensive test coverage for all core modules:
  - `test_session_parser.py` (24 tests) — Session parsing and filtering
  - `test_example_builder.py` (13 tests) — Example formatting and batch building
  - `test_metrics.py` (23 tests) — All metric functions and edge cases
  - `test_optimizer.py` (14 tests) — Cache bypass, LM configuration, score extraction
  - `test_exporter.py` (16 tests) — Export formats and template generation
- **CI/CD Ready** — Tests run in isolated containers via Dagger

### CI/CD Pipeline (Dagger)
Reproducible containerized workflows:
```bash
# Run tests in container
dagger call test --source .

# Validate configuration
dagger call validate --source .

# Run full pipeline
dagger call all --source .

# Train with API key
dagger call train --source . --experiment-name my-run
```

**Why Dagger?**
- Same behavior locally and in CI
- No "works on my machine" issues
- Dependency caching via `uv` and cache volumes
- Secret management for API keys

### Configuration
- **No Hardcoded Paths** — Removed `/home/alan/opencode` references; source path is now optional and configurable
- **Portable** — Works on any machine without manual path adjustments

## Quick Start

### 1. Capture Training Data

The plugin at `.opencode/plugin/session-logger.ts` automatically captures your OpenCode sessions.

**Setup:**
```bash
# Plugin auto-loads on OpenCode restart
# Look for: "SessionLogger: Initialized"
```

**What Gets Captured:**
- Complete tool execution traces (read, edit, write, bash commands)
- Project context (files, LSP diagnostics, git status)
- Success metrics and quality scores
- Agent/model metadata
- Conversation history

**Output Files:**
```bash
.opencode-logs/
├── dspy-*.json        # Training data (successful sessions only)
├── session-*.json     # Raw session logs (all sessions)
└── plugin.log         # Activity log
```

### 2. Train DSPy Models

Once you've collected 50-100 successful examples:

```bash
cd dspy-trainingv2
pip install -r requirements.txt   # or: uv pip install -r requirements.txt
cp config/openai-example.yaml config/local.yaml  # Add your API keys
python cli.py train --experiment-name my-run
```

See [`dspy-trainingv2/README.md`](dspy-trainingv2/README.md) for detailed training instructions.

## Features

### Session Logger

- **Smart Filtering** — Only saves successful sessions for training
- **Complete Tool Traces** — Every action with args and results
- **Rich Context** — Project files, LSP errors, git state
- **Quality Metrics** — Correctness, efficiency, minimal edits
- **Auto-Save** — Saves every 5 updates and on session idle

**Success Criteria:**
Sessions are saved for training ONLY if:
- ✅ Task completed successfully (no errors)
- ✅ At least one tool was used
- ✅ Has real conversation (≥2 messages)
- ✅ Completed in reasonable time (<5 minutes)

### DSPy Training

- **Flexible Data Loading** — Automatically loads session logs
- **Multiple Optimizers** — MIPROv2, COPRO, BootstrapFewShot
- **Custom Metrics** — Success rate, efficiency, correctness
- **OpenCode Integration** — Exports optimized prompts ready to use

## Project Structure

```
opencode-dspy/
├── .opencode/
│   └── plugin/
│       └── session-logger.ts          # Session capture plugin
├── .opencode-logs/                    # Generated training data
├── dspy-trainingv2/                   # DSPy optimization pipeline
│   ├── tests/                         # 109 unit tests
│   ├── src/                           # Source modules
│   ├── config/                        # YAML configurations
│   ├── cli.py                         # CLI entry point
│   └── requirements.txt               # Dependencies
├── dagger/                            # Dagger CI/CD module
│   └── main.go                        # Pipeline definitions
├── dagger.json                        # Dagger module config
├── README.md                          # This file
├── DSPY_PLUGIN_DOCUMENTATION.md       # Detailed plugin documentation
└── example-dspy-enhanced-output.json  # Example output format
```

## Session Data Format

Each successful session creates a JSON file with this structure:

```json
{
  "session": "ses_123abc",
  "outcome": {
    "success": true,
    "metrics": {
      "filesModified": 2,
      "timeToCompletion": 45.2,
      "toolCallCount": 5,
      "tokenCost": { "input": 1842, "output": 326 }
    },
    "evaluation": {
      "correctness": 1.0,
      "efficiency": 0.89
    }
  },
  "examples": [
    {
      "input": {
        "task": "User's request",
        "context": {
          "workingDirectory": "/path",
          "projectType": "typescript",
          "relevantFiles": [...],
          "lspDiagnostics": { "errors": [...] },
          "gitStatus": { "branch": "main" }
        },
        "conversationHistory": [...]
      },
      "actions": [
        {
          "step": 1,
          "tool": "read",
          "args": { "filePath": "..." },
          "result": "...",
          "success": true
        }
      ],
      "output": {
        "response": "Assistant's explanation"
      },
      "outcome": { /* metrics */ },
      "agent": {
        "model": "anthropic/claude-sonnet-4-5",
        "promptTokens": 1842
      }
    }
  ]
}
```

See [`example-dspy-enhanced-output.json`](example-dspy-enhanced-output.json) for a complete example.

## Using DSPy with Session Data

### Basic Example

```python
import json
from dspy import Example

# Load training data
with open('.opencode-logs/dspy-ses_123.json') as f:
    data = json.load(f)

# Only use successful sessions
if data['outcome']['success']:
    for ex in data['examples']:
        example = Example(
            task=ex['input']['task'],
            context=ex['input']['context'],
            actions=ex['actions'],
            response=ex['output']['response']
        ).with_inputs('task', 'context')
```

### Training an Agent

```python
import dspy
from dspy import ChainOfThought
from dspy.teleprompt import BootstrapFewShot

# Define signature
class CodingTask(dspy.Signature):
    """Solve a coding task by analyzing context and taking actions."""
    task = dspy.InputField(desc="The coding task to complete")
    context = dspy.InputField(desc="Project context")
    response = dspy.OutputField(desc="Explanation of what was done")

# Create agent
class CodingAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_solution = ChainOfThought(CodingTask)
    
    def forward(self, task, context):
        return self.generate_solution(task=task, context=context)

# Load examples and optimize
# ... (see dspy-trainingv2/ for complete pipeline)
```

For complete DSPy integration examples, see the [`dspy-trainingv2/`](dspy-trainingv2/) directory.

## Monitoring

### Check Plugin Status
```bash
tail -f .opencode-logs/plugin.log
```

### Count Successful Sessions
```bash
ls .opencode-logs/dspy-*.json | wc -l
```

### View Metrics
```bash
cat .opencode-logs/dspy-*.json | jq '.outcome.metrics'
```

### Check Success Rate
```bash
grep "SUCCESS=" .opencode-logs/plugin.log
```

## Troubleshooting

### No Training Files Created?
- Check `.opencode-logs/plugin.log` for errors
- Ensure tasks complete successfully
- Verify plugin initialized (look for "Initialized" message)

### Low Success Rate?
- Tasks may be too complex
- Check for errors in sessions
- Review plugin.log for failure reasons

### Training Issues?
See [`dspy-trainingv2/README.md`](dspy-trainingv2/README.md) troubleshooting section.

## Documentation

- **[DSPY_PLUGIN_DOCUMENTATION.md](DSPY_PLUGIN_DOCUMENTATION.md)** — Complete technical documentation for the session logger plugin
- **[dspy-trainingv2/README.md](dspy-trainingv2/README.md)** — Complete DSPy training pipeline documentation
- **[dspy-trainingv2/QUICKSTART.md](dspy-trainingv2/QUICKSTART.md)** — Quick start guide for training
- **[dspy-trainingv2/TROUBLESHOOTING.md](dspy-trainingv2/TROUBLESHOOTING.md)** — Common issues and solutions
- **[example-dspy-enhanced-output.json](example-dspy-enhanced-output.json)** — Example session output format

## Workflow

1. **Collect** — Use OpenCode for 1-2 weeks (aim for 50-100 examples)
2. **Filter** — Only successful examples are saved automatically
3. **Train** — Load into DSPy and optimize prompts
4. **Evaluate** — Measure improvements
5. **Iterate** — Continuously collect more data

## Best Practices

### Data Collection
- Complete 50-100 successful examples before training
- Include diverse tasks (bug fixes, features, refactoring)
- Ensure examples represent real usage patterns

### Training
- Start with light optimization mode (`auto_mode: "light"`)
- Use 20/80 train/validation split
- Monitor costs with teacher model API usage
- Validate improvements on held-out data

### Quality Control
- Only use `outcome.success === true` examples
- Filter by quality metrics (correctness ≥ 1.0, efficiency > 0.7)
- Remove outliers (very long or very short sessions)

## Technical Details

### Session Logger Plugin
- **Language:** TypeScript
- **Lines:** ~500
- **Hooks:** `event`, `tool.execute.before`, `tool.execute.after`
- **Output:** JSON files in `.opencode-logs/`

### DSPy Training Pipeline
- **Language:** Python 3.8+
- **Tests:** 109 unit tests (pytest)
- **Optimizers:** MIPROv2, COPRO, BootstrapFewShot
- **Teacher Models:** Anthropic Claude, OpenAI GPT-4
- **Target Models:** Any LLM (Ollama, OpenAI, Anthropic, etc.)
- **CI/CD:** Dagger (containerized, reproducible)

## Contributing

Contributions welcome! The project is structured to make it easy to:
- Add new metrics for evaluation
- Support additional optimizers
- Enhance context collection
- Improve data quality filtering

## License

CRAPL License — see [LICENSE](LICENSE) file

## Community

Built for the OpenCode and DSPy communities:
- **OpenCode:** https://opencode.ai
- **DSPy:** https://github.com/stanfordnlp/dspy

## Status

✅ **Production Ready**
- Plugin captures all critical information for DSPy
- Training pipeline tested and working (109 tests)
- Documentation complete
- CI/CD pipeline with Dagger
- Ready for community use

## Version

- **Session Logger:** v1.2.0
- **DSPy Training:** v2.0.0
- **Last Updated:** 2026-05-24
