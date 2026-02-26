# GUI Agent Usage Guide

GUI Agents use specialized fine-tuned models for end-to-end GUI action generation. This guide covers AppAgent, TARS 1.5, and UI-TARS agents.

📖 [Back to Main README](../README.md)

## 📋 Table of Contents

- [Available Agents](#available-agents)
- [Configuration](#configuration)
- [Running GUI Agents](#running-gui-agents)
  - [Using Local Models](#using-local-models)
  - [Using Remote Models](#using-remote-models)
- [Output](#output)

---

## Available Agents


### 1. TARS 1.5 Agent (`test_loop_tars15.py`)

- **Model**: Qwen2.5 VL-based
- **Format**: TARS action format
- **Actions**: click(), long_press(), type(), scroll(), press_home(), press_back(), wait(), finished()

### 2. UI-TARS Agent (`test_loop_uitars.py`)

- **Model**: Qwen2VL-based (UI-TARS 7B SFT)
- **Format**: TARS action format
- **Actions**: Same as TARS 1.5

---

## Configuration

Edit `gui_agent/config.yaml`:

```yaml
# ============================================================================
# INFERENCE SECTION: Configuration for GUI Agents (used by test_loop_*.py)
# ============================================================================
inference:
  mode: "remote"  # "local" for local HuggingFace execution or "remote" for vLLM/OpenAI-compatible server
  remote_api_url: "https://your-vllm-server/v1/"  # API endpoint for GUI Agent's core model
  remote_api_key: null  # API key for GUI Agent's model
  remote_model: "/path/to/UI-TARS-1.5-7B"  # Core model for GUI action generation
  remote_temperature: 0.0  # Sampling temperature
  remote_top_p: null  # Nucleus sampling parameter
  remote_extra_params: {}  # Additional parameters for remote inference
  remote_timeout: 120.0  # HTTP timeout for remote requests
  remote_basic_user: null  # HTTP Basic auth username (if required)
  remote_basic_password: null  # HTTP Basic auth password (if required)
```

### Configuration Modes

#### Remote Mode (Default)

```yaml
inference:
  mode: "remote"
  remote_api_url: "https://your-server/v1/"
  remote_model: "your-model-name"
  remote_api_key: "your-api-key"  # Optional
```

#### Local Mode

```yaml
inference:
  mode: "local"
  # remote_* parameters are ignored in local mode
```

---

## Running GUI Agents

### Basic Usage

```bash
cd gui_agent
conda activate safeagent
python test_loop_tars15.py  # or test_loop_uitars.py
```

### Using Local Models

When using local models, specify the model path:

```bash
python test_loop_tars15.py \
  --inference-mode local \
  --model-path "/path/to/UI-TARS-7B-SFT" \
  --device cuda \
  --torch-dtype bfloat16
```

**Requirements for Local Mode:**
- Sufficient GPU memory (typically 16GB+ for 7B models)
- CUDA-enabled PyTorch installation
- Model files downloaded and accessible

### Using Remote Models

1. **Configure `gui_agent/config.yaml`**:

```yaml
inference:
  mode: "remote"
  remote_api_url: "https://your-server/v1/"
  remote_model: "your-model-name"
  remote_api_key: "your-key"  # If required
```

2. **Run the agent**:

```bash
python test_loop_tars15.py
```

The script will automatically use the configuration from `gui_agent/config.yaml`.

### Overriding Configuration

You can override config values via command line:

```bash
python test_loop_tars15.py \
  --remote-api-url "https://different-server/v1/" \
  --remote-model "different-model" \
  --remote-temperature 0.1
```

### Example Commands

**Local model with custom settings:**
```bash
python test_loop_tars15.py \
  --inference-mode local \
  --model-path "/path/to/model" \
  --device cuda \
  --max-new-tokens 256 \
  --base-data-path "../datasets/custom.jsonl"
```

**Remote model with overrides:**
```bash
python test_loop_tars15.py \
  --remote-api-url "https://new-server/v1/" \
  --remote-model "new-model" \
  --remote-temperature 0.2 \
  --log-dir "./custom_outputs"
```

---

## Output

Results are saved in the configured `log_dir` (default: `./outputs`):

### Output Structure

```
outputs/
├── {agent_name}_scenario_{id}_{timestamp}.log  # Execution log for each scenario
├── {agent_name}.jsonl                          # Summary results
├── thinking.jsonl                              # Thinking process (if available)
└── {scenario_id}/                              # Scenario-specific outputs
    ├── *.jpg                                   # Screenshots captured during execution
    └── ...
```

---

## Related Documentation

- [Main README](../README.md) - Environment setup and overview
- [MobileAgent Documentation](../MobileAgent/README.md) - MobileAgent usage guide
