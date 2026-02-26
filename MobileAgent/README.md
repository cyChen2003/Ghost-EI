# MobileAgent Usage Guide

MobileAgent is a multi-model reasoning agent that uses separate models for understanding, decision-making, reflection, and judgment.

📖 [Back to Main README](../README.md)

## 📋 Table of Contents

- [Overview](#overview)
- [Configuration](#configuration)
- [Running MobileAgent](#running-mobileagent)
- [Output](#output)

---

## Overview

MobileAgent uses a multi-step reasoning approach:

1. **Understanding**: Uses `caption_model` to understand screenshots and generate actions
2. **Decision Making**: Uses `caption_model` to make action decisions
3. **Reflection**: Uses `reflect_model` to reflect on action success/failure (if enabled)
4. **Judgment**: Uses `judge_model` to evaluate task completion

### Key Features

- Multi-model architecture for complex reasoning
- Reflection mechanism for error recovery
- Memory system for task planning
- Support for overlay and SMS attack scenarios

---

## Configuration

Edit `config.yaml` in the project root:

```yaml
# ============================================================================
# ENV SECTION: Configuration for MobileAgent (used by run.py)
# ============================================================================
env:
  adb_path: ""  # Path to adb executable
  API_url: "http://your-api-server/v1/"  # API endpoint for models
  token: "your-api-key"  # API key/token
  caption_call_method: "api"  # "api" or "local" - how to call caption_model
  caption_model: "gpt-4o"  # Model for action generation
  reflect_model: "gpt-4o"  # Model for reflecting on action success/failure
  judge_model: "gpt-4o"  # Model for final task completion judgment
  qwen_api_key: ""  # Qwen API key (if using Qwen models)
  add_info: 'If you want to tap an icon of an app, use the action "Open app". If you want to exit an app, use the action "Home"'
  reflection_switch: false  # Enable reflection mode
  memory_switch: true  # Enable memory mode
  reasoning_switch: false  # Enable reasoning mode
  device: "cuda"  # Device for local models
  temp_dir: "temp"  # Temporary files directory
  screenshot_dir: "screenshot"  # Screenshot storage
  file_dir: "./files"  # Test files directory
  seed: 1234  # Random seed

runtime:
  dataset_path: "./datasets/GhostEI.jsonl"  # Path to test dataset
  output_dir: null  # Output directory (auto-generated if null)
  thinking_output_path: null  # Thinking log path (auto-generated if null)
  judgement_dir: "judgement"  # Judgment output directory

hooks:
  overlay:
    component: "com.example.myapplication/.AdbReceiver"
    action: "com.example.broadcast.UPDATE_POPUP"
    server_hostport: "http://10.0.2.2:8000"
```

### Key Configuration Points

1. **API_url**: 
   - Should end with `/v1/` (will be auto-normalized to `/v1/chat/completions`)
   - Can use OpenAI-compatible APIs (GPT-4o, Qwen-VL, etc.)

2. **adb_path**: 
   - Full path to `adb.exe` on Windows
   - Example: `"path/to/adb"`

3. **Models**: 
   - `caption_model`: Used for action generation in the main loop
   - `reflect_model`: Used for reflecting on action outcomes (if `reflection_switch: true`)
   - `judge_model`: Used for final task completion judgment

4. **Switches**:
   - `reflection_switch`: Enable/disable reflection mechanism
   - `memory_switch`: Enable/disable memory system
   - `reasoning_switch`: Enable/disable reasoning mode (uses different API call)

---

## Running MobileAgent

### Basic Usage

```bash
conda activate safeagent
python run.py
```

### Prerequisites

Before running, ensure:

1. ✅ Android device/emulator is connected (`adb devices`)
2. ✅ `config.yaml` is properly configured
3. ✅ API credentials are set (`API_url` and `token`)
4. ✅ Dataset file exists at the configured path
5. ✅ Required apps are installed on the device

### Environment Variables

You can override config values using environment variables:

```bash
# Override API URL
export OPENAI_API_URL="http://your-server/v1/"

# Override API key
export OPENAI_API_KEY="your-key"

# Override config file path
export GHOST_EI_CONFIG="path/to/config.yaml"
```

---

## Output

Results are saved in the following locations:

### Output Directory

- **Location**: `{output_dir}/` (auto-generated as `{caption_model}-{reflection_switch}` if not specified)
- **Contents**: 
  - `{scenario_id}/`: Screenshots and judgment results for each scenario

---

## Related Documentation

- [Main README](../README.md) - Environment setup and overview
- [GUI Agent Documentation](../gui_agent/README.md) - GUI Agent usage guide
