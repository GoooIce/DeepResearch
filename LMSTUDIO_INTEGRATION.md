# LMStudio SDK Integration

This update integrates the LMStudio SDK into the DeepResearch project, allowing you to use LMStudio as the model provider instead of OpenAI-compatible APIs.

## What Changed

### 1. New LMStudio LLM Implementation

- **File**: `WebAgent/WebDancer/demos/llm/lmstudio_llm.py`
- **Class**: `TextChatAtLMStudio`
- **Registration**: Available as `'lmstudio'` model type

This implementation provides:
- Native LMStudio SDK integration using `lmstudio.llm()` 
- Support for both streaming and non-streaming chat
- Parameter mapping from OpenAI-style to LMStudio format
- Proper error handling with LMStudio-specific exceptions

### 2. Enhanced React Agent

- **File**: `inference/react_agent.py`
- **Changes**: Added LMStudio SDK support with fallback to OpenAI-compatible API

The React Agent now:
- Tries LMStudio SDK first if configured
- Falls back to OpenAI-compatible endpoints if LMStudio is unavailable
- Maintains backward compatibility with existing setups

### 3. Updated Configuration

- **File**: `inference/run_multi_react.py`
- **Change**: Default model type changed from `'qwen_dashscope'` to `'lmstudio'`

### 4. Dependencies

- **File**: `requirements.txt`
- **Added**: `lmstudio==1.5.0`

## Usage

### For WebAgent Module

```python
from qwen_agent.llm.base import LLM_REGISTRY
from WebAgent.WebDancer.demos.llm.lmstudio_llm import TextChatAtLMStudio

# Create LMStudio LLM instance
cfg = {
    'model': 'your-model-name',
    'api_host': 'http://localhost:1234'  # Optional, uses default if not specified
}

llm = TextChatAtLMStudio(cfg)

# Use with qwen_agent framework
llm_cfg = {
    'model': 'your-model-name',
    'model_type': 'lmstudio',
    'api_host': 'http://localhost:1234'  # Optional
}
```

### For Inference Module

```python
from inference.react_agent import MultiTurnReactAgent

llm_cfg = {
    'model': 'your-model-name',
    'generate_cfg': {
        'temperature': 0.6,
        'top_p': 0.95,
        'max_tokens': 10000,
        'presence_penalty': 1.1
    },
    'model_type': 'lmstudio',
    'api_host': 'http://localhost:1234'  # Optional
}

agent = MultiTurnReactAgent(llm=llm_cfg)
```

### Running the Inference Script

```bash
# Set environment variables
export MODEL_PATH="your-model-name"
export DATASET="your_dataset_name" 
export OUTPUT_PATH="/your/output/path"

# Run with LMStudio (the new default)
python inference/run_multi_react.py --dataset "$DATASET" --output "$OUTPUT_PATH" --model "$MODEL_PATH"
```

## Configuration Options

### LMStudio-Specific Parameters

- `api_host`: LMStudio server URL (default: auto-detect local LMStudio)
- `model`: Model identifier in LMStudio

### Supported Generation Parameters

The implementation maps these OpenAI-style parameters to LMStudio:

- `temperature`: Controls randomness
- `top_p`: Nucleus sampling parameter  
- `max_tokens`: Maximum tokens to generate
- `presence_penalty`: Presence penalty for repetition
- `frequency_penalty`: Frequency penalty for repetition
- `stop`: Stop sequences

## Testing

Use the provided test script to verify the integration:

```bash
# Test without LMStudio server (basic import/registration test)
python test_lmstudio_integration.py --llm-only

# Test with running LMStudio server
python test_lmstudio_integration.py --api-host http://localhost:1234 --model "your-model-name"

# Test only React Agent integration
python test_lmstudio_integration.py --agent-only
```

## Prerequisites

1. **Install LMStudio**: Download and install LMStudio from https://lmstudio.ai/
2. **Load a Model**: Start LMStudio and load your desired model
3. **Enable API Server**: Start the local API server in LMStudio (usually on port 1234)

## Backward Compatibility

The integration maintains full backward compatibility:

- Existing OpenAI-compatible configurations continue to work
- The React Agent falls back to OpenAI-compatible APIs if LMStudio is unavailable
- All existing model types (`'oai'`, `'qwen_dashscope'`, etc.) remain functional

## Error Handling

The implementation includes robust error handling:

- LMStudio connection errors are caught and logged
- Automatic fallback to OpenAI-compatible APIs in React Agent
- Clear error messages guide troubleshooting

## Performance Considerations

- LMStudio SDK provides direct integration without HTTP overhead
- Streaming responses are efficiently handled
- Connection pooling and retry logic included

## Troubleshooting

### Common Issues

1. **"LM Studio is not reachable"**
   - Ensure LMStudio is running
   - Check that the API server is enabled in LMStudio
   - Verify the API host URL

2. **"Model not found"**
   - Ensure the model is loaded in LMStudio
   - Check the model name matches exactly

3. **Import errors**
   - Ensure `lmstudio` package is installed: `pip install lmstudio==1.5.0`
   - Check that all dependencies are satisfied

### Debug Mode

Enable debug logging to see detailed LMStudio interactions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will show LLM inputs, responses, and SDK calls for troubleshooting.