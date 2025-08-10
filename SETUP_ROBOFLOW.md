# Roboflow Integration Setup Guide

## Overview

The Clash Royale Troop Annotation Tool now includes AI-powered card detection using Roboflow's `run_workflow` API. This provides enhanced accuracy for card recognition compared to the fallback template matching method.

## Prerequisites

1. **Roboflow Account**: Sign up at [https://app.roboflow.com](https://app.roboflow.com)
2. **API Key**: Get your API key from [https://app.roboflow.com/account/api](https://app.roboflow.com/account/api)
3. **Workspace**: Create or identify your Roboflow workspace name
4. **Custom Workflow**: Ensure you have a "custom-workflow" in your workspace

## Setup Steps

### 1. Set Environment Variables

#### Windows (PowerShell):
```powershell
$env:ROBOFLOW_API_KEY="your_api_key_here"
$env:WORKSPACE_CARD_DETECTION="your_workspace_name"
```

#### Windows (Command Prompt):
```cmd
set ROBOFLOW_API_KEY=your_api_key_here
set WORKSPACE_CARD_DETECTION=your_workspace_name
```

#### Linux/Mac:
```bash
export ROBOFLOW_API_KEY=your_api_key_here
export WORKSPACE_CARD_DETECTION=your_workspace_name
```

### 2. Create .env File (Alternative)

Create a `.env` file in your project root:
```
ROBOFLOW_API_KEY=your_api_key_here
WORKSPACE_CARD_DETECTION=your_workspace_name
```

### 3. Test the Integration

Run the test script to verify everything is working:
```bash
python test_roboflow.py
```

Expected output:
```
✅ Roboflow configuration is properly set
   API Key: abc123...xyz789
   Workspace: your_workspace_name
   Workflow ID: custom-workflow
✅ All Roboflow integration tests passed!
```

## How It Works

### With Roboflow Enabled:
1. **Card Detection**: Uses AI model for accurate card recognition
2. **Workflow Processing**: Sends images to Roboflow's custom workflow
3. **Real-time Results**: Processes detection results in real-time
4. **Fallback Protection**: Automatically falls back to template matching if AI fails

### Without Roboflow (Fallback Mode):
1. **Template Matching**: Uses traditional computer vision techniques
2. **Color Analysis**: Analyzes pixel differences for troop detection
3. **Basic Recognition**: Limited but functional card identification

## Configuration Details

- **API Endpoint**: Roboflow cloud API
- **Workflow ID**: `custom-workflow` (hardcoded)
- **Model Type**: Custom card detection model
- **Response Format**: Nested predictions structure

## Troubleshooting

### Common Issues:

1. **"ROBOFLOW_API_KEY not set"**
   - Solution: Set the environment variable or create .env file

2. **"WORKSPACE_CARD_DETECTION not set"**
   - Solution: Set your Roboflow workspace name

3. **"Failed to initialize Roboflow model"**
   - Solution: Check API key validity and internet connection

4. **"Inference failed"**
   - Solution: Verify workspace exists and workflow is accessible

### Testing Commands:

```bash
# Test Roboflow configuration
python roboflow_config.py

# Test Roboflow integration
python test_roboflow.py

# Test fallback mode
python test_card_tracker.py

# Test basic components
python test_basic.py
```

## Performance Notes

- **AI Detection**: Higher accuracy, slower processing
- **Fallback Mode**: Lower accuracy, faster processing
- **Hybrid Approach**: Best of both worlds with automatic fallback

## Security

- API keys are stored in environment variables (not in code)
- No sensitive data is logged
- Fallback mode ensures tool works even without API access

## Support

If you encounter issues:
1. Check environment variables are set correctly
2. Verify your Roboflow workspace and workflow exist
3. Test with the provided test scripts
4. Check internet connectivity and API key validity 