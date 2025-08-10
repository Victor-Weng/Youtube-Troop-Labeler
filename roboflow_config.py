"""
Roboflow Configuration Template

To use your Roboflow model, you need to set these environment variables:

1. Create a .env file in your project root with:
   ROBOFLOW_API_KEY=your_roboflow_api_key_here
   WORKSPACE_CARD_DETECTION=your_workspace_name_here

2. Or set them directly in your system:
   - Windows: set ROBOFLOW_API_KEY=your_key
   - Linux/Mac: export ROBOFLOW_API_KEY=your_key

3. Get your API key from: https://app.roboflow.com/account/api

4. Your workspace name is the name of your Roboflow workspace
5. The tool will use the "custom-workflow" workflow ID for inference
"""

import os

def check_roboflow_config():
    """
    Check if Roboflow configuration is properly set
    """
    api_key = os.getenv('ROBOFLOW_API_KEY')
    workspace = os.getenv('WORKSPACE_CARD_DETECTION')
    
    if not api_key:
        print("❌ ROBOFLOW_API_KEY environment variable not set")
        print("   Get your API key from: https://app.roboflow.com/account/api")
        return False
    
    if not workspace:
        print("❌ WORKSPACE_CARD_DETECTION environment variable not set")
        print("   Set this to your Roboflow workspace name")
        return False
    
    print("✅ Roboflow configuration is properly set")
    print(f"   API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    print(f"   Workspace: {workspace}")
    print("   Workflow ID: custom-workflow")
    return True

if __name__ == "__main__":
    check_roboflow_config() 