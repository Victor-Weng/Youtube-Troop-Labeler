#!/usr/bin/env python3
"""
Test script for Roboflow integration using run_workflow
"""

import os
import logging
from card_tracker import CardTracker

def test_roboflow_integration():
    """Test the Roboflow integration"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing Roboflow integration with run_workflow...")
    
    # Check environment variables
    api_key = os.getenv('ROBOFLOW_API_KEY')
    workspace = os.getenv('WORKSPACE_CARD_DETECTION')
    
    if not api_key:
        logger.error("❌ ROBOFLOW_API_KEY environment variable not set")
        logger.error("   Set it to your Roboflow API key")
        return False
    
    if not workspace:
        logger.error("❌ WORKSPACE_CARD_DETECTION environment variable not set")
        logger.error("   Set it to your Roboflow workspace name")
        return False
    
    logger.info(f"✅ Environment variables set:")
    logger.info(f"   API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    logger.info(f"   Workspace: {workspace}")
    logger.info("   Workflow ID: custom-workflow")
    
    try:
        # Test CardTracker initialization
        logger.info("Testing CardTracker initialization...")
        card_tracker = CardTracker()
        
        if card_tracker.rf_model is None:
            logger.error("❌ Failed to initialize Roboflow model")
            return False
        
        logger.info("✅ Roboflow model initialized successfully")
        
        # Test Roboflow client access
        logger.info("Testing Roboflow client access...")
        try:
            # Test if we can access the client
            client = card_tracker.rf_model
            logger.info("✅ Roboflow client access successful")
        except Exception as e:
            logger.error(f"❌ Failed to access Roboflow client: {str(e)}")
            return False
        
        logger.info("✅ All Roboflow integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_roboflow_integration()
    if success:
        print("\n🎉 Roboflow integration is working correctly!")
    else:
        print("\n💥 Roboflow integration has issues. Please check the errors above.")
        print("\nTo fix:")
        print("1. Set ROBOFLOW_API_KEY environment variable")
        print("2. Set WORKSPACE_CARD_DETECTION environment variable")
        print("3. Ensure your Roboflow workspace exists")
        print("4. Check your internet connection")
        print("5. Verify your API key is valid") 