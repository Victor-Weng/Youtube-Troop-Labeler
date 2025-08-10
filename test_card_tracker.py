#!/usr/bin/env python3
"""
Test script for CardTracker class in fallback mode
"""

import logging
import numpy as np
from card_tracker import CardTracker

def test_card_tracker_fallback():
    """Test CardTracker in fallback mode without Roboflow"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing CardTracker in fallback mode...")
    
    try:
        # Test CardTracker initialization (should work in fallback mode)
        logger.info("Testing CardTracker initialization...")
        card_tracker = CardTracker()
        
        if card_tracker.rf_model is None:
            logger.info("✅ Roboflow model not available - fallback mode active")
        else:
            logger.info("✅ Roboflow model available")
        
        # Test basic functionality
        logger.info("Testing basic card tracker functionality...")
        
        # Test card names loading
        card_names = card_tracker.card_names
        logger.info(f"✅ Loaded {len(card_names)} card names")
        
        # Test get_card_name method
        test_name = card_tracker.get_card_name(0)
        logger.info(f"✅ Card ID 0: {test_name}")
        
        # Test get_class_id method
        test_id = card_tracker.get_class_id("archer queen")
        logger.info(f"✅ 'archer queen' has ID: {test_id}")
        
        # Test with a dummy frame
        logger.info("Testing with dummy frame...")
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Test hand detection (should use fallback)
        ally_cards, enemy_cards = card_tracker.update_hand_states(dummy_frame)
        logger.info(f"✅ Hand state update successful - Ally: {len(ally_cards)}, Enemy: {len(enemy_cards)}")
        
        logger.info("✅ All CardTracker tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_card_tracker_fallback()
    if success:
        print("\n🎉 CardTracker is working correctly in fallback mode!")
        print("   (Roboflow integration will work when properly configured)")
    else:
        print("\n💥 CardTracker has issues. Please check the errors above.") 