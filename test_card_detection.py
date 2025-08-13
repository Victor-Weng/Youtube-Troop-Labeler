import os
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def setup_card_roboflow():
    """Initialize Roboflow client for card detection"""
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY environment variable is not set. Please check your .env file.")

    return InferenceHTTPClient(
        api_url="http://localhost:9001",
        api_key=api_key
    )


def test_card_detection(image_path):
    """Test card detection on a single image"""
    try:
        print("🔧 Step 1: Setting up Roboflow client...")

        # Setup Roboflow client
        card_model = setup_card_roboflow()
        print("✅ Roboflow client created successfully")

        # Get workspace name
        workspace_name = os.getenv('WORKSPACE_CARD_DETECTION')
        if not workspace_name:
            raise ValueError(
                "WORKSPACE_CARD_DETECTION environment variable is not set. Please check your .env file.")

        print(f"🔧 Step 2: Configuration check...")
        print(f"  - Image: {image_path}")
        print(f"  - Workspace: {workspace_name}")
        print(f"  - API URL: http://localhost:9001")
        print(f"  - Workflow ID: crbot")

        # Check if localhost server is running
        print(f"🔧 Step 3: Testing localhost connection...")
        try:
            import requests
            response = requests.get("http://localhost:9001", timeout=5)
            print(f"✅ Localhost server responding: {response.status_code}")
        except Exception as e:
            print(f"❌ Localhost server NOT responding: {e}")
            print("💡 Make sure to run: inference server start")
            return None, 0.0

        # Check image file details
        print(f"🔧 Step 4: Image file check...")
        if os.path.exists(image_path):
            file_size = os.path.getsize(image_path)
            print(f"✅ Image exists, size: {file_size} bytes")
        else:
            print(f"❌ Image file not found: {image_path}")
            return None, 0.0

        print(f"🔧 Step 5: Running inference...")
        # Run inference
        results = card_model.infer(
            image_path,
            model_id="cards-clash-royale-4rn4u/1")

        print("🔧 Step 6: Analyzing results...")
        print(f"  - Results type: {type(results)}")
        print(
            f"  - Results length: {len(results) if isinstance(results, (list, dict)) else 'N/A'}")
        print(f"  - Results content: {results['top']}")

        # Check if results is empty
        if not results:
            print("❌ Results is empty/None!")
            return "Unknown", 0.0

        # Parse results (same logic as detector.py)
        predictions = []
        if isinstance(results, list) and results:
            print(f"🔧 Step 7: Parsing list results...")
            first_result = results[0]
            print(f"  - First result type: {type(first_result)}")
            print(
                f"  - First result keys: {first_result.keys() if isinstance(first_result, dict) else 'Not a dict'}")

            preds_dict = first_result.get("predictions", {})
            print(f"  - Predictions dict: {preds_dict}")
            print(f"  - Predictions dict type: {type(preds_dict)}")

            if isinstance(preds_dict, dict):
                predictions = preds_dict.get("predictions", [])
                print(f"  - Extracted predictions: {predictions}")
                print(
                    f"  - Predictions count: {len(predictions) if predictions else 0}")

        if predictions:
            card_name = predictions[0]["class"]
            confidence = predictions[0].get("confidence", 1.0)
            print(
                f"\n✅ SUCCESS: Detected card: {card_name} (confidence: {confidence})")
            return card_name, confidence
        else:
            print(f"\n❌ NO PREDICTIONS: No card detected in {image_path}")
            print("💡 This could mean:")
            print("   - Wrong workspace/workflow ID")
            print("   - Image quality issues")
            print("   - Model not trained on this type of image")
            return "Unknown", 0.0

    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0


if __name__ == "__main__":
    print("=== CARD DETECTION DEBUG TEST ===")

    # Check what images are available
    print("🔧 Checking available screenshot files...")
    if os.path.exists("screenshots"):
        files = os.listdir("screenshots")
        print(f"Found {len(files)} files in screenshots/:")
        for file in files:
            print(f"  - {file}")
    else:
        print("❌ screenshots/ directory not found")

    # Test images to try
    test_images = [
        "screenshots/ally_card_1.png",
        "screenshots/ally_card_2.png",
        "screenshots/enemy_card_1.png",
        "screenshots/enemy_card_2.png"
    ]

    for test_image in test_images:
        print(f"\n{'='*50}")
        print(f"Testing: {test_image}")

        if not os.path.exists(test_image):
            print(f"❌ File not found: {test_image}")
            continue

        # Run the test
        card_name, confidence = test_card_detection(test_image)

        if card_name:
            print(f"🎯 Result: {card_name} ({confidence})")
        else:
            print("💥 Test failed")
            break  # Stop on first error to see what's wrong
