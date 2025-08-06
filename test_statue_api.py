#!/usr/bin/env python3
"""
Test script to test statue restoration API endpoint
"""
import requests
import base64
import json
from PIL import Image
import io

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (512, 512), color='blue')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def test_statue_restoration_api():
    """Test statue restoration API endpoint"""
    print("🧪 Testing statue restoration API endpoint...")
    
    # Create test image
    test_image = create_test_image()
    
    # Prepare restoration request
    payload = {
        "image": test_image,
        "prompt": "classical marble statue, restored, ancient sculpture",
        "guidance_scale": 7.5,
        "num_inference_steps": 20,  # Use fewer steps for testing
        "strength": 0.8
    }
    
    try:
        print("📡 Sending restoration request...")
        response = requests.post(
            "http://localhost:5002/api/statue-restoration/restore",
            json=payload,
            timeout=300  # Give more time for model loading
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Statue restoration successful!")
                print(f"📄 Has restored image: {'restored_image' in result}")
                print(f"📄 Has mask: {'mask_used' in result}")
                print(f"📄 Has comparison: {'comparison' in result}")
            else:
                print(f"❌ Restoration failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")

if __name__ == "__main__":
    test_statue_restoration_api()
