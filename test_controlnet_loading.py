#!/usr/bin/env python3
"""
Test script to trigger ControlNet loading in Ancient Vision
"""
import requests
import base64
import json
from PIL import Image
import io

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (512, 512), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def test_controlnet_loading():
    """Test ControlNet loading by making a transformation request"""
    print("🧪 Testing ControlNet loading with art transformation...")
    
    # Create test image
    test_image = create_test_image()
    
    # Prepare transformation request
    payload = {
        "image": test_image,
        "prompt": "test transformation",
        "style": "classic_ukiyo",
        "strength": 0.7,
        "guidance_scale": 8.0,
        "num_inference_steps": 20,
        "model_id": "ukiyo_e_lora"
    }
    
    try:
        print("📡 Sending transformation request to trigger ControlNet loading...")
        response = requests.post(
            "http://localhost:5001/api/transform",
            json=payload,
            timeout=120  # Give time for model loading
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ Transformation successful - ControlNet should be loaded!")
                print(f"📄 Transformation completed in {result.get('processing_time', 'unknown')} seconds")
            else:
                print(f"❌ Transformation failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"❌ HTTP Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
    
    # Check ControlNet status after transformation
    print("\n🔍 Checking ControlNet status after transformation...")
    try:
        health_response = requests.get("http://localhost:5001/api/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"📊 ControlNet loaded: {health_data.get('controlnet_loaded', False)}")
            print(f"📊 Models loaded: {health_data.get('models_loaded', False)}")
        
        controlnet_response = requests.get("http://localhost:5001/api/controlnet")
        if controlnet_response.status_code == 200:
            controlnet_data = controlnet_response.json()
            available_controllers = [
                name for name, info in controlnet_data.get('controlnet_options', {}).items()
                if info.get('available', False)
            ]
            print(f"🎛️ Available ControlNet processors: {available_controllers}")
            
    except Exception as e:
        print(f"❌ Status check failed: {str(e)}")

if __name__ == "__main__":
    test_controlnet_loading()
