#!/usr/bin/env python3
"""
Test script to check statue restoration model loading
"""
import sys
import os
sys.path.append('/Users/vinaysurwase/Documents/PR_NEW/Ancient-Vision/modules/statue_restoration')

from statue_restoration import StatueRestorer

def test_statue_loading():
    """Test statue restoration model loading"""
    print("🧪 Testing statue restoration model loading...")
    
    try:
        # Create restorer instance
        restorer = StatueRestorer()
        print(f"✅ StatueRestorer created on device: {restorer.device}")
        print(f"📁 Weights path: {restorer.weights_path}")
        print(f"📋 Weights exist: {os.path.exists(restorer.weights_path)}")
        
        # Test model loading
        print("🔄 Attempting to load model...")
        success = restorer.load_model()
        
        if success:
            print("✅ Model loaded successfully!")
            print(f"📊 Model loaded: {restorer.is_loaded}")
            print(f"🔧 Pipeline available: {restorer.pipeline is not None}")
        else:
            print("❌ Model loading failed!")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_statue_loading()
