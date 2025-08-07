#!/usr/bin/env python3
"""
Statue Restoration API for Ancient Vision
========================================

Flask API endpoints for the statue restoration module.
Provides REST API for statue restoration functionality.
"""

import os
import sys
import logging
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import traceback

# Add parent directories to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from statue_restoration import StatueRestorer, get_restorer, initialize_restorer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3001"])

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert base64 string to PIL Image"""
    if base64_str.startswith('data:image'):
        base64_str = base64_str.split(',')[1]
    
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

@app.route('/api/statue-restoration/health', methods=['GET'])
def health_check():
    """Health check endpoint with lazy loading status"""
    try:
        restorer = get_restorer()
        status = "loaded" if restorer.is_loaded else "ready_for_lazy_loading"
        
        return jsonify({
            "status": "healthy",
            "module": "statue_restoration",
            "pipeline_status": status,
            "lazy_loading": True,
            "device": str(restorer.device) if restorer else "unknown",
            "weights_available": os.path.exists(restorer.weights_path) if restorer else False
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/api/statue-restoration/initialize', methods=['POST'])
def initialize():
    """Initialize the statue restoration pipeline (legacy endpoint - now uses lazy loading)"""
    try:
        logger.info("🔄 Initialize request received - using lazy loading approach...")
        restorer = get_restorer()
        
        return jsonify({
            "success": True,
            "message": "Statue restoration ready with lazy loading - models will load on first restoration",
            "lazy_loading": True,
            "device": str(restorer.device),
            "weights_available": os.path.exists(restorer.weights_path)
        })
            
    except Exception as e:
        logger.error(f"Error in initialize endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/statue-restoration/generate-mask', methods=['POST'])
def generate_mask():
    """Generate damage mask for a statue image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        # Convert base64 to PIL Image
        image = base64_to_image(data['image'])
        
        # Get mask generation parameters
        edge_threshold1 = data.get('edge_threshold1', 50)
        edge_threshold2 = data.get('edge_threshold2', 150)
        dilate_iterations = data.get('dilate_iterations', 2)
        kernel_size = data.get('kernel_size', 5)
        
        # Get restorer instance
        restorer = get_restorer()
        
        # Models load on-demand - no need to check if loaded for mask generation
        logger.info("🔍 Generating damage mask (no model loading required)...")
        
        # Generate mask
        mask = restorer.generate_damage_mask(
            image,
            edge_threshold1=edge_threshold1,
            edge_threshold2=edge_threshold2,
            dilate_iterations=dilate_iterations,
            morphology_kernel_size=kernel_size
        )
        
        # Convert mask to base64
        mask_base64 = image_to_base64(mask.convert("RGB"))
        
        return jsonify({
            "success": True,
            "mask": mask_base64,
            "parameters_used": {
                "edge_threshold1": edge_threshold1,
                "edge_threshold2": edge_threshold2,
                "dilate_iterations": dilate_iterations,
                "kernel_size": kernel_size
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating mask: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/restore', methods=['POST'])
@app.route('/api/statue-restoration/restore', methods=['POST'])
def restore():
    """Restore a damaged statue image"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        # Convert base64 to PIL Image
        image = base64_to_image(data['image'])
        
        # Get optional mask
        mask = None
        if 'mask' in data:
            mask = base64_to_image(data['mask'])
        
        # Get restoration parameters
        prompt = data.get('prompt', "face of a classical marble statue, delicate features, weathered stone")
        negative_prompt = data.get('negative_prompt', "blurry, low quality, distorted, modern, colorful, painted")
        num_inference_steps = data.get('num_inference_steps', 50)
        guidance_scale = data.get('guidance_scale', 7.5)
        strength = data.get('strength', 0.8)
        
        # Get restorer instance
        restorer = get_restorer()
        
        # Generate mask if not provided
        if mask is None:
            mask = restorer.generate_damage_mask(image)
        
        # Lazy loading: Model will load automatically when restore_statue is called
        logger.info("🎨 Starting statue restoration with lazy loading...")
        if not restorer.is_loaded:
            logger.info("🔄 Model will load on-demand during restoration...")
        
        # Perform restoration (model loads automatically if needed)
        restored_image = restorer.restore_statue(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        )
        
        # Convert results to base64
        restored_base64 = image_to_base64(restored_image)
        mask_base64 = image_to_base64(mask.convert("RGB"))
        
        # Create comparison image
        comparison = restorer.create_comparison_image(image, mask, restored_image)
        comparison_base64 = image_to_base64(comparison)
        
        return jsonify({
            "success": True,
            "restored_image": restored_base64,
            "mask_used": mask_base64,
            "comparison": comparison_base64,
            "parameters_used": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "strength": strength
            }
        })
        
    except Exception as e:
        logger.error(f"Error during restoration: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/statue-restoration/presets', methods=['GET'])
def get_presets():
    """Get preset configurations for different restoration types"""
    presets = {
        "classical_marble": {
            "name": "Classical Marble",
            "prompt": "face of a classical marble statue, delicate features, weathered stone, ancient sculpture",
            "negative_prompt": "blurry, low quality, distorted, modern, colorful, painted, cartoon",
            "guidance_scale": 7.5,
            "strength": 0.8,
            "num_inference_steps": 50
        },
        "renaissance_sculpture": {
            "name": "Renaissance Sculpture",
            "prompt": "renaissance marble sculpture, detailed craftsmanship, classical proportions, museum quality",
            "negative_prompt": "blurry, low quality, distorted, modern, colorful, painted, digital art",
            "guidance_scale": 8.0,
            "strength": 0.75,
            "num_inference_steps": 60
        },
        "ancient_statue": {
            "name": "Ancient Statue",
            "prompt": "ancient stone statue, weathered by time, historical artifact, archaeological discovery",
            "negative_prompt": "modern, new, shiny, colorful, painted, plastic, digital",
            "guidance_scale": 7.0,
            "strength": 0.85,
            "num_inference_steps": 45
        },
        "greek_sculpture": {
            "name": "Greek Sculpture",
            "prompt": "ancient greek marble sculpture, classical features, perfect proportions, museum piece",
            "negative_prompt": "modern, colorful, painted, low quality, distorted, cartoon",
            "guidance_scale": 7.5,
            "strength": 0.8,
            "num_inference_steps": 50
        }
    }
    
    return jsonify({
        "success": True,
        "presets": presets
    })

@app.route('/api/statue-restoration/status', methods=['GET'])
def get_status():
    """Get current status of the restoration module with lazy loading info"""
    try:
        restorer = get_restorer()
        
        return jsonify({
            "success": True,
            "status": {
                "is_loaded": restorer.is_loaded,
                "lazy_loading_enabled": True,
                "device": str(restorer.device),
                "weights_path": restorer.weights_path,
                "weights_exist": os.path.exists(restorer.weights_path),
                "ready_for_restoration": True,
                "memory_optimized": True
            }
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == "__main__":
    logger.info("🏛️  Starting Statue Restoration API Server - LAZY LOADING MODE")
    
    # Initialize restorer instance without loading models (lazy loading)
    logger.info("🔄 Initializing statue restoration with lazy loading...")
    try:
        restorer = get_restorer()
        logger.info(f"✅ Statue restoration ready on device: {restorer.device}")
        logger.info("💡 Models will load on-demand for optimal performance")
    except Exception as e:
        logger.warning(f"⚠️  Restorer initialization warning: {e}")
        logger.info("🔄 Will initialize on first request")
    
    # Start the Flask app
    logger.info("🚀 Starting API server on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=True)
