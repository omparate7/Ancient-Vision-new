from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import base64
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
import json
from datetime import datetime
import uuid
import logging
import cv2
import numpy as np
try:
    from controlnet_aux import CannyDetector, MidasDetector, OpenposeDetector
    CONTROLNET_AVAILABLE = True
    print("✅ ControlNet auxiliary libraries loaded successfully")
except ImportError as e:
    print(f"⚠️ ControlNet-aux not available: {e}. Some advanced features may be limited.")
    CONTROLNET_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'], 
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'OPTIONS'])

# Configuration
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Global variables for models - NOW LAZY LOADED
current_pipeline = None
current_model = None
available_models = {}
models_loaded = False

# ControlNet models for structural guidance - LAZY LOADED
controlnet_models = {}
controlnet_processors = {}
controlnet_loaded = False

def get_device():
    """Get the optimal device for model loading"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# ControlNet configurations for different guidance types
CONTROLNET_CONFIGS = {
    "canny": {
        "model_id": "lllyasviel/sd-controlnet-canny",
        "name": "Canny Edge Detection",
        "description": "Preserves edge structure and outlines"
    },
    "depth": {
        "model_id": "lllyasviel/sd-controlnet-depth", 
        "name": "Depth Estimation",
        "description": "Maintains depth and spatial relationships"
    },
    "openpose": {
        "model_id": "lllyasviel/sd-controlnet-openpose",
        "name": "Pose Detection", 
        "description": "Preserves human pose and body structure"
    }
}

# Predefined Ukiyo-e styles - Multiple authentic styles
STYLES = {
    "classic_ukiyo": {
        "name": "Classic Ukiyo-e",
        "prompt": "ukiyo-e woodblock print, traditional Japanese art, Edo period style, bold black outlines, flat color blocks, no gradients, simplified forms, decorative patterns, traditional pigments, woodcut texture, printmaking style, Japanese aesthetics, stylized composition, geometric shapes, strong contrast, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, gradients, shading, shadows, depth, modern style, western art, photograph, detailed textures, soft edges, blurred lines, contemporary, digital art, airbrushed, smooth surfaces"
    },
    "landscape_ukiyo": {
        "name": "Landscape Ukiyo-e", 
        "prompt": "ukiyo-e landscape print, traditional Japanese scenery, Mount Fuji, cherry blossoms, traditional architecture, bold outlines, flat colors, stylized nature, Edo period landscape art, woodblock print texture, Japanese countryside, serene composition, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, gradients, shading, shadows, modern buildings, western landscape, photograph, detailed textures, contemporary, digital art"
    },
    "portrait_ukiyo": {
        "name": "Portrait Ukiyo-e",
        "prompt": "ukiyo-e portrait print, traditional Japanese figure, geisha, samurai, kabuki actor, elegant kimono patterns, bold facial features, flat color blocks, decorative background, Edo period portraiture, woodblock print style, stylized human figure, masterpiece", 
        "negative_prompt": "photorealistic, realistic, 3D render, gradients, shading, modern clothing, western portrait, photograph, detailed skin texture, contemporary fashion, digital art"
    },
    "nature_ukiyo": {
        "name": "Nature Ukiyo-e",
        "prompt": "ukiyo-e nature print, traditional Japanese flora and fauna, birds, flowers, bamboo, pine trees, decorative natural elements, bold outlines, seasonal themes, woodblock print texture, stylized organic forms, flat color palette, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, gradients, shading, modern nature photography, western botanical art, detailed textures, contemporary, digital art"
    },
    "urban_ukiyo": {
        "name": "Urban Ukiyo-e", 
        "prompt": "ukiyo-e urban scene, traditional Japanese city life, Edo period streets, merchant districts, traditional architecture, busy marketplace, bold outlines, flat colors, stylized urban composition, woodblock print style, cultural scenes, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, gradients, shading, modern city, skyscrapers, cars, contemporary urban life, western architecture, photograph, digital art"
    },
    "seasonal_ukiyo": {
        "name": "Seasonal Ukiyo-e",
        "prompt": "ukiyo-e seasonal print, traditional Japanese seasons, spring cherry blossoms, summer festivals, autumn maple leaves, winter snow scenes, seasonal activities, bold outlines, flat colors, decorative seasonal elements, woodblock print texture, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, gradients, shading, modern seasonal scenes, western seasonal art, photograph, detailed textures, contemporary, digital art"
    },
    "gond": {
        "name": "Gond Art",
        "prompt": "gond art, traditional tribal art from central India, intricate dot patterns, nature motifs, trees, animals, birds, vibrant colors, geometric patterns, tribal storytelling, traditional pigments, folk art style, madhya pradesh art, adivasi culture, detailed line work, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, modern art, western art, photograph, digital art, contemporary, abstract"
    },
    "kalighat": {
        "name": "Kalighat Painting",
        "prompt": "kalighat painting, bengali folk art, bold lines, flat colors, traditional bengal painting, simple compositions, mythological themes, religious subjects, hand painted, traditional indian pigments, kolkata art style, expressive figures, cultural heritage, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, gradients, modern art, western art, photograph, digital art, contemporary"
    },
    "kangra": {
        "name": "Kangra Miniature",
        "prompt": "kangra miniature painting, himalayan art, pahari school, delicate brushwork, fine details, natural colors, landscape backgrounds, royal court scenes, romantic themes, traditional indian miniature style, himachal pradesh art, detailed architecture, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, modern art, western art, photograph, digital art, contemporary, abstract"
    },
    "kerala_mural": {
        "name": "Kerala Mural",
        "prompt": "kerala mural painting, south indian temple art, mythological scenes, vibrant colors, traditional pigments, religious themes, decorative patterns, classical indian art, temple wall painting, traditional kerala style, divine figures, ornate details, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, modern art, western art, photograph, digital art, contemporary"
    },
    "madhubani": {
        "name": "Madhubani Art",
        "prompt": "madhubani painting, mithila art, bihar folk art, intricate patterns, geometric designs, nature themes, vibrant colors, traditional motifs, ritualistic art, hand painted, traditional indian pigments, cultural heritage, detailed linework, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, modern art, western art, photograph, digital art, contemporary, abstract"
    },
    "mandana": {
        "name": "Mandana Art",
        "prompt": "mandana art, rajasthani folk art, geometric patterns, wall painting, traditional decorative art, symmetrical designs, auspicious symbols, festival art, tribal patterns, hand painted, natural pigments, cultural decoration, traditional motifs, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, modern art, western art, photograph, digital art, contemporary"
    },
    "pichwai": {
        "name": "Pichwai Painting",
        "prompt": "pichwai painting, nathdwara art, krishna themes, temple backdrop art, rajasthani painting, religious art, traditional indian painting, devotional art, intricate details, vibrant colors, spiritual themes, temple decoration, cultural heritage, masterpiece",
        "negative_prompt": "photorealistic, realistic, 3D render, modern art, western art, photograph, digital art, contemporary"
    }
}

def scan_for_models():
    """Scan for available models in the models directory"""
    global available_models
    
    available_models = {}
    
    # Scan the models directory for available models
    if os.path.exists(MODELS_FOLDER):
        logger.info(f"Scanning models directory: {MODELS_FOLDER}")
        
        for item in os.listdir(MODELS_FOLDER):
            item_path = os.path.join(MODELS_FOLDER, item)
            
            # Skip files, only check directories
            if not os.path.isdir(item_path):
                continue
                
            # Check if it's a full Stable Diffusion model (has model_index.json)
            model_index_path = os.path.join(item_path, "model_index.json")
            if os.path.exists(model_index_path):
                model_name = "Ukiyo-e Traditional Art" if item == "ukiyo_e_lora" else f"{item.replace('_', ' ').title()}"
                available_models[item] = {
                    "name": model_name,
                    "type": "local",
                    "path": item_path,
                    "description": f"Local model: {model_name}"
                }
                logger.info(f"Found full model at: {item_path}")
            
            # Check if it's a LoRA model (has .safetensors files)
            elif any(f.endswith('.safetensors') for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))):
                model_name = "Indian Traditional Art" if item == "lora_IndianArt" else f"{item.replace('_', ' ').title()}"
                available_models[item] = {
                    "name": model_name,
                    "type": "lora",
                    "path": item_path,
                    "description": f"LoRA adapter: {model_name}"
                }
                logger.info(f"Found LoRA model at: {item_path}")
    
    # If no models found, add fallback
    if not available_models:
        logger.warning("No local models found, adding fallback online model")
        available_models["runwayml/stable-diffusion-v1-5"] = {
            "name": "Stable Diffusion v1.5 (Fallback)",
            "type": "online",
            "path": "runwayml/stable-diffusion-v1-5",
            "description": "Standard model (will be converted to traditional art style)"
        }
    
    logger.info(f"Available models: {list(available_models.keys())}")
    return available_models

def initialize_controlnet_models():
    """Initialize ControlNet models and processors - LAZY LOADING"""
    global controlnet_models, controlnet_processors, controlnet_loaded
    
    # Only load when actually needed
    if controlnet_loaded:
        logger.info("♻️ ControlNet models already loaded")
        return
    
    try:
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float32
        else:
            device = "cpu"
            torch_dtype = torch.float32
            
        logger.info(f"🔄 Lazy loading ControlNet models on device: {device}")
        
        # Initialize ControlNet models only when needed
        for control_type, config in CONTROLNET_CONFIGS.items():
            try:
                logger.info(f"Loading ControlNet model: {config['name']}")
                controlnet = ControlNetModel.from_pretrained(
                    config['model_id'],
                    torch_dtype=torch_dtype
                ).to(device)
                controlnet_models[control_type] = controlnet
                logger.info(f"✓ Loaded {config['name']} ControlNet")
            except Exception as e:
                logger.warning(f"Failed to load {config['name']} ControlNet: {str(e)}")
        
        controlnet_loaded = True
        
        # Initialize preprocessors only if ControlNet aux is available
        if CONTROLNET_AVAILABLE:
            try:
                controlnet_processors['canny'] = CannyDetector()
                logger.info("✓ Initialized Canny detector")
            except Exception as e:
                logger.warning(f"Failed to initialize Canny detector: {str(e)}")
                
            try:
                controlnet_processors['depth'] = MidasDetector.from_pretrained('valhalla/t2iadapter-aux-models')
                logger.info("✓ Initialized Depth detector")
            except Exception as e:
                logger.warning(f"Failed to initialize Depth detector: {str(e)}")
                
            try:
                controlnet_processors['openpose'] = OpenposeDetector.from_pretrained('valhalla/t2iadapter-aux-models')
                logger.info("✓ Initialized Pose detector")
            except Exception as e:
                logger.warning(f"Failed to initialize Pose detector: {str(e)}")
        else:
            logger.warning("⚠️ ControlNet auxiliary libraries not available - processors not initialized")
            
        logger.info(f"ControlNet initialization complete. Models: {list(controlnet_models.keys())}, Processors: {list(controlnet_processors.keys())}")
        
    except Exception as e:
        logger.error(f"Error initializing ControlNet models: {str(e)}")

def preprocess_controlnet_input(image, control_type, **kwargs):
    """Preprocess image for specific ControlNet type"""
    try:
        # Check if ControlNet aux is available and processors are initialized
        if not CONTROLNET_AVAILABLE:
            logger.warning(f"⚠️ ControlNet auxiliary not available for {control_type} preprocessing")
            return image  # Return original image as fallback
            
        if control_type not in controlnet_processors:
            logger.warning(f"⚠️ Processor for {control_type} not available, using fallback")
            return image  # Return original image as fallback
            
        if control_type == 'canny':
            low_threshold = kwargs.get('low_threshold', 100)
            high_threshold = kwargs.get('high_threshold', 200)
            control_image = controlnet_processors['canny'](
                image, 
                low_threshold=low_threshold, 
                high_threshold=high_threshold
            )
            
        elif control_type == 'depth':
            control_image = controlnet_processors['depth'](image)
            
        elif control_type == 'openpose':
            control_image = controlnet_processors['openpose'](image)
            
        else:
            logger.warning(f"Unsupported ControlNet type: {control_type}, using original image")
            return image
            
        return control_image
        
    except Exception as e:
        logger.error(f"Error preprocessing image for {control_type}: {str(e)}")
        logger.info("Falling back to original image")
        return image  # Return original image as fallback instead of raising

def load_model(model_id, control_type=None):
    """Load a specific model ONLY when needed for transformation"""
    global current_pipeline, current_model
    
    # Create a unique identifier for the pipeline configuration
    pipeline_id = f"{model_id}_{control_type}" if control_type else model_id
    
    # Only return existing pipeline if it's the exact same configuration
    if current_model == pipeline_id and current_pipeline is not None:
        logger.info(f"♻️ Reusing already loaded pipeline: {pipeline_id}")
        return current_pipeline
    
    try:
        model_info = available_models.get(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")
        
        logger.info(f"🔄 Loading model: {model_info['name']} (Type: {model_info['type']}) with ControlNet: {control_type or 'None'}")
        
        # Clear current pipeline to free memory
        if current_pipeline:
            logger.info("🗑️ Clearing previous pipeline to save memory")
            del current_pipeline
            current_pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        # Determine device and dtype
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float32
        else:
            device = "cpu"
            torch_dtype = torch.float32
        
        logger.info(f"Using device: {device}")
        
        # Determine base model path based on model type
        if model_info['type'] == 'local':
            # Full Stable Diffusion model
            base_model_path = model_info['path']
        elif model_info['type'] == 'lora':
            # LoRA adapter - use base model
            base_model_path = "runwayml/stable-diffusion-v1-5"
        else:
            # Online model
            base_model_path = model_info['path']
        
        # Load the pipeline with ControlNet if specified
        if control_type and control_type in controlnet_models:
            # Load ControlNet pipeline
            controlnet = controlnet_models[control_type]
            pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                base_model_path,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True if device == "cpu" else False
            )
            logger.info(f"Loaded ControlNet pipeline with {CONTROLNET_CONFIGS[control_type]['name']}")
        else:
            # Load standard pipeline
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                base_model_path,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True if device == "cpu" else False
            )
        
        # Apply LoRA weights if it's a LoRA model
        if model_info['type'] == 'lora':
            lora_path = None
            for file in os.listdir(model_info['path']):
                if file.endswith('.safetensors'):
                    lora_path = os.path.join(model_info['path'], file)
                    break
            
            if lora_path:
                logger.info(f"Applying LoRA weights from: {lora_path}")
                try:
                    pipeline.load_lora_weights(lora_path)
                except Exception as e:
                    logger.warning(f"Could not load LoRA weights: {e}")
                    logger.info("Continuing with base model...")
        
        # Use DPM Solver for better quality
        from diffusers import DPMSolverMultistepScheduler
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        # Move to device
        pipeline = pipeline.to(device)
        
        # Enable memory efficient optimizations
        if hasattr(pipeline, 'enable_attention_slicing'):
            pipeline.enable_attention_slicing()
        if hasattr(pipeline, 'enable_vae_slicing'):
            pipeline.enable_vae_slicing()
        
        current_pipeline = pipeline
        current_model = pipeline_id
        logger.info(f"Model loaded successfully: {model_info['name']} on {device}")
        
        return pipeline
        
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {str(e)}")
        raise

def process_image(image_data, prompt="", negative_prompt="", style="classic_ukiyo", 
                 strength=0.85, guidance_scale=12.0, num_inference_steps=30, model_id=None):
    """Process image with transformation and automatic ControlNet guidance - LOADS MODELS ON-DEMAND"""
    try:
        logger.info(f"🎨 Starting image transformation with style: {style}")
        
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize image - use fixed 512x512 dimensions
        width = 512
        height = 512
        
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Automatically determine best ControlNet type
        control_type = "canny"  # Default to canny for edge preservation
        controlnet_conditioning_scale = 0.6  # Lower for more artistic freedom
        
        # Load model on-demand - default to available model
        if not model_id:
            model_id = "ukiyo_e_lora" if "ukiyo_e_lora" in available_models else list(available_models.keys())[0]
        
        logger.info(f"🔄 Loading pipeline on-demand: {model_id} with ControlNet: {control_type}")
        
        # Initialize ControlNet models if needed for this transformation
        if control_type and not controlnet_loaded:
            logger.info("🔄 Initializing ControlNet models for this transformation...")
            initialize_controlnet_models()
        
        # Load pipeline with automatic ControlNet ON-DEMAND
        pipeline = load_model(model_id, control_type)
        
        # Get style information
        style_info = STYLES.get(style, STYLES["classic_ukiyo"])
        
        # Build prompts - if user prompt is provided, append it to the default prompt
        if prompt.strip():
            # Append user prompt to the default style prompt
            full_prompt = f"{style_info['prompt']}, {prompt.strip()}"
        else:
            # Use only the default style prompt
            full_prompt = style_info['prompt']
        
        # Always use style negative prompt
        full_negative_prompt = style_info['negative_prompt']
        
        logger.info(f"Processing image with Ukiyo-e style: {style}")
        logger.info(f"Using prompt: {full_prompt}")
        logger.info(f"Auto-applying ControlNet: {CONTROLNET_CONFIGS[control_type]['name']}")
        
        # Prepare generation parameters
        generation_kwargs = {
            "prompt": full_prompt,
            "image": image,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "negative_prompt": full_negative_prompt,
            "num_inference_steps": num_inference_steps,
            "width": width,
            "height": height
        }
        
        # Add ControlNet parameters automatically
        if control_type and control_type in controlnet_models:
            # Preprocess image for ControlNet
            control_image = preprocess_controlnet_input(
                image, 
                control_type,
                low_threshold=100,
                high_threshold=200
            )
            
            generation_kwargs.update({
                "control_image": control_image,
                "controlnet_conditioning_scale": controlnet_conditioning_scale
            })
            
            logger.info(f"Applied automatic ControlNet guidance with strength {controlnet_conditioning_scale}")
        
        # Generate image with Ukiyo-e optimized parameters
        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            result = pipeline(**generation_kwargs).images[0]
        
        # Convert result to base64
        output_buffer = io.BytesIO()
        result.save(output_buffer, format='PNG')
        output_buffer.seek(0)
        
        result_base64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        
        return {
            'success': True,
            'image': f"data:image/png;base64,{result_base64}",
            'prompt_used': full_prompt,
            'negative_prompt_used': full_negative_prompt,
            'style_applied': style_info['name'],
            'controlnet_used': CONTROLNET_CONFIGS[control_type]['name'] if control_type else None,
            'dimensions': f"{width}x{height}"
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    scan_for_models()
    return jsonify({
        'models': available_models,
        'current_model': current_model
    })

@app.route('/api/styles', methods=['GET'])
def get_styles():
    """Get available styles"""
    return jsonify({'styles': STYLES})

@app.route('/api/controlnet', methods=['GET'])
def get_controlnet_options():
    """Get available ControlNet options"""
    available_controlnets = {}
    for control_type, config in CONTROLNET_CONFIGS.items():
        if control_type in controlnet_models:
            available_controlnets[control_type] = {
                "name": config["name"],
                "description": config["description"],
                "available": True
            }
        else:
            available_controlnets[control_type] = {
                "name": config["name"], 
                "description": config["description"],
                "available": False
            }
    
    return jsonify({'controlnet_options': available_controlnets})

@app.route('/api/transform', methods=['POST'])
def transform_image():
    """Transform an image with Ukiyo-e style and automatic ControlNet guidance"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Extract parameters - make prompt optional
        image_data = data['image']
        prompt = data.get('prompt', '')  # Optional, can be empty
        style = data.get('style', 'classic_ukiyo')  # Default to classic Ukiyo-e
        strength = float(data.get('strength', 0.85))  # Higher for more style transformation
        guidance_scale = float(data.get('guidance_scale', 12.0))  # Higher for stronger style adherence
        num_inference_steps = int(data.get('num_inference_steps', 30))  # More steps for better quality
        model_id = data.get('model_id')  # Will default to Ukiyo-e model
        
        # Process the image with automatic ControlNet
        result = process_image(
            image_data=image_data,
            prompt=prompt,
            style=style,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            model_id=model_id
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in transform_image: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with lazy loading status"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        'device': get_device(),
        'current_model': current_model,
        'models_loaded': models_loaded,
        'controlnet_loaded': controlnet_loaded,
        'lazy_loading': True,
        'available_models': list(available_models.keys()),
        'pipeline_ready': current_pipeline is not None
    })

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'UkiyoeFusion API',
        'version': '1.0.0',
        'endpoints': [
            '/api/models',
            '/api/styles', 
            '/api/controlnet',
            '/api/transform',
            '/api/health'
        ]
    })

if __name__ == '__main__':
    logger.info("🎨 Starting Ancient Vision API Server - LAZY LOADING MODE")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    # Only scan for available models on startup - NO LOADING
    scan_for_models()
    logger.info("📋 Model scanning complete. Models will load on-demand for better performance.")
    
    # ControlNet models will be loaded when needed
    logger.info("⚡ ControlNet models will load when transformation requires them.")
    
    # Use port 5001 to avoid conflicts with AirPlay
    port = int(os.environ.get('FLASK_RUN_PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
