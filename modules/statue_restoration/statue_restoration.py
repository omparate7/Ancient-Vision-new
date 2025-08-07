#!/usr/bin/env python3
"""
Statue Restoration Module for Ancient Vision
===========================================

This module provides AI-powered statue restoration capabilities using 
stable diffusion inpainting with custom LoRA weights trained specifically 
for classical marble statue restoration.

Key Features:
- Automatic damage detection and masking
- Classical marble statue style inpainting
- Configurable restoration parameters
- GPU/CPU compatibility
"""

import os
import sys
import logging
import traceback
import torch
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from diffusers import StableDiffusionInpaintPipeline
from safetensors.torch import load_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatueRestorer:
    """AI-powered statue restoration using Stable Diffusion inpainting - LAZY LOADING"""
    
    def __init__(self, weights_path: str = None):
        """
        Initialize the statue restoration pipeline - MODELS LOAD ON-DEMAND
        
        Args:
            weights_path: Path to LoRA weights directory
        """
        self.pipeline = None
        self.pipe = None  # For compatibility
        self.device = self._get_device()
        self.weights_path = weights_path or self._get_default_weights_path()
        self.is_loaded = False
        
        logger.info(f"StatueRestorer initialized on device: {self.device} - LAZY LOADING MODE")
        logger.info("🔄 Models will load when restoration starts for optimal performance")
    
    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _get_default_weights_path(self) -> str:
        """Get default path for LoRA weights"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "weights")
    
    def load_model(self):
        """Load the model on-demand for memory optimization"""
        if self.is_loaded and self.pipeline is not None:
            logger.info("✅ Statue restoration model already loaded")
            return True
            
        logger.info("🔄 Loading statue restoration model on-demand...")
        
        try:
            # Clear any existing GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 Cleared GPU cache for statue restoration")
            
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(f"Weights not found at: {self.weights_path}")
            
            # Load the pipeline with device optimization
            logger.info(f"📥 Loading inpainting pipeline from: {self.weights_path}")
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.weights_path,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            self.pipeline = self.pipeline.to(self.device)
            self.pipe = self.pipeline  # For compatibility
            
            # Set memory-efficient attention if available
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
                logger.info("⚡ Enabled attention slicing for memory efficiency")
            
            if hasattr(self.pipeline, 'enable_model_cpu_offload') and self.device == 'cuda':
                self.pipeline.enable_model_cpu_offload()
                logger.info("💾 Enabled CPU offload for memory optimization")
            
            self.is_loaded = True
            logger.info("✅ Statue restoration model loaded successfully - ready for restoration!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load statue restoration model: {str(e)}")
            self.is_loaded = False
            return False
    
    def generate_damage_mask(self, image: Image.Image, 
                           edge_threshold1: int = 50, 
                           edge_threshold2: int = 150,
                           dilate_iterations: int = 2,
                           morphology_kernel_size: int = 5) -> Image.Image:
        """
        Generate a mask for damaged areas of the statue using edge detection
        
        Args:
            image: Input PIL image
            edge_threshold1: Lower threshold for Canny edge detection
            edge_threshold2: Upper threshold for Canny edge detection
            dilate_iterations: Number of dilation iterations
            morphology_kernel_size: Kernel size for morphological operations
        
        Returns:
            PIL Image: Grayscale mask highlighting damaged areas
        """
        try:
            # Convert PIL to OpenCV format
            image_array = np.array(image.convert("RGB"))
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, threshold1=edge_threshold1, threshold2=edge_threshold2)
            
            # Morphological operations to close gaps
            kernel = np.ones((morphology_kernel_size, morphology_kernel_size), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=dilate_iterations)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
            
            # Threshold to get binary mask
            _, mask = cv2.threshold(closed, 25, 255, cv2.THRESH_BINARY)
            
            # Convert to PIL Image
            mask_pil = Image.fromarray(mask).convert("L")
            return mask_pil
            
        except Exception as e:
            logger.error(f"Error generating damage mask: {str(e)}")
            raise
    
    def restore_statue(self, 
                      image: Image.Image,
                      mask: Optional[Image.Image] = None,
                      prompt: str = "face of a classical marble statue, delicate features, weathered stone",
                      negative_prompt: str = "blurry, low quality, distorted, modern, colorful, painted",
                      num_inference_steps: int = 50,
                      guidance_scale: float = 7.5,
                      strength: float = 0.8) -> Image.Image:
        """
        Restore a damaged statue image using AI inpainting - LOADS MODEL ON DEMAND
        
        Args:
            image: Input PIL image of damaged statue
            mask: Optional mask image (if None, will auto-generate)
            prompt: Text prompt for restoration style
            negative_prompt: Negative prompt to avoid unwanted features
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            strength: How much to change the masked area (0.0-1.0)
        
        Returns:
            PIL Image: Restored statue image
        """
        # Load model on-demand when restoration starts
        if not self.is_loaded:
            logger.info("🔄 Model not loaded - loading on-demand for restoration...")
            self.load_model()
        
        try:
            logger.info("🎨 Starting statue restoration with lazy loading...")
            
            # Ensure image is RGB and proper size
            image = image.convert("RGB")
            original_size = image.size
            
            # Resize for processing (diffusion models work best at 512x512)
            if image.size != (512, 512):
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Generate mask if not provided
            if mask is None:
                logger.info("🔍 Auto-generating damage mask...")
                mask = self.generate_damage_mask(image)
            else:
                mask = mask.convert("L")
                if mask.size != (512, 512):
                    mask = mask.resize((512, 512), Image.Resampling.LANCZOS)
            
            logger.info("🎭 Performing statue restoration with AI inpainting...")
            
            # Perform inpainting
            result = self.pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                negative_prompt=negative_prompt
            ).images[0]
            
            # Resize back to original size if needed
            if original_size != (512, 512):
                result = result.resize(original_size, Image.Resampling.LANCZOS)
            
            logger.info("✅ Statue restoration completed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during statue restoration: {str(e)}")
            raise
    
    def create_comparison_image(self, 
                               original: Image.Image, 
                               mask: Image.Image, 
                               restored: Image.Image) -> Image.Image:
        """
        Create a side-by-side comparison image
        
        Args:
            original: Original damaged statue image
            mask: Damage mask image
            restored: Restored statue image
        
        Returns:
            PIL Image: Comparison image with original, mask, and restored
        """
        try:
            # Ensure all images are the same size
            size = original.size
            mask = mask.resize(size, Image.Resampling.LANCZOS)
            restored = restored.resize(size, Image.Resampling.LANCZOS)
            
            # Create comparison image
            comparison_width = size[0] * 3
            comparison_height = size[1]
            comparison = Image.new("RGB", (comparison_width, comparison_height))
            
            # Paste images side by side
            comparison.paste(original, (0, 0))
            comparison.paste(mask.convert("RGB"), (size[0], 0))
            comparison.paste(restored, (size[0] * 2, 0))
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error creating comparison image: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        logger.info("Pipeline cleaned up")


# Global instance for API usage
_restorer_instance = None

def get_restorer() -> StatueRestorer:
    """Get or create global restorer instance"""
    global _restorer_instance
    if _restorer_instance is None:
        _restorer_instance = StatueRestorer()
    return _restorer_instance

def initialize_restorer() -> bool:
    """Initialize the global restorer instance"""
    restorer = get_restorer()
    return restorer.load_model()

def restore_statue_image(image: Image.Image, **kwargs) -> Tuple[Image.Image, Image.Image]:
    """
    Convenience function for statue restoration
    
    Args:
        image: Input PIL image
        **kwargs: Additional arguments for restore_statue method
    
    Returns:
        Tuple[Image.Image, Image.Image]: (restored_image, damage_mask)
    """
    restorer = get_restorer()
    if not restorer.is_loaded:
        if not restorer.load_pipeline():
            raise RuntimeError("Failed to load statue restoration pipeline")
    
    # Generate mask
    mask = restorer.generate_damage_mask(image)
    
    # Restore statue
    restored = restorer.restore_statue(image, mask, **kwargs)
    
    return restored, mask


if __name__ == "__main__":
    # Test the statue restoration module
    print("Testing Statue Restoration Module...")
    
    restorer = StatueRestorer()
    if restorer.load_pipeline():
        print("✅ Pipeline loaded successfully")
        
        # Test with a dummy image (would be replaced with actual image in real usage)
        test_image = Image.new("RGB", (512, 512), color="white")
        try:
            restored, mask = restore_statue_image(test_image)
            print("✅ Restoration test completed")
        except Exception as e:
            print(f"❌ Restoration test failed: {e}")
    else:
        print("❌ Failed to load pipeline")
