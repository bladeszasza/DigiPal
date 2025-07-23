"""
Image generation system for DigiPal visualization using FLUX.1-dev model.
"""

import os
import torch
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from PIL import Image
import hashlib
import json
from datetime import datetime

from ..core.models import DigiPal
from ..core.enums import LifeStage, EggType

# Set up logging
logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    Handles image generation for DigiPal pets using FLUX.1-dev model.
    Includes caching, fallback systems, and professional prompt generation.
    """
    
    def __init__(self, 
                 model_name: str = "black-forest-labs/FLUX.1-dev",
                 cache_dir: str = "demo_assets/images",
                 fallback_dir: str = "demo_assets/images/fallbacks"):
        """
        Initialize the image generator.
        
        Args:
            model_name: HuggingFace model name for image generation
            cache_dir: Directory to cache generated images
            fallback_dir: Directory containing fallback images
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.fallback_dir = Path(fallback_dir)
        self.pipe = None
        self._model_loaded = False
        
        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        
        # Image generation parameters
        self.generation_params = {
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "num_inference_steps": 50,
            "max_sequence_length": 512
        }
        
        # Initialize prompt templates
        self._init_prompt_templates()
        
        # Initialize fallback images
        self._init_fallback_images()
    
    def _init_prompt_templates(self):
        """Initialize professional prompt templates for each life stage and egg type."""
        
        # Base style modifiers
        self.style_base = "digital art, high quality, detailed, vibrant colors, anime style"
        
        # Egg type characteristics
        self.egg_type_traits = {
            EggType.RED: {
                "element": "fire",
                "colors": "red, orange, golden",
                "traits": "fierce, energetic, blazing aura",
                "environment": "volcanic, warm lighting"
            },
            EggType.BLUE: {
                "element": "water",
                "colors": "blue, cyan, silver",
                "traits": "calm, protective, flowing aura",
                "environment": "aquatic, cool lighting"
            },
            EggType.GREEN: {
                "element": "earth",
                "colors": "green, brown, gold",
                "traits": "sturdy, wise, natural aura",
                "environment": "forest, natural lighting"
            }
        }
        
        # Life stage characteristics
        self.stage_traits = {
            LifeStage.EGG: {
                "form": "mystical egg with glowing patterns",
                "size": "medium sized",
                "features": "smooth shell, magical runes, soft glow"
            },
            LifeStage.BABY: {
                "form": "small cute creature",
                "size": "tiny, adorable",
                "features": "big eyes, soft fur, playful expression"
            },
            LifeStage.CHILD: {
                "form": "young creature",
                "size": "small but growing",
                "features": "curious eyes, developing features, energetic pose"
            },
            LifeStage.TEEN: {
                "form": "adolescent creature",
                "size": "medium sized",
                "features": "developing strength, confident stance, maturing features"
            },
            LifeStage.YOUNG_ADULT: {
                "form": "strong young creature",
                "size": "well-proportioned",
                "features": "athletic build, determined expression, full power"
            },
            LifeStage.ADULT: {
                "form": "mature powerful creature",
                "size": "large and imposing",
                "features": "wise eyes, peak physical form, commanding presence"
            },
            LifeStage.ELDERLY: {
                "form": "ancient wise creature",
                "size": "dignified stature",
                "features": "wise expression, weathered but noble, mystical aura"
            }
        }
    
    def _init_fallback_images(self):
        """Initialize fallback image mappings."""
        self.fallback_images = {}
        
        # Create simple fallback images if they don't exist
        for stage in LifeStage:
            for egg_type in EggType:
                fallback_path = self.fallback_dir / f"{stage.value}_{egg_type.value}.png"
                self.fallback_images[f"{stage.value}_{egg_type.value}"] = str(fallback_path)
                
                # Create a simple placeholder if file doesn't exist
                if not fallback_path.exists():
                    self._create_placeholder_image(fallback_path, stage, egg_type)
    
    def _create_placeholder_image(self, path: Path, stage: LifeStage, egg_type: EggType):
        """Create a simple placeholder image."""
        try:
            # Create a simple colored rectangle as placeholder
            color_map = {
                EggType.RED: (255, 100, 100),
                EggType.BLUE: (100, 100, 255),
                EggType.GREEN: (100, 255, 100)
            }
            
            color = color_map.get(egg_type, (128, 128, 128))
            img = Image.new('RGB', (512, 512), color)
            img.save(path)
            logger.info(f"Created placeholder image: {path}")
            
        except Exception as e:
            logger.error(f"Failed to create placeholder image {path}: {e}")
    
    def _load_model(self):
        """Load the FLUX.1-dev model for image generation."""
        if self._model_loaded:
            return
        
        try:
            from diffusers import FluxPipeline
            
            logger.info(f"Loading image generation model: {self.model_name}")
            self.pipe = FluxPipeline.from_pretrained(
                self.model_name, 
                torch_dtype=torch.bfloat16
            )
            
            # Enable CPU offload to save VRAM
            self.pipe.enable_model_cpu_offload()
            
            self._model_loaded = True
            logger.info("Image generation model loaded successfully")
            
        except ImportError:
            logger.error("diffusers library not installed. Run: pip install -U diffusers")
            raise
        except Exception as e:
            logger.error(f"Failed to load image generation model: {e}")
            raise
    
    def generate_prompt(self, digipal: DigiPal) -> str:
        """
        Generate a professional prompt for DigiPal image generation.
        
        Args:
            digipal: DigiPal instance to generate prompt for
            
        Returns:
            Professional prompt string for image generation
        """
        egg_traits = self.egg_type_traits.get(digipal.egg_type, self.egg_type_traits[EggType.RED])
        stage_traits = self.stage_traits.get(digipal.life_stage, self.stage_traits[LifeStage.BABY])
        
        # Build attribute modifiers based on DigiPal stats
        attribute_modifiers = []
        
        # High offense = more aggressive/fierce appearance
        if digipal.offense > 50:
            attribute_modifiers.append("fierce expression, sharp features")
        
        # High defense = more armored/protective appearance
        if digipal.defense > 50:
            attribute_modifiers.append("armored, protective stance")
        
        # High speed = more sleek/agile appearance
        if digipal.speed > 50:
            attribute_modifiers.append("sleek, agile build")
        
        # High brains = more intelligent/wise appearance
        if digipal.brains > 50:
            attribute_modifiers.append("intelligent eyes, wise demeanor")
        
        # Happiness affects expression
        if digipal.happiness > 70:
            attribute_modifiers.append("happy, cheerful expression")
        elif digipal.happiness < 30:
            attribute_modifiers.append("sad, tired expression")
        
        # Build the complete prompt
        prompt_parts = [
            f"a {stage_traits['form']} digimon",
            f"touched by the power of {egg_traits['element']}",
            f"{stage_traits['size']}, {stage_traits['features']}",
            f"colors: {egg_traits['colors']}",
            f"{egg_traits['traits']}"
        ]
        
        if attribute_modifiers:
            prompt_parts.append(", ".join(attribute_modifiers))
        
        prompt_parts.extend([
            f"in {egg_traits['environment']}",
            f"life stage: {digipal.life_stage.value}",
            self.style_base
        ])
        
        prompt = ", ".join(prompt_parts)
        
        logger.debug(f"Generated prompt for {digipal.name}: {prompt}")
        return prompt
    
    def _get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate cache key for image based on prompt and parameters."""
        cache_data = {
            "prompt": prompt,
            "params": params
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_image_path(self, cache_key: str) -> Optional[Path]:
        """Check if cached image exists and return path."""
        cache_path = self.cache_dir / f"{cache_key}.png"
        if cache_path.exists():
            logger.debug(f"Found cached image: {cache_path}")
            return cache_path
        return None
    
    def _save_to_cache(self, image: Image.Image, cache_key: str) -> Path:
        """Save generated image to cache."""
        cache_path = self.cache_dir / f"{cache_key}.png"
        image.save(cache_path)
        logger.info(f"Saved generated image to cache: {cache_path}")
        return cache_path
    
    def _get_fallback_image(self, digipal: DigiPal) -> str:
        """Get fallback image path for DigiPal."""
        fallback_key = f"{digipal.life_stage.value}_{digipal.egg_type.value}"
        fallback_path = self.fallback_images.get(fallback_key)
        
        if fallback_path and Path(fallback_path).exists():
            logger.info(f"Using fallback image: {fallback_path}")
            return fallback_path
        
        # Ultimate fallback - create a generic placeholder
        generic_fallback = self.fallback_dir / "generic_placeholder.png"
        if not generic_fallback.exists():
            self._create_placeholder_image(generic_fallback, digipal.life_stage, digipal.egg_type)
        
        logger.warning(f"Using generic fallback image: {generic_fallback}")
        return str(generic_fallback)
    
    def generate_image(self, digipal: DigiPal, force_regenerate: bool = False) -> str:
        """
        Generate or retrieve cached image for DigiPal.
        
        Args:
            digipal: DigiPal instance to generate image for
            force_regenerate: Force regeneration even if cached image exists
            
        Returns:
            Path to generated or cached image file
        """
        try:
            # Generate prompt
            prompt = self.generate_prompt(digipal)
            
            # Check cache first (unless force regenerate)
            cache_key = self._get_cache_key(prompt, self.generation_params)
            
            if not force_regenerate:
                cached_path = self._get_cached_image_path(cache_key)
                if cached_path:
                    return str(cached_path)
            
            # Load model if not already loaded
            self._load_model()
            
            # Generate image
            logger.info(f"Generating image for {digipal.name} ({digipal.life_stage.value})")
            
            generator = torch.Generator("cpu").manual_seed(
                hash(digipal.id) % (2**32)  # Consistent seed based on DigiPal ID
            )
            
            image = self.pipe(
                prompt,
                generator=generator,
                **self.generation_params
            ).images[0]
            
            # Save to cache
            cache_path = self._save_to_cache(image, cache_key)
            
            # Update DigiPal with new image info
            digipal.current_image_path = str(cache_path)
            digipal.image_generation_prompt = prompt
            
            return str(cache_path)
            
        except Exception as e:
            logger.error(f"Image generation failed for {digipal.name}: {e}")
            
            # Return fallback image
            fallback_path = self._get_fallback_image(digipal)
            digipal.current_image_path = fallback_path
            digipal.image_generation_prompt = f"Fallback image for {digipal.life_stage.value} {digipal.egg_type.value}"
            
            return fallback_path
    
    def update_image_for_evolution(self, digipal: DigiPal) -> str:
        """
        Generate new image when DigiPal evolves to new life stage.
        
        Args:
            digipal: DigiPal that has evolved
            
        Returns:
            Path to new image file
        """
        logger.info(f"Generating evolution image for {digipal.name} -> {digipal.life_stage.value}")
        return self.generate_image(digipal, force_regenerate=True)
    
    def cleanup_cache(self, max_age_days: int = 30):
        """
        Clean up old cached images.
        
        Args:
            max_age_days: Maximum age of cached images in days
        """
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
            for image_file in self.cache_dir.glob("*.png"):
                file_age = current_time - datetime.fromtimestamp(image_file.stat().st_mtime)
                
                if file_age.days > max_age_days:
                    image_file.unlink()
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old cached images")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the image cache."""
        try:
            cache_files = list(self.cache_dir.glob("*.png"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cache_dir": str(self.cache_dir),
                "cached_images": len(cache_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "model_loaded": self._model_loaded
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {"error": str(e)}