"""
Demo script for DigiPal image generation system.
Showcases image generation for different life stages and egg types.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import digipal
sys.path.insert(0, str(Path(__file__).parent.parent))

from digipal.ai.image_generator import ImageGenerator
from digipal.core.models import DigiPal
from digipal.core.enums import LifeStage, EggType


def demo_prompt_generation():
    """Demonstrate prompt generation for different DigiPal configurations."""
    print("=== DigiPal Image Generation Demo ===\n")
    
    # Initialize image generator
    generator = ImageGenerator()
    
    print("1. Prompt Generation Examples\n")
    
    # Demo different life stages
    print("Life Stage Progression (Red Egg):")
    for stage in LifeStage:
        digipal = DigiPal(
            name=f"RedPal-{stage.value}",
            egg_type=EggType.RED,
            life_stage=stage,
            offense=60,
            defense=40,
            happiness=75
        )
        
        prompt = generator.generate_prompt(digipal)
        print(f"  {stage.value.upper()}: {prompt[:100]}...")
    
    print("\n" + "="*80 + "\n")
    
    # Demo different egg types
    print("Egg Type Variations (Adult Stage):")
    for egg_type in EggType:
        digipal = DigiPal(
            name=f"{egg_type.value}Pal",
            egg_type=egg_type,
            life_stage=LifeStage.ADULT,
            offense=50,
            defense=50,
            happiness=60
        )
        
        prompt = generator.generate_prompt(digipal)
        print(f"  {egg_type.value.upper()}: {prompt[:100]}...")
    
    print("\n" + "="*80 + "\n")
    
    # Demo attribute influence on prompts
    print("Attribute Influence on Prompts:")
    
    # High offense DigiPal
    fierce_pal = DigiPal(
        name="FiercePal",
        offense=90,
        defense=30,
        speed=40,
        brains=35,
        happiness=50
    )
    print(f"  HIGH OFFENSE: {generator.generate_prompt(fierce_pal)[:100]}...")
    
    # High defense DigiPal
    tank_pal = DigiPal(
        name="TankPal",
        offense=30,
        defense=90,
        speed=25,
        brains=40,
        happiness=50
    )
    print(f"  HIGH DEFENSE: {generator.generate_prompt(tank_pal)[:100]}...")
    
    # High speed DigiPal
    speedy_pal = DigiPal(
        name="SpeedyPal",
        offense=40,
        defense=30,
        speed=90,
        brains=35,
        happiness=50
    )
    print(f"  HIGH SPEED: {generator.generate_prompt(speedy_pal)[:100]}...")
    
    # High brains DigiPal
    smart_pal = DigiPal(
        name="SmartPal",
        offense=35,
        defense=40,
        speed=30,
        brains=90,
        happiness=50
    )
    print(f"  HIGH BRAINS: {generator.generate_prompt(smart_pal)[:100]}...")
    
    print("\n" + "="*80 + "\n")


def demo_cache_system():
    """Demonstrate the caching system."""
    print("2. Cache System Demo\n")
    
    generator = ImageGenerator()
    
    # Show cache info
    cache_info = generator.get_cache_info()
    print("Cache Information:")
    for key, value in cache_info.items():
        print(f"  {key}: {value}")
    
    print("\nCache Key Generation:")
    
    # Demo cache key generation
    sample_pal = DigiPal(name="CachePal", egg_type=EggType.BLUE)
    prompt = generator.generate_prompt(sample_pal)
    cache_key = generator._get_cache_key(prompt, generator.generation_params)
    
    print(f"  Prompt: {prompt[:50]}...")
    print(f"  Cache Key: {cache_key}")
    
    # Show that same DigiPal produces same cache key
    same_pal = DigiPal(name="CachePal", egg_type=EggType.BLUE)
    same_prompt = generator.generate_prompt(same_pal)
    same_cache_key = generator._get_cache_key(same_prompt, generator.generation_params)
    
    print(f"  Same DigiPal Cache Key: {same_cache_key}")
    print(f"  Keys Match: {cache_key == same_cache_key}")
    
    print("\n" + "="*80 + "\n")


def demo_fallback_system():
    """Demonstrate the fallback image system."""
    print("3. Fallback System Demo\n")
    
    generator = ImageGenerator()
    
    print("Fallback Images Available:")
    for stage in LifeStage:
        for egg_type in EggType:
            key = f"{stage.value}_{egg_type.value}"
            fallback_path = generator.fallback_images.get(key)
            exists = Path(fallback_path).exists() if fallback_path else False
            print(f"  {key}: {fallback_path} ({'✓' if exists else '✗'})")
    
    print("\nTesting Fallback Retrieval:")
    test_pal = DigiPal(
        name="FallbackTest",
        egg_type=EggType.GREEN,
        life_stage=LifeStage.TEEN
    )
    
    fallback_path = generator._get_fallback_image(test_pal)
    print(f"  DigiPal: {test_pal.name} ({test_pal.life_stage.value}, {test_pal.egg_type.value})")
    print(f"  Fallback Path: {fallback_path}")
    print(f"  File Exists: {'✓' if Path(fallback_path).exists() else '✗'}")
    
    print("\n" + "="*80 + "\n")


def demo_evolution_scenario():
    """Demonstrate image generation through evolution stages."""
    print("4. Evolution Scenario Demo\n")
    
    generator = ImageGenerator()
    
    # Create a DigiPal and simulate evolution
    evolving_pal = DigiPal(
        name="EvolvePal",
        egg_type=EggType.RED,
        life_stage=LifeStage.EGG,
        offense=45,
        defense=35,
        speed=55,
        brains=40,
        happiness=70
    )
    
    print(f"Following {evolving_pal.name} through evolution:")
    print(f"Initial Stats - Offense: {evolving_pal.offense}, Defense: {evolving_pal.defense}, "
          f"Speed: {evolving_pal.speed}, Brains: {evolving_pal.brains}, Happiness: {evolving_pal.happiness}")
    
    evolution_stages = [LifeStage.EGG, LifeStage.BABY, LifeStage.CHILD, 
                       LifeStage.TEEN, LifeStage.YOUNG_ADULT, LifeStage.ADULT]
    
    for i, stage in enumerate(evolution_stages):
        evolving_pal.life_stage = stage
        
        # Simulate attribute growth
        if i > 0:  # Don't modify egg stage
            evolving_pal.offense += 5
            evolving_pal.defense += 3
            evolving_pal.speed += 4
            evolving_pal.brains += 6
        
        prompt = generator.generate_prompt(evolving_pal)
        print(f"\n  Stage {i+1} - {stage.value.upper()}:")
        print(f"    Stats: O:{evolving_pal.offense} D:{evolving_pal.defense} "
              f"S:{evolving_pal.speed} B:{evolving_pal.brains}")
        print(f"    Prompt: {prompt[:80]}...")
        
        # In a real scenario, this would generate the actual image
        # For demo, we just show what would happen
        print(f"    → Would generate/cache image for evolution stage")
    
    print("\n" + "="*80 + "\n")


def demo_image_generation_simulation():
    """Simulate the image generation process without actually generating images."""
    print("5. Image Generation Process Simulation\n")
    
    generator = ImageGenerator()
    
    test_pal = DigiPal(
        name="TestGenPal",
        egg_type=EggType.BLUE,
        life_stage=LifeStage.YOUNG_ADULT,
        offense=65,
        defense=70,
        speed=55,
        brains=60,
        happiness=85
    )
    
    print(f"Simulating image generation for {test_pal.name}:")
    print(f"  Life Stage: {test_pal.life_stage.value}")
    print(f"  Egg Type: {test_pal.egg_type.value}")
    print(f"  Attributes: O:{test_pal.offense} D:{test_pal.defense} S:{test_pal.speed} B:{test_pal.brains}")
    print(f"  Happiness: {test_pal.happiness}")
    
    # Generate prompt
    prompt = generator.generate_prompt(test_pal)
    print(f"\n  Generated Prompt:")
    print(f"    {prompt}")
    
    # Show cache key
    cache_key = generator._get_cache_key(prompt, generator.generation_params)
    print(f"\n  Cache Key: {cache_key}")
    
    # Show generation parameters
    print(f"\n  Generation Parameters:")
    for param, value in generator.generation_params.items():
        print(f"    {param}: {value}")
    
    # Show what would happen in actual generation
    print(f"\n  Generation Process:")
    print(f"    1. Check cache for key: {cache_key}")
    print(f"    2. If not cached, load FLUX.1-dev model")
    print(f"    3. Generate image with consistent seed based on DigiPal ID")
    print(f"    4. Save to cache: {generator.cache_dir}/{cache_key}.png")
    print(f"    5. Update DigiPal image path and prompt")
    print(f"    6. Return image path")
    
    # Show fallback process
    fallback_path = generator._get_fallback_image(test_pal)
    print(f"\n  Fallback Process (if generation fails):")
    print(f"    1. Retrieve fallback image: {fallback_path}")
    print(f"    2. Update DigiPal with fallback path")
    print(f"    3. Return fallback path")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run all demo functions."""
    print("DigiPal Image Generation System Demo")
    print("=" * 80)
    print("This demo showcases the image generation capabilities")
    print("without actually generating images (to avoid model loading).")
    print("=" * 80 + "\n")
    
    try:
        demo_prompt_generation()
        demo_cache_system()
        demo_fallback_system()
        demo_evolution_scenario()
        demo_image_generation_simulation()
        
        print("Demo completed successfully!")
        print("\nTo actually generate images:")
        print("1. Install required dependencies: pip install -U diffusers torch")
        print("2. Ensure you have sufficient GPU memory or enable CPU offload")
        print("3. Call generator.generate_image(digipal) with a real DigiPal instance")
        
    except Exception as e:
        print(f"Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()