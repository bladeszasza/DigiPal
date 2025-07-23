#!/usr/bin/env python3
"""
Evolution System Demo - Demonstrates DigiPal evolution and inheritance mechanics.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import digipal modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from digipal.core.models import DigiPal
from digipal.core.enums import EggType, LifeStage, AttributeType
from digipal.core.evolution_controller import EvolutionController
from digipal.core.attribute_engine import AttributeEngine


def print_pet_status(pet: DigiPal, title: str = "Pet Status"):
    """Print detailed pet status information."""
    print(f"\n=== {title} ===")
    print(f"Name: {pet.name}")
    print(f"Life Stage: {pet.life_stage.value.title()}")
    print(f"Generation: {pet.generation}")
    print(f"Age: {pet.get_age_hours():.1f} hours")
    print(f"Egg Type: {pet.egg_type.value.title()}")
    print(f"\nAttributes:")
    print(f"  HP: {pet.hp}")
    print(f"  MP: {pet.mp}")
    print(f"  Offense: {pet.offense}")
    print(f"  Defense: {pet.defense}")
    print(f"  Speed: {pet.speed}")
    print(f"  Brains: {pet.brains}")
    print(f"\nSecondary Attributes:")
    print(f"  Happiness: {pet.happiness}")
    print(f"  Discipline: {pet.discipline}")
    print(f"  Weight: {pet.weight}")
    print(f"  Energy: {pet.energy}")
    print(f"  Care Mistakes: {pet.care_mistakes}")
    print(f"\nLearned Commands: {', '.join(sorted(pet.learned_commands))}")


def demonstrate_evolution_path():
    """Demonstrate complete evolution path from baby to elderly."""
    print("🌟 DigiPal Evolution System Demo 🌟")
    print("=" * 50)
    
    # Create evolution controller and attribute engine
    evolution_controller = EvolutionController()
    attribute_engine = AttributeEngine()
    
    # Create a new DigiPal
    pet = DigiPal(
        user_id="demo_user",
        name="EvoPal",
        egg_type=EggType.BLUE,  # Water-oriented
        life_stage=LifeStage.BABY,
        birth_time=datetime.now() - timedelta(hours=1)
    )
    
    print_pet_status(pet, "Initial Baby DigiPal")
    
    # Simulate good care to meet evolution requirements
    print("\n🎯 Providing good care to meet evolution requirements...")
    
    # Apply some training and care
    attribute_engine.apply_care_action(pet, "strength_training")
    attribute_engine.apply_care_action(pet, "defense_training")
    attribute_engine.apply_care_action(pet, "brain_training")
    attribute_engine.apply_care_action(pet, "meat")  # Feed to restore energy and increase attributes
    attribute_engine.apply_care_action(pet, "rest")  # Rest to restore energy
    attribute_engine.apply_care_action(pet, "praise")  # Increase happiness
    
    # Simulate time passing for evolution eligibility
    pet.birth_time = datetime.now() - timedelta(hours=25)  # Make pet old enough
    
    print(f"\nAfter care - Happiness: {pet.happiness}, Care Mistakes: {pet.care_mistakes}")
    
    # Demonstrate evolution through all stages
    stages_to_evolve = [LifeStage.CHILD, LifeStage.TEEN, LifeStage.YOUNG_ADULT, LifeStage.ADULT, LifeStage.ELDERLY]
    
    for target_stage in stages_to_evolve:
        print(f"\n🔄 Attempting evolution to {target_stage.value.title()}...")
        
        # Check evolution eligibility
        eligible, next_stage, requirements_status = evolution_controller.check_evolution_eligibility(pet)
        
        print(f"Evolution eligible: {eligible}")
        if not eligible:
            print("Requirements not met:")
            for req, status in requirements_status.items():
                if not status:
                    print(f"  ❌ {req}")
            print("Forcing evolution for demo purposes...")
        
        # Trigger evolution (force if necessary for demo)
        result = evolution_controller.trigger_evolution(pet, force=not eligible)
        
        if result.success:
            print(f"✅ Evolution successful: {result.old_stage.value.title()} → {result.new_stage.value.title()}")
            print(f"Attribute changes: {result.attribute_changes}")
            print_pet_status(pet, f"After Evolution to {result.new_stage.value.title()}")
        else:
            print(f"❌ Evolution failed: {result.message}")
            break
        
        # Provide more care between evolutions
        if pet.life_stage != LifeStage.ELDERLY:
            attribute_engine.apply_care_action(pet, "rest")
            attribute_engine.apply_care_action(pet, "meat")
            pet.birth_time -= timedelta(hours=50)  # Age the pet for next evolution
    
    return pet


def demonstrate_inheritance():
    """Demonstrate generational inheritance system."""
    print("\n\n🧬 Generational Inheritance Demo 🧬")
    print("=" * 50)
    
    evolution_controller = EvolutionController()
    attribute_engine = AttributeEngine()
    
    # Create a parent with high attributes (simulating good care)
    parent = DigiPal(
        user_id="parent_user",
        name="ParentPal",
        egg_type=EggType.RED,  # Fire-oriented
        life_stage=LifeStage.ADULT,
        generation=1,
        hp=250,
        mp=150,
        offense=80,
        defense=60,
        speed=70,
        brains=65,
        happiness=85,
        discipline=55,
        care_mistakes=3
    )
    
    print_pet_status(parent, "High-Quality Parent DigiPal")
    
    # Assess care quality
    care_assessment = attribute_engine.get_care_quality_assessment(parent)
    care_quality = care_assessment.get('care_quality', 'fair')
    
    print(f"\nCare Quality Assessment: {care_quality}")
    print("Care Assessment Details:")
    for aspect, rating in care_assessment.items():
        print(f"  {aspect}: {rating}")
    
    # Create inheritance DNA
    print(f"\n🧬 Creating inheritance DNA with '{care_quality}' care quality...")
    dna = evolution_controller.create_inheritance_dna(parent, care_quality)
    
    print(f"DNA Generation: {dna.generation}")
    print(f"Parent Final Stage: {dna.parent_final_stage.value.title()}")
    print(f"Parent Egg Type: {dna.parent_egg_type.value.title()}")
    print(f"Inheritance Bonuses:")
    for attr, bonus in dna.inheritance_bonuses.items():
        print(f"  {attr.value}: +{bonus}")
    
    # Create offspring
    offspring = DigiPal(
        user_id="offspring_user",
        name="OffspringPal",
        egg_type=EggType.RED,  # Same as parent
        life_stage=LifeStage.EGG
    )
    
    print_pet_status(offspring, "Offspring Before Inheritance")
    
    # Apply inheritance
    print(f"\n🔄 Applying inheritance to offspring...")
    evolution_controller.apply_inheritance(offspring, dna)
    
    print_pet_status(offspring, "Offspring After Inheritance")
    
    # Compare parent and offspring
    print(f"\n📊 Parent vs Offspring Comparison:")
    attributes = [AttributeType.HP, AttributeType.MP, AttributeType.OFFENSE, 
                 AttributeType.DEFENSE, AttributeType.SPEED, AttributeType.BRAINS]
    
    for attr in attributes:
        parent_val = parent.get_attribute(attr)
        offspring_val = offspring.get_attribute(attr)
        inheritance_bonus = dna.inheritance_bonuses.get(attr, 0)
        print(f"  {attr.value.title()}: Parent={parent_val}, Offspring={offspring_val} (bonus: +{inheritance_bonus})")
    
    return parent, offspring


def demonstrate_time_based_evolution():
    """Demonstrate time-based evolution triggers."""
    print("\n\n⏰ Time-Based Evolution Demo ⏰")
    print("=" * 50)
    
    evolution_controller = EvolutionController()
    
    # Show evolution timings
    print("Evolution Timings (hours):")
    timings = evolution_controller.get_all_evolution_timings()
    for stage, hours in timings.items():
        print(f"  {stage.value.title()}: {hours}")
    
    # Create a pet and test time-based evolution
    pet = DigiPal(
        user_id="time_user",
        name="TimePal",
        egg_type=EggType.GREEN,
        life_stage=LifeStage.BABY,
        birth_time=datetime.now() - timedelta(hours=10)  # 10 hours old
    )
    
    print(f"\nPet Age: {pet.get_age_hours():.1f} hours")
    print(f"Time-based evolution ready: {evolution_controller.check_time_based_evolution(pet)}")
    
    # Age the pet to trigger time-based evolution
    pet.birth_time = datetime.now() - timedelta(hours=25)  # 25 hours old
    print(f"\nAfter aging to {pet.get_age_hours():.1f} hours:")
    print(f"Time-based evolution ready: {evolution_controller.check_time_based_evolution(pet)}")
    
    # Show evolution requirements
    child_requirements = evolution_controller.get_evolution_requirements(LifeStage.CHILD)
    if child_requirements:
        print(f"\nChild Evolution Requirements:")
        print(f"  Min Age: {child_requirements.min_age_hours} hours")
        print(f"  Max Care Mistakes: {child_requirements.max_care_mistakes}")
        print(f"  Happiness Threshold: {child_requirements.happiness_threshold}")
        print(f"  Min Attributes: {child_requirements.min_attributes}")


def demonstrate_death_and_rebirth():
    """Demonstrate death and rebirth cycle."""
    print("\n\n💀 Death and Rebirth Cycle Demo 💀")
    print("=" * 50)
    
    evolution_controller = EvolutionController()
    
    # Create an elderly pet near death
    elderly_pet = DigiPal(
        user_id="elderly_user",
        name="ElderPal",
        egg_type=EggType.BLUE,
        life_stage=LifeStage.ELDERLY,
        generation=1,
        hp=180,
        mp=120,
        offense=45,
        defense=70,
        speed=25,  # Reduced due to age
        brains=90,  # Increased wisdom
        birth_time=datetime.now() - timedelta(hours=700)  # Very old
    )
    
    print_pet_status(elderly_pet, "Elderly DigiPal Near Death")
    
    # Check if it's death time
    is_death_time = evolution_controller.is_death_time(elderly_pet)
    print(f"\nIs death time: {is_death_time}")
    
    if is_death_time:
        print("💀 DigiPal has reached the end of its natural lifespan")
        
        # Create inheritance for next generation
        care_quality = "good"  # Assume good care
        dna = evolution_controller.create_inheritance_dna(elderly_pet, care_quality)
        
        print(f"\n🌱 Creating next generation (Generation {dna.generation})...")
        
        # Create new egg with inheritance
        new_egg = DigiPal(
            user_id="elderly_user",
            name="NewGenPal",
            egg_type=elderly_pet.egg_type,
            life_stage=LifeStage.EGG,
            generation=dna.generation
        )
        
        evolution_controller.apply_inheritance(new_egg, dna)
        
        print_pet_status(new_egg, f"Generation {dna.generation} Egg with Inheritance")
        
        print(f"\n🔄 The cycle continues with improved genetics!")


def main():
    """Run all evolution system demonstrations."""
    try:
        # Demonstrate complete evolution path
        evolved_pet = demonstrate_evolution_path()
        
        # Demonstrate inheritance system
        parent, offspring = demonstrate_inheritance()
        
        # Demonstrate time-based evolution
        demonstrate_time_based_evolution()
        
        # Demonstrate death and rebirth
        demonstrate_death_and_rebirth()
        
        print("\n\n🎉 Evolution System Demo Complete! 🎉")
        print("=" * 50)
        print("The evolution system successfully demonstrates:")
        print("✅ Life stage progression with requirements")
        print("✅ Time-based evolution triggers")
        print("✅ Attribute changes during evolution")
        print("✅ Egg type specific bonuses")
        print("✅ Generational inheritance with DNA")
        print("✅ Care quality impact on inheritance")
        print("✅ Death and rebirth cycles")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()