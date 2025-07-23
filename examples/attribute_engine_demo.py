#!/usr/bin/env python3
"""
Demonstration of the AttributeEngine and care mechanics.
"""

from digipal.core import DigiPal, AttributeEngine, EggType, LifeStage, AttributeType


def main():
    """Demonstrate the AttributeEngine functionality."""
    print("=== DigiPal AttributeEngine Demo ===\n")
    
    # Create a new DigiPal
    pet = DigiPal(
        user_id="demo_user",
        name="DemoPal",
        egg_type=EggType.RED,
        life_stage=LifeStage.CHILD
    )
    
    # Create the AttributeEngine
    engine = AttributeEngine()
    
    print(f"Initial DigiPal Stats:")
    print(f"  Name: {pet.name}")
    print(f"  Life Stage: {pet.life_stage.value}")
    print(f"  HP: {pet.hp}, MP: {pet.mp}")
    print(f"  Offense: {pet.offense}, Defense: {pet.defense}")
    print(f"  Speed: {pet.speed}, Brains: {pet.brains}")
    print(f"  Energy: {pet.energy}, Happiness: {pet.happiness}")
    print(f"  Weight: {pet.weight}, Discipline: {pet.discipline}")
    print(f"  Care Mistakes: {pet.care_mistakes}\n")
    
    # Show available actions
    available_actions = engine.get_available_actions(pet)
    print(f"Available Actions: {', '.join(available_actions)}\n")
    
    # Show care quality assessment
    assessment = engine.get_care_quality_assessment(pet)
    print(f"Care Quality Assessment:")
    for metric, value in assessment.items():
        print(f"  {metric}: {value}")
    print()
    
    # Demonstrate training
    print("=== Training Demonstration ===")
    print("Performing strength training...")
    success, interaction = engine.apply_care_action(pet, "strength_training")
    print(f"Success: {success}")
    print(f"Response: {interaction.pet_response}")
    print(f"Attribute Changes: {interaction.attribute_changes}")
    print(f"New Stats - Offense: {pet.offense}, HP: {pet.hp}, Energy: {pet.energy}, Happiness: {pet.happiness}\n")
    
    # Demonstrate feeding
    print("=== Feeding Demonstration ===")
    print("Feeding meat...")
    success, interaction = engine.apply_care_action(pet, "meat")
    print(f"Success: {success}")
    print(f"Response: {interaction.pet_response}")
    print(f"Attribute Changes: {interaction.attribute_changes}")
    print(f"New Stats - Weight: {pet.weight}, HP: {pet.hp}, Offense: {pet.offense}, Happiness: {pet.happiness}\n")
    
    # Demonstrate care actions
    print("=== Care Actions Demonstration ===")
    print("Praising the DigiPal...")
    success, interaction = engine.apply_care_action(pet, "praise")
    print(f"Success: {success}")
    print(f"Response: {interaction.pet_response}")
    print(f"Attribute Changes: {interaction.attribute_changes}")
    print(f"New Stats - Happiness: {pet.happiness}, Discipline: {pet.discipline}\n")
    
    # Demonstrate rest
    print("=== Rest Demonstration ===")
    print("Letting DigiPal rest...")
    success, interaction = engine.apply_care_action(pet, "rest")
    print(f"Success: {success}")
    print(f"Response: {interaction.pet_response}")
    print(f"Attribute Changes: {interaction.attribute_changes}")
    print(f"New Stats - Energy: {pet.energy}, Happiness: {pet.happiness}\n")
    
    # Demonstrate time decay
    print("=== Time Decay Demonstration ===")
    print("Simulating 3 hours passing...")
    changes = engine.apply_time_decay(pet, 3.0)
    print(f"Time Decay Changes: {changes}")
    print(f"New Stats - Energy: {pet.energy}, Happiness: {pet.happiness}\n")
    
    # Demonstrate new food types
    print("=== New Food Types Demonstration ===")
    print("Feeding protein shake...")
    success, interaction = engine.apply_care_action(pet, "protein_shake")
    print(f"Success: {success}")
    print(f"Response: {interaction.pet_response}")
    print(f"Attribute Changes: {interaction.attribute_changes}")
    print(f"New Stats - Weight: {pet.weight}, Offense: {pet.offense}, HP: {pet.hp}\n")
    
    print("Feeding energy drink...")
    success, interaction = engine.apply_care_action(pet, "energy_drink")
    print(f"Success: {success}")
    print(f"Response: {interaction.pet_response}")
    print(f"Attribute Changes: {interaction.attribute_changes}")
    print(f"New Stats - Energy: {pet.energy}, Speed: {pet.speed}, MP: {pet.mp}\n")
    
    # Demonstrate advanced training
    print("=== Advanced Training Demonstration ===")
    print("Performing endurance training...")
    success, interaction = engine.apply_care_action(pet, "endurance_training")
    print(f"Success: {success}")
    print(f"Response: {interaction.pet_response}")
    print(f"Attribute Changes: {interaction.attribute_changes}")
    print(f"New Stats - HP: {pet.hp}, Defense: {pet.defense}, Weight: {pet.weight}, Energy: {pet.energy}\n")
    
    print("Performing agility training...")
    success, interaction = engine.apply_care_action(pet, "agility_training")
    print(f"Success: {success}")
    print(f"Response: {interaction.pet_response}")
    print(f"Attribute Changes: {interaction.attribute_changes}")
    print(f"New Stats - Speed: {pet.speed}, Offense: {pet.offense}, Weight: {pet.weight}, Energy: {pet.energy}\n")
    
    # Show updated care quality assessment
    print("=== Updated Care Quality Assessment ===")
    assessment = engine.get_care_quality_assessment(pet)
    for metric, value in assessment.items():
        print(f"  {metric}: {value}")
    print()
    
    # Demonstrate insufficient energy
    print("=== Insufficient Energy Demonstration ===")
    pet.energy = 5  # Set very low energy
    print(f"Current Energy: {pet.energy}")
    print("Attempting strength training with low energy...")
    success, interaction = engine.apply_care_action(pet, "strength_training")
    print(f"Success: {success}")
    print(f"Response: {interaction.pet_response}")
    print(f"Attribute Changes: {interaction.attribute_changes}")
    print(f"Care Mistakes: {pet.care_mistakes}\n")
    
    # Show final stats
    print("=== Final DigiPal Stats ===")
    print(f"  HP: {pet.hp}, MP: {pet.mp}")
    print(f"  Offense: {pet.offense}, Defense: {pet.defense}")
    print(f"  Speed: {pet.speed}, Brains: {pet.brains}")
    print(f"  Energy: {pet.energy}, Happiness: {pet.happiness}")
    print(f"  Weight: {pet.weight}, Discipline: {pet.discipline}")
    print(f"  Care Mistakes: {pet.care_mistakes}")
    print(f"  Interactions: {len(pet.conversation_history)}")


if __name__ == "__main__":
    main()