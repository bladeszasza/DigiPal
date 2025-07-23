#!/usr/bin/env python3
"""
Demo script showcasing the AI communication layer functionality.

This script demonstrates:
- Command interpretation across different life stages
- Response generation based on pet state
- Conversation memory management
- Personality trait development
"""

from digipal.ai.communication import AICommunication
from digipal.core.models import DigiPal
from digipal.core.enums import EggType, LifeStage


def demo_command_interpretation():
    """Demonstrate command interpretation across life stages."""
    print("=== Command Interpretation Demo ===")
    
    ai_comm = AICommunication()
    
    # Test commands across different life stages
    stages_to_test = [
        (LifeStage.BABY, "Baby DigiPal"),
        (LifeStage.CHILD, "Child DigiPal"),
        (LifeStage.TEEN, "Teen DigiPal"),
        (LifeStage.ADULT, "Adult DigiPal")
    ]
    
    test_commands = [
        "eat some food",
        "let's train",
        "time to sleep",
        "good job!",
        "show me your status"
    ]
    
    for stage, stage_name in stages_to_test:
        print(f"\n{stage_name} ({stage.value}):")
        pet = DigiPal(life_stage=stage, happiness=60)
        
        for command_text in test_commands:
            command = ai_comm.interpret_command(command_text, pet)
            status = "✓" if command.stage_appropriate else "✗"
            print(f"  {status} '{command_text}' -> {command.action} (energy: {command.energy_required})")


def demo_response_generation():
    """Demonstrate response generation for different life stages."""
    print("\n=== Response Generation Demo ===")
    
    ai_comm = AICommunication()
    
    # Create pets at different life stages
    pets = [
        (DigiPal(life_stage=LifeStage.BABY, happiness=70), "Happy Baby"),
        (DigiPal(life_stage=LifeStage.CHILD, happiness=30), "Sad Child"),
        (DigiPal(life_stage=LifeStage.ADULT, happiness=80), "Happy Adult")
    ]
    
    test_inputs = ["eat", "good job", "let's play", "unknown command"]
    
    for pet, pet_description in pets:
        print(f"\n{pet_description}:")
        for input_text in test_inputs:
            response = ai_comm.generate_response(input_text, pet)
            print(f"  User: '{input_text}' -> Pet: '{response}'")


def demo_conversation_flow():
    """Demonstrate a complete conversation flow with memory."""
    print("\n=== Conversation Flow Demo ===")
    
    ai_comm = AICommunication()
    pet = DigiPal(
        name="Buddy",
        life_stage=LifeStage.CHILD,
        happiness=50,
        egg_type=EggType.BLUE
    )
    
    print(f"Starting conversation with {pet.name} (Child stage, happiness: {pet.happiness})")
    
    # Simulate a conversation
    conversation = [
        "hello there",
        "let's eat something",
        "good job eating!",
        "time to play",
        "let's train together",
        "you're doing great!",
        "how are you feeling?"
    ]
    
    for i, user_input in enumerate(conversation, 1):
        print(f"\nTurn {i}:")
        print(f"User: {user_input}")
        
        interaction = ai_comm.process_interaction(user_input, pet)
        print(f"Pet: {interaction.pet_response}")
        print(f"Command understood: {interaction.interpreted_command}")
        print(f"Success: {interaction.success}")
        
        # Show learned commands
        if len(pet.learned_commands) > 0:
            print(f"Learned commands: {sorted(pet.learned_commands)}")
    
    # Show conversation summary
    summary = ai_comm.memory_manager.get_interaction_summary(pet)
    print(f"\n=== Conversation Summary ===")
    print(f"Total interactions: {summary['total_interactions']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Most common commands: {summary['most_common_commands'][:3]}")
    
    # Show personality development
    print(f"\n=== Personality Traits ===")
    for trait, value in pet.personality_traits.items():
        print(f"{trait.capitalize()}: {value:.2f}")


def demo_personality_development():
    """Demonstrate personality trait development over time."""
    print("\n=== Personality Development Demo ===")
    
    ai_comm = AICommunication()
    pet = DigiPal(life_stage=LifeStage.CHILD, happiness=50)
    
    # Initialize personality traits
    pet.personality_traits = {
        'friendliness': 0.5,
        'playfulness': 0.5,
        'obedience': 0.5,
        'curiosity': 0.5
    }
    
    print("Initial personality traits:")
    for trait, value in pet.personality_traits.items():
        print(f"  {trait}: {value:.2f}")
    
    # Simulate interactions that should affect personality
    training_sessions = [
        ("good job!", 3, "praise"),
        ("let's play", 2, "play"),
        ("bad behavior", 1, "scolding"),
        ("what's this?", 2, "curiosity")
    ]
    
    for command, count, interaction_type in training_sessions:
        print(f"\n{interaction_type.capitalize()} training ({count} times): '{command}'")
        
        for _ in range(count):
            ai_comm.process_interaction(command, pet)
        
        print("Updated personality traits:")
        for trait, value in pet.personality_traits.items():
            print(f"  {trait}: {value:.2f}")


def main():
    """Run all demo functions."""
    print("DigiPal AI Communication Layer Demo")
    print("=" * 50)
    
    demo_command_interpretation()
    demo_response_generation()
    demo_conversation_flow()
    demo_personality_development()
    
    print("\n" + "=" * 50)
    print("Demo completed! The AI communication layer provides:")
    print("✓ Command interpretation based on life stage")
    print("✓ Contextual response generation")
    print("✓ Conversation memory management")
    print("✓ Personality trait development")
    print("✓ Stage-appropriate interaction handling")


if __name__ == "__main__":
    main()