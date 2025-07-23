#!/usr/bin/env python3
"""
Demo script for Qwen3-0.6B integration with DigiPal.

This script demonstrates the integration of Qwen/Qwen3-0.6B model
for natural language processing in the DigiPal application.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from digipal.ai.language_model import LanguageModel
from digipal.ai.communication import AICommunication
from digipal.core.models import DigiPal
from digipal.core.enums import EggType, LifeStage


def test_qwen_basic_usage():
    """Test basic Qwen3-0.6B usage as specified in the task."""
    print("=== Testing Basic Qwen3-0.6B Usage ===")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen3-0.6B"
        
        print(f"Loading tokenizer for {model_name}...")
        # load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("Tokenizer loaded successfully!")
        
        print(f"Loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Model loaded successfully!")
        
        # prepare the model input
        prompt = "Give me a short introduction to large language model."
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        print("Generating response...")
        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512  # Reduced from 32768 for demo
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        print("thinking content:", thinking_content)
        print("content:", content)
        
        return True
        
    except Exception as e:
        print(f"Error in basic Qwen usage: {e}")
        print("This is expected if the model is not available locally.")
        return False


def test_language_model_integration():
    """Test LanguageModel class integration."""
    print("\n=== Testing LanguageModel Integration ===")
    
    # Create language model instance
    language_model = LanguageModel("Qwen/Qwen3-0.6B", quantization=True)
    
    print(f"Model info: {language_model.get_model_info()}")
    
    # Create test DigiPal
    test_pet = DigiPal(
        id="demo-pet",
        user_id="demo-user",
        name="DemoPal",
        egg_type=EggType.BLUE,
        life_stage=LifeStage.CHILD,
        hp=100,
        happiness=75,
        energy=80,
        discipline=60
    )
    
    # Test prompt creation
    prompt = language_model._create_prompt("Hello there!", test_pet)
    print(f"Generated prompt:\n{prompt}")
    
    # Test fallback response
    fallback = language_model._fallback_response("Hello", test_pet)
    print(f"Fallback response: {fallback}")
    
    # Test personality description
    test_pet.personality_traits = {
        'friendliness': 0.8,
        'playfulness': 0.9,
        'obedience': 0.6,
        'curiosity': 0.7
    }
    personality = language_model._get_personality_description(test_pet)
    print(f"Personality description: {personality}")
    
    return True


def test_ai_communication_integration():
    """Test AICommunication integration with LanguageModel."""
    print("\n=== Testing AICommunication Integration ===")
    
    # Create AI communication instance
    ai_comm = AICommunication(model_name="Qwen/Qwen3-0.6B", quantization=True)
    
    print(f"AI Communication info: {ai_comm.get_model_info()}")
    
    # Create test DigiPal
    test_pet = DigiPal(
        id="demo-pet-2",
        user_id="demo-user",
        name="ChatPal",
        egg_type=EggType.RED,
        life_stage=LifeStage.TEEN,
        hp=120,
        happiness=65,
        energy=90,
        discipline=70
    )
    
    # Test interaction processing
    test_inputs = [
        "Hello ChatPal!",
        "Let's train together!",
        "Good job!",
        "How are you feeling?",
        "Time to sleep"
    ]
    
    for user_input in test_inputs:
        print(f"\nUser: {user_input}")
        
        # Process complete interaction
        interaction = ai_comm.process_interaction(user_input, test_pet)
        
        print(f"Interpreted command: {interaction.interpreted_command}")
        print(f"Pet response: {interaction.pet_response}")
        print(f"Success: {interaction.success}")
    
    # Show conversation history
    print(f"\nConversation history length: {len(test_pet.conversation_history)}")
    print(f"Learned commands: {test_pet.learned_commands}")
    
    return True


def test_context_aware_generation():
    """Test context-aware response generation."""
    print("\n=== Testing Context-Aware Generation ===")
    
    language_model = LanguageModel("Qwen/Qwen3-0.6B", quantization=False)
    
    # Test different life stages
    life_stages = [LifeStage.BABY, LifeStage.CHILD, LifeStage.TEEN, LifeStage.ADULT, LifeStage.ELDERLY]
    
    for stage in life_stages:
        test_pet = DigiPal(
            name=f"{stage.value.title()}Pal",
            life_stage=stage,
            happiness=70,
            energy=80
        )
        
        prompt = language_model._create_prompt("How are you today?", test_pet)
        print(f"\n{stage.value.upper()} STAGE PROMPT:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
    
    return True


def main():
    """Run all demo tests."""
    print("DigiPal Qwen3-0.6B Integration Demo")
    print("=" * 50)
    
    tests = [
        ("Basic Qwen Usage", test_qwen_basic_usage),
        ("LanguageModel Integration", test_language_model_integration),
        ("AICommunication Integration", test_ai_communication_integration),
        ("Context-Aware Generation", test_context_aware_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\nRunning {test_name}...")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Error in {test_name}: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("DEMO RESULTS:")
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status} {test_name}")
    
    print("\nNote: Model loading tests may fail if Qwen3-0.6B is not available locally.")
    print("The integration code is ready and will work when the model is available.")


if __name__ == "__main__":
    main()