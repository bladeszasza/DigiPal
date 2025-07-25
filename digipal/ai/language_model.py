"""
Language model integration for DigiPal using Qwen3-0.6B.

This module handles the integration with Qwen/Qwen3-0.6B model for natural language
processing, including model loading, quantization, and context-aware response generation.
"""

import logging
import torch
from typing import Dict, List, Optional, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
from datetime import datetime

from ..core.models import DigiPal, Interaction
from ..core.enums import LifeStage
from ..core.exceptions import AIModelError, NetworkError
from ..core.error_handler import with_error_handling, with_retry, RetryConfig
from .graceful_degradation import with_ai_fallback, ai_service_manager


logger = logging.getLogger(__name__)


class LanguageModel:
    """
    Manages Qwen3-0.6B model for natural language processing with DigiPal context.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", quantization: bool = True):
        """
        Initialize the language model.
        
        Args:
            model_name: HuggingFace model identifier
            quantization: Whether to use quantization for memory optimization
        """
        self.model_name = model_name
        self.quantization = quantization
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize prompt templates
        self.prompt_templates = self._initialize_prompt_templates()
        
        logger.info(f"LanguageModel initialized with model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Quantization: {quantization}")
    
    @with_error_handling(fallback_value=False, context={'operation': 'model_loading'})
    @with_retry(RetryConfig(max_attempts=3, retry_on=[NetworkError, ConnectionError]))
    def load_model(self) -> bool:
        """
        Load the Qwen3-0.6B model and tokenizer.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading tokenizer for {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Configure quantization if enabled
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto"
            }
            
            if self.quantization and torch.cuda.is_available():
                logger.info("Configuring 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
            
            logger.info(f"Loading model {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            logger.info("Model loaded successfully")
            return True
            
        except (ConnectionError, TimeoutError) as e:
            raise NetworkError(f"Network error loading model: {str(e)}")
        except Exception as e:
            raise AIModelError(f"Failed to load model: {str(e)}")
    
    @with_ai_fallback("language_model")
    def generate_response(self, user_input: str, pet: DigiPal, memory_context: str = "", max_tokens: int = 150) -> str:
        """
        Generate contextual response using Qwen3-0.6B model.
        
        Args:
            user_input: User's input text
            pet: DigiPal instance for context
            memory_context: Additional memory context from RAG system
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        if not self.model or not self.tokenizer:
            logger.warning("Model not loaded, using fallback response")
            raise AIModelError("Language model not loaded")
        
        try:
            # Create context-aware prompt with memory context
            prompt = self._create_prompt(user_input, pet, memory_context)
            
            # Prepare messages for chat template
            messages = [
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            
            # Tokenize input
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract generated tokens
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # Parse thinking content and actual response
            thinking_content, content = self._parse_response(output_ids)
            
            # Log thinking content for debugging
            if thinking_content:
                logger.debug(f"Model thinking: {thinking_content[:100]}...")
            
            # Clean and validate response
            response = self._clean_response(content, pet)
            
            logger.debug(f"Generated response: {response}")
            return response
            
        except torch.cuda.OutOfMemoryError as e:
            raise AIModelError(f"GPU memory error: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise AIModelError(f"Language model generation failed: {str(e)}")
    
    def _create_prompt(self, user_input: str, pet: DigiPal, memory_context: str = "") -> str:
        """
        Create context-aware prompt incorporating pet state, personality, and memory context.
        
        Args:
            user_input: User's input text
            pet: DigiPal instance for context
            memory_context: Additional memory context from RAG system
            
        Returns:
            Formatted prompt string
        """
        # Get base template for life stage
        template = self.prompt_templates.get(pet.life_stage, self.prompt_templates[LifeStage.BABY])
        
        # Get recent conversation context
        recent_interactions = pet.conversation_history[-3:] if pet.conversation_history else []
        conversation_context = ""
        if recent_interactions:
            conversation_context = "\n".join([
                f"User: {interaction.user_input}\nDigiPal: {interaction.pet_response}"
                for interaction in recent_interactions
            ])
        
        # Calculate personality description
        personality_desc = self._get_personality_description(pet)
        
        # Format the prompt with memory context
        prompt = template.format(
            name=pet.name,
            life_stage=pet.life_stage.value,
            hp=pet.hp,
            happiness=pet.happiness,
            energy=pet.energy,
            discipline=pet.discipline,
            age_hours=pet.get_age_hours(),
            personality=personality_desc,
            recent_conversation=conversation_context,
            memory_context=memory_context,
            user_input=user_input
        )
        
        return prompt
    
    def _initialize_prompt_templates(self) -> Dict[LifeStage, str]:
        """
        Initialize prompt templates for each life stage.
        
        Returns:
            Dictionary mapping life stages to prompt templates
        """
        return {
            LifeStage.EGG: """
You are a DigiPal egg named {name}. You cannot speak or respond directly, but you can show subtle reactions.
The user said: "{user_input}"
Respond with a very brief description of the egg's reaction (1-2 words or simple action).
""",
            
            LifeStage.BABY: """
You are {name}, a baby DigiPal in the {life_stage} stage. You are {age_hours:.1f} hours old.
Current stats: HP={hp}, Happiness={happiness}, Energy={energy}, Discipline={discipline}
Personality: {personality}

As a baby, you can only understand basic commands: eat, sleep, good, bad.
You communicate with simple baby sounds, single words, and basic emotions.
You are curious, innocent, and learning about the world.

Recent conversation:
{recent_conversation}

{memory_context}

User just said: "{user_input}"

Respond as a baby DigiPal would - keep it simple, innocent, and age-appropriate. Use baby talk, simple words, and express basic emotions.
""",
            
            LifeStage.CHILD: """
You are {name}, a child DigiPal in the {life_stage} stage. You are {age_hours:.1f} hours old.
Current stats: HP={hp}, Happiness={happiness}, Energy={energy}, Discipline={discipline}
Personality: {personality}

As a child, you understand: eat, sleep, good, bad, play, train.
You are energetic, playful, and eager to learn. You speak in simple sentences and show enthusiasm.

Recent conversation:
{recent_conversation}

{memory_context}

User just said: "{user_input}"

Respond as a child DigiPal would - enthusiastic, simple language, and show interest in play and learning.
""",
            
            LifeStage.TEEN: """
You are {name}, a teenage DigiPal in the {life_stage} stage. You are {age_hours:.1f} hours old.
Current stats: HP={hp}, Happiness={happiness}, Energy={energy}, Discipline={discipline}
Personality: {personality}

As a teen, you understand most commands and can have conversations.
You're developing your own personality, sometimes moody, but generally cooperative.
You can be a bit rebellious but still care about your relationship with your caretaker.

Recent conversation:
{recent_conversation}

{memory_context}

User just said: "{user_input}"

Respond as a teenage DigiPal would - more complex thoughts, some attitude, but still caring.
""",
            
            LifeStage.YOUNG_ADULT: """
You are {name}, a young adult DigiPal in the {life_stage} stage. You are {age_hours:.1f} hours old.
Current stats: HP={hp}, Happiness={happiness}, Energy={energy}, Discipline={discipline}
Personality: {personality}

As a young adult, you're confident, capable, and have developed your full personality.
You can engage in complex conversations and understand all commands.
You're at your physical and mental peak, ready for challenges.

Recent conversation:
{recent_conversation}

{memory_context}

User just said: "{user_input}"

Respond as a confident young adult DigiPal - articulate, capable, and engaging.
""",
            
            LifeStage.ADULT: """
You are {name}, an adult DigiPal in the {life_stage} stage. You are {age_hours:.1f} hours old.
Current stats: HP={hp}, Happiness={happiness}, Energy={energy}, Discipline={discipline}
Personality: {personality}

As an adult, you're wise, mature, and thoughtful in your responses.
You have deep understanding and can provide guidance and wisdom.
You're protective and caring, with a strong bond to your caretaker.

Recent conversation:
{recent_conversation}

{memory_context}

User just said: "{user_input}"

Respond as a mature adult DigiPal - wise, thoughtful, and caring.
""",
            
            LifeStage.ELDERLY: """
You are {name}, an elderly DigiPal in the {life_stage} stage. You are {age_hours:.1f} hours old.
Current stats: HP={hp}, Happiness={happiness}, Energy={energy}, Discipline={discipline}
Personality: {personality}

As an elderly DigiPal, you're wise from experience but also nostalgic and gentle.
You move slower but think deeply. You cherish every moment with your caretaker.
You often reflect on memories and share wisdom from your long life.

Recent conversation:
{recent_conversation}

{memory_context}

User just said: "{user_input}"

Respond as an elderly DigiPal - gentle, wise, nostalgic, and deeply caring.
"""
        }
    
    def _get_personality_description(self, pet: DigiPal) -> str:
        """
        Generate personality description from pet's personality traits.
        
        Args:
            pet: DigiPal instance
            
        Returns:
            Human-readable personality description
        """
        if not pet.personality_traits:
            return "developing personality"
        
        traits = []
        
        # Analyze personality traits
        if pet.personality_traits.get('friendliness', 0.5) > 0.7:
            traits.append("very friendly")
        elif pet.personality_traits.get('friendliness', 0.5) < 0.3:
            traits.append("somewhat shy")
        
        if pet.personality_traits.get('playfulness', 0.5) > 0.7:
            traits.append("very playful")
        elif pet.personality_traits.get('playfulness', 0.5) < 0.3:
            traits.append("more serious")
        
        if pet.personality_traits.get('obedience', 0.5) > 0.7:
            traits.append("well-behaved")
        elif pet.personality_traits.get('obedience', 0.5) < 0.3:
            traits.append("a bit rebellious")
        
        if pet.personality_traits.get('curiosity', 0.5) > 0.7:
            traits.append("very curious")
        
        return ", ".join(traits) if traits else "balanced personality"
    
    def _parse_response(self, output_ids: List[int]) -> Tuple[str, str]:
        """
        Parse thinking content and actual response from model output.
        
        Args:
            output_ids: Generated token IDs
            
        Returns:
            Tuple of (thinking_content, actual_response)
        """
        try:
            # Look for thinking end token (151668 = </think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            # No thinking content found
            index = 0
        
        thinking_content = ""
        content = ""
        
        if index > 0:
            thinking_content = self.tokenizer.decode(
                output_ids[:index], 
                skip_special_tokens=True
            ).strip("\n")
        
        if index < len(output_ids):
            content = self.tokenizer.decode(
                output_ids[index:], 
                skip_special_tokens=True
            ).strip("\n")
        else:
            # If no content after thinking, use full output
            content = self.tokenizer.decode(
                output_ids, 
                skip_special_tokens=True
            ).strip("\n")
        
        return thinking_content, content
    
    def _clean_response(self, response: str, pet: DigiPal) -> str:
        """
        Clean and validate the generated response.
        
        Args:
            response: Raw generated response
            pet: DigiPal instance for context
            
        Returns:
            Cleaned response string
        """
        # Remove any unwanted prefixes or suffixes
        response = response.strip()
        
        # Remove common AI assistant prefixes (more precise matching)
        prefixes_to_remove = [
            "As a DigiPal, ", "As your DigiPal, ", "DigiPal: ", f"{pet.name}: ",
            "Response: "
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
                break  # Only remove one prefix
        
        # Limit response length based on life stage
        max_lengths = {
            LifeStage.EGG: 20,
            LifeStage.BABY: 50,
            LifeStage.CHILD: 100,
            LifeStage.TEEN: 150,
            LifeStage.YOUNG_ADULT: 200,
            LifeStage.ADULT: 200,
            LifeStage.ELDERLY: 180
        }
        
        max_length = max_lengths.get(pet.life_stage, 100)
        if len(response) > max_length:
            # Find last complete sentence within limit
            sentences = response.split('.')
            truncated = ""
            for sentence in sentences:
                potential = truncated + sentence.strip()
                if len(potential) <= max_length - 1:  # Leave room for period
                    truncated = potential + "."
                else:
                    break
            
            if truncated and len(truncated) > 10:  # Ensure we have meaningful content
                response = truncated.strip()
            else:
                # If no complete sentence fits, truncate at word boundary
                words = response.split()
                truncated_words = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 <= max_length - 3:  # Leave room for "..."
                        truncated_words.append(word)
                        current_length += len(word) + 1
                    else:
                        break
                
                if truncated_words:
                    response = " ".join(truncated_words) + "..."
                else:
                    response = response[:max_length-3] + "..."
        
        # Ensure response is not empty
        if not response:
            response = self._fallback_response("", pet)
        
        return response
    
    def _fallback_response(self, user_input: str, pet: DigiPal) -> str:
        """
        Generate fallback response when model is unavailable.
        
        Args:
            user_input: User's input text
            pet: DigiPal instance
            
        Returns:
            Fallback response string
        """
        fallback_responses = {
            LifeStage.EGG: "*The egg remains silent*",
            LifeStage.BABY: "*baby sounds*",
            LifeStage.CHILD: "I'm still learning!",
            LifeStage.TEEN: "Hmm, let me think about that...",
            LifeStage.YOUNG_ADULT: "That's interesting to consider.",
            LifeStage.ADULT: "I understand what you're saying.",
            LifeStage.ELDERLY: "Ah, yes... I see..."
        }
        
        return fallback_responses.get(pet.life_stage, "I'm listening...")
    
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'quantization': self.quantization,
            'device': str(self.device),
            'loaded': self.is_loaded(),
            'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        }