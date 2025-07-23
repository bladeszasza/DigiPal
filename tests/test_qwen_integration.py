"""
Integration tests for Qwen3-0.6B model integration.

These tests verify the complete integration of Qwen/Qwen3-0.6B model
with the DigiPal system, including quantization, context handling,
and response generation.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from digipal.ai.language_model import LanguageModel
from digipal.ai.communication import AICommunication
from digipal.core.models import DigiPal, Interaction
from digipal.core.enums import EggType, LifeStage


class TestQwenIntegration:
    """Test cases for Qwen3-0.6B integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_name = "Qwen/Qwen3-0.6B"
        
        # Create test DigiPal with rich context
        self.test_pet = DigiPal(
            id="qwen-test-pet",
            user_id="qwen-test-user",
            name="QwenPal",
            egg_type=EggType.BLUE,
            life_stage=LifeStage.YOUNG_ADULT,
            hp=150,
            happiness=85,
            energy=90,
            discipline=75,
            personality_traits={
                'friendliness': 0.9,
                'playfulness': 0.7,
                'obedience': 0.8,
                'curiosity': 0.6
            }
        )
        
        # Add conversation history
        self.test_pet.conversation_history = [
            Interaction(
                timestamp=datetime.now() - timedelta(minutes=10),
                user_input="Let's train!",
                pet_response="Let's push our limits!",
                success=True
            ),
            Interaction(
                timestamp=datetime.now() - timedelta(minutes=5),
                user_input="Good job!",
                pet_response="I appreciate the encouragement!",
                success=True
            )
        ]
    
    def test_qwen_model_specification_compliance(self):
        """Test that the integration follows the exact Qwen3-0.6B specification."""
        language_model = LanguageModel(self.model_name, quantization=False)
        
        # Verify model name matches specification
        assert language_model.model_name == "Qwen/Qwen3-0.6B"
        
        # Verify prompt templates include thinking mode support
        prompt = language_model._create_prompt("Test input", self.test_pet)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Verify response parsing handles thinking content
        output_ids = [100, 200, 151668, 300, 400]  # 151668 is </think> token
        language_model.tokenizer = Mock()
        language_model.tokenizer.decode.side_effect = [
            "This is thinking content",
            "This is the response"
        ]
        
        thinking, content = language_model._parse_response(output_ids)
        assert thinking == "This is thinking content"
        assert content == "This is the response"
    
    @patch('digipal.ai.language_model.BitsAndBytesConfig')
    @patch('digipal.ai.language_model.AutoTokenizer')
    @patch('digipal.ai.language_model.AutoModelForCausalLM')
    @patch('digipal.ai.language_model.torch.cuda.is_available')
    def test_quantization_configuration(self, mock_cuda, mock_model_class, mock_tokenizer_class, mock_quantization_config):
        """Test quantization configuration for memory optimization."""
        mock_cuda.return_value = True
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test with quantization enabled
        language_model = LanguageModel(self.model_name, quantization=True)
        success = language_model.load_model()
        
        assert success == True
        
        # Verify quantization config was created
        mock_quantization_config.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Verify model was loaded with quantization config
        call_args = mock_model_class.from_pretrained.call_args
        assert 'quantization_config' in call_args[1]
    
    @patch('digipal.ai.language_model.AutoTokenizer')
    @patch('digipal.ai.language_model.AutoModelForCausalLM')
    def test_context_aware_prompt_generation(self, mock_model_class, mock_tokenizer_class):
        """Test context-aware prompt generation with pet state and history."""
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        language_model = LanguageModel(self.model_name, quantization=False)
        language_model.load_model()
        
        prompt = language_model._create_prompt("How are you feeling?", self.test_pet)
        
        # Verify pet context is included
        assert "QwenPal" in prompt
        assert "young adult" in prompt.lower()
        assert "HP=150" in prompt
        assert "Happiness=85" in prompt
        
        # Verify personality is included
        assert "very friendly" in prompt
        
        # Verify conversation history is included
        assert "Let's train!" in prompt
        assert "Good job!" in prompt
        assert "Let's push our limits!" in prompt
    
    @patch('digipal.ai.language_model.torch.no_grad')
    def test_response_generation_pipeline(self, mock_no_grad):
        """Test complete response generation pipeline."""
        # Mock the entire generation pipeline
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.device = torch.device('cpu')
        
        # Mock chat template application
        mock_tokenizer.apply_chat_template.return_value = "formatted_prompt"
        
        # Mock tokenization - return dict-like object that can be unpacked
        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_inputs_obj = Mock()
        mock_inputs_obj.input_ids = torch.tensor([[1, 2, 3]])
        mock_inputs_obj.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs_obj
        mock_tokenizer.eos_token_id = 2
        
        # Mock generation with thinking content
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 100, 200, 151668, 300, 400]])
        
        # Mock decoding
        mock_tokenizer.decode.side_effect = [
            "I'm thinking about this...",  # thinking content
            "I'm feeling great today!"     # actual response
        ]
        
        # Set up language model
        language_model = LanguageModel(self.model_name, quantization=False)
        language_model.model = mock_model
        language_model.tokenizer = mock_tokenizer
        
        # Mock context manager
        mock_context = Mock()
        mock_no_grad.return_value.__enter__ = Mock(return_value=mock_context)
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)
        
        response = language_model.generate_response("How are you?", self.test_pet)
        
        # Verify response generation
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Verify chat template was used with thinking enabled
        mock_tokenizer.apply_chat_template.assert_called_once()
        call_args = mock_tokenizer.apply_chat_template.call_args
        assert call_args[1]['enable_thinking'] == True
        
        # Verify model generation was called
        mock_model.generate.assert_called_once()
        
        # Verify torch.no_grad was used
        mock_no_grad.assert_called_once()
    
    def test_life_stage_appropriate_responses(self):
        """Test that responses are appropriate for different life stages."""
        language_model = LanguageModel(self.model_name, quantization=False)
        
        life_stages = [
            (LifeStage.BABY, "baby", "eat, sleep, good, bad"),
            (LifeStage.CHILD, "child", "play, train"),
            (LifeStage.TEEN, "teenage", "conversations"),
            (LifeStage.ADULT, "adult", "wise, mature"),
            (LifeStage.ELDERLY, "elderly", "wisdom")
        ]
        
        for stage, stage_desc, expected_content in life_stages:
            self.test_pet.life_stage = stage
            prompt = language_model._create_prompt("Hello", self.test_pet)
            
            assert stage_desc in prompt.lower()
            assert expected_content in prompt.lower()
    
    def test_response_length_limits_by_stage(self):
        """Test that response length limits are enforced by life stage."""
        language_model = LanguageModel(self.model_name, quantization=False)
        
        # Test different stages with long responses
        long_response = "This is a very long response that should be truncated based on the life stage of the DigiPal. " * 10
        
        stage_limits = [
            (LifeStage.BABY, 50),
            (LifeStage.CHILD, 100),
            (LifeStage.TEEN, 150),
            (LifeStage.ADULT, 200),
            (LifeStage.ELDERLY, 180)
        ]
        
        for stage, expected_limit in stage_limits:
            self.test_pet.life_stage = stage
            cleaned = language_model._clean_response(long_response, self.test_pet)
            assert len(cleaned) <= expected_limit
    
    def test_personality_trait_integration(self):
        """Test that personality traits are properly integrated into prompts."""
        language_model = LanguageModel(self.model_name, quantization=False)
        
        # Test different personality combinations
        personality_tests = [
            ({'friendliness': 0.9, 'playfulness': 0.2}, "very friendly"),
            ({'friendliness': 0.1, 'playfulness': 0.9}, "somewhat shy"),
            ({'obedience': 0.9}, "well-behaved"),
            ({'obedience': 0.1}, "a bit rebellious"),
            ({'curiosity': 0.9}, "very curious")
        ]
        
        for traits, expected_desc in personality_tests:
            self.test_pet.personality_traits = traits
            desc = language_model._get_personality_description(self.test_pet)
            assert expected_desc in desc
    
    def test_conversation_memory_integration(self):
        """Test that conversation memory is properly integrated."""
        language_model = LanguageModel(self.model_name, quantization=False)
        
        # Add more conversation history
        additional_interactions = [
            Interaction(
                timestamp=datetime.now() - timedelta(minutes=3),
                user_input="Let's play!",
                pet_response="A good balance of work and play!",
                success=True
            ),
            Interaction(
                timestamp=datetime.now() - timedelta(minutes=1),
                user_input="You're amazing!",
                pet_response="Your support means everything",
                success=True
            )
        ]
        self.test_pet.conversation_history.extend(additional_interactions)
        
        prompt = language_model._create_prompt("What's next?", self.test_pet)
        
        # Verify recent interactions are included (last 3)
        assert "Let's play!" in prompt
        assert "You're amazing!" in prompt
        assert "A good balance of work and play!" in prompt
        
        # Verify older interactions are not included (only last 3)
        recent_count = prompt.count("User:") + prompt.count("DigiPal:")
        assert recent_count <= 6  # 3 interactions * 2 (user + pet response)


class TestAICommunicationQwenIntegration:
    """Test AICommunication integration with Qwen3-0.6B."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ai_comm = AICommunication(
            model_name="Qwen/Qwen3-0.6B",
            quantization=True
        )
        
        self.test_pet = DigiPal(
            name="CommPal",
            life_stage=LifeStage.ADULT,
            happiness=80,
            energy=75
        )
    
    @patch.object(LanguageModel, 'load_model')
    @patch.object(LanguageModel, 'is_loaded')
    @patch.object(LanguageModel, 'generate_response')
    def test_model_integration_flow(self, mock_generate, mock_is_loaded, mock_load):
        """Test complete model integration flow."""
        # Setup mocks
        mock_load.return_value = True
        mock_is_loaded.return_value = True
        mock_generate.return_value = "I understand your request!"
        
        # Load model
        success = self.ai_comm.load_model()
        assert success == True
        
        # Generate response
        response = self.ai_comm.generate_response("Hello", self.test_pet)
        assert response == "I understand your request!"
        
        # Verify model was used
        mock_generate.assert_called_once_with("Hello", self.test_pet)
    
    def test_fallback_when_model_unavailable(self):
        """Test fallback behavior when Qwen model is unavailable."""
        # Don't load the model
        response = self.ai_comm.generate_response("Hello", self.test_pet)
        
        # Should get fallback response
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_model_info_reporting(self):
        """Test model information reporting."""
        info = self.ai_comm.get_model_info()
        
        assert info['model_name'] == "Qwen/Qwen3-0.6B"
        assert info['quantization'] == True
        assert 'loaded' in info
        assert 'device' in info


if __name__ == "__main__":
    pytest.main([__file__])