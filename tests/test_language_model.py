"""
Tests for Qwen3-0.6B language model integration.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from digipal.ai.language_model import LanguageModel
from digipal.ai.communication import AICommunication
from digipal.core.models import DigiPal, Interaction
from digipal.core.enums import EggType, LifeStage, CommandType, InteractionResult


class TestLanguageModel:
    """Test cases for LanguageModel class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model_name = "Qwen/Qwen3-0.6B"
        self.language_model = LanguageModel(self.model_name, quantization=False)
        
        # Create test DigiPal
        self.test_pet = DigiPal(
            id="test-pet-1",
            user_id="test-user",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.CHILD,
            hp=100,
            happiness=75,
            energy=80,
            discipline=60
        )
    
    def test_initialization(self):
        """Test LanguageModel initialization."""
        assert self.language_model.model_name == self.model_name
        assert self.language_model.quantization == False
        assert self.language_model.tokenizer is None
        assert self.language_model.model is None
        assert not self.language_model.is_loaded()
    
    def test_initialization_with_quantization(self):
        """Test LanguageModel initialization with quantization."""
        model = LanguageModel(quantization=True)
        assert model.quantization == True
    
    @patch('digipal.ai.language_model.AutoTokenizer')
    @patch('digipal.ai.language_model.AutoModelForCausalLM')
    def test_load_model_success(self, mock_model_class, mock_tokenizer_class):
        """Test successful model loading."""
        # Mock tokenizer and model
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Test model loading
        success = self.language_model.load_model()
        
        assert success == True
        assert self.language_model.tokenizer == mock_tokenizer
        assert self.language_model.model == mock_model
        assert self.language_model.is_loaded() == True
        
        # Verify correct method calls
        mock_tokenizer_class.from_pretrained.assert_called_once_with(self.model_name)
        mock_model_class.from_pretrained.assert_called_once()
    
    @patch('digipal.ai.language_model.AutoTokenizer')
    def test_load_model_failure(self, mock_tokenizer_class):
        """Test model loading failure."""
        # Mock tokenizer loading failure
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")
        
        # Test model loading
        success = self.language_model.load_model()
        
        assert success == False
        assert not self.language_model.is_loaded()
    
    @patch('digipal.ai.language_model.torch.cuda.is_available')
    @patch('digipal.ai.language_model.AutoTokenizer')
    @patch('digipal.ai.language_model.AutoModelForCausalLM')
    def test_load_model_with_quantization(self, mock_model_class, mock_tokenizer_class, mock_cuda):
        """Test model loading with quantization."""
        # Setup mocks
        mock_cuda.return_value = True
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Create model with quantization
        model = LanguageModel(quantization=True)
        success = model.load_model()
        
        assert success == True
        # Verify quantization config was used
        call_args = mock_model_class.from_pretrained.call_args
        assert 'quantization_config' in call_args[1]
    
    def test_create_prompt_baby_stage(self):
        """Test prompt creation for baby stage."""
        self.test_pet.life_stage = LifeStage.BABY
        self.test_pet.personality_traits = {
            'friendliness': 0.8,
            'playfulness': 0.6,
            'obedience': 0.7,
            'curiosity': 0.5
        }
        
        prompt = self.language_model._create_prompt("Hello baby!", self.test_pet)
        
        assert "baby DigiPal" in prompt
        assert "TestPal" in prompt
        assert "eat, sleep, good, bad" in prompt
        assert "Hello baby!" in prompt
        assert "very friendly" in prompt
    
    def test_create_prompt_with_conversation_history(self):
        """Test prompt creation with conversation history."""
        # Add some conversation history
        interaction1 = Interaction(
            timestamp=datetime.now() - timedelta(minutes=5),
            user_input="Good job!",
            pet_response="Thank you!",
            success=True
        )
        interaction2 = Interaction(
            timestamp=datetime.now() - timedelta(minutes=2),
            user_input="Let's play!",
            pet_response="Yay! Let's play!",
            success=True
        )
        self.test_pet.conversation_history = [interaction1, interaction2]
        
        prompt = self.language_model._create_prompt("How are you?", self.test_pet)
        
        assert "Good job!" in prompt
        assert "Let's play!" in prompt
        assert "Thank you!" in prompt
        assert "Yay! Let's play!" in prompt
    
    def test_get_personality_description(self):
        """Test personality description generation."""
        # Test high friendliness
        self.test_pet.personality_traits = {'friendliness': 0.8}
        desc = self.language_model._get_personality_description(self.test_pet)
        assert "very friendly" in desc
        
        # Test low friendliness
        self.test_pet.personality_traits = {'friendliness': 0.2}
        desc = self.language_model._get_personality_description(self.test_pet)
        assert "somewhat shy" in desc
        
        # Test multiple traits
        self.test_pet.personality_traits = {
            'friendliness': 0.8,
            'playfulness': 0.9,
            'obedience': 0.1,
            'curiosity': 0.8
        }
        desc = self.language_model._get_personality_description(self.test_pet)
        assert "very friendly" in desc
        assert "very playful" in desc
        assert "a bit rebellious" in desc
        assert "very curious" in desc
    
    def test_parse_response_with_thinking(self):
        """Test response parsing with thinking content."""
        # Mock token IDs with thinking end token (151668)
        output_ids = [100, 200, 300, 151668, 400, 500, 600]
        
        # Mock tokenizer decode
        self.language_model.tokenizer = Mock()
        self.language_model.tokenizer.decode.side_effect = [
            "This is thinking content",  # thinking part
            "This is the actual response"  # response part
        ]
        
        thinking, content = self.language_model._parse_response(output_ids)
        
        assert thinking == "This is thinking content"
        assert content == "This is the actual response"
    
    def test_parse_response_without_thinking(self):
        """Test response parsing without thinking content."""
        # Mock token IDs without thinking end token
        output_ids = [100, 200, 300, 400, 500]
        
        # Mock tokenizer decode
        self.language_model.tokenizer = Mock()
        self.language_model.tokenizer.decode.return_value = "Direct response"
        
        thinking, content = self.language_model._parse_response(output_ids)
        
        assert thinking == ""
        assert content == "Direct response"
    
    def test_clean_response_length_limits(self):
        """Test response cleaning with length limits."""
        # Test baby stage length limit
        self.test_pet.life_stage = LifeStage.BABY
        long_response = "This is a very long response that exceeds the baby stage limit and should be truncated appropriately."
        
        cleaned = self.language_model._clean_response(long_response, self.test_pet)
        assert len(cleaned) <= 50
    
    def test_clean_response_removes_prefixes(self):
        """Test response cleaning removes unwanted prefixes."""
        response = "As a DigiPal, I am happy to see you!"
        cleaned = self.language_model._clean_response(response, self.test_pet)
        assert not cleaned.startswith("As a DigiPal,")
        assert "I am happy to see you!" in cleaned
    
    def test_fallback_response(self):
        """Test fallback response generation."""
        for stage in LifeStage:
            self.test_pet.life_stage = stage
            response = self.language_model._fallback_response("test input", self.test_pet)
            assert isinstance(response, str)
            assert len(response) > 0
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        info = self.language_model.get_model_info()
        
        assert info['model_name'] == self.model_name
        assert info['quantization'] == False
        assert info['loaded'] == False
        assert 'device' in info
        assert 'memory_usage' in info
    
    @patch('digipal.ai.language_model.torch.no_grad')
    def test_generate_response_without_model(self, mock_no_grad):
        """Test response generation when model is not loaded."""
        response = self.language_model.generate_response("Hello", self.test_pet)
        
        # Should return fallback response
        assert isinstance(response, str)
        assert len(response) > 0
        # Should not call torch.no_grad since model is not loaded
        mock_no_grad.assert_not_called()
    
    @patch('digipal.ai.language_model.torch.no_grad')
    def test_generate_response_with_model(self, mock_no_grad):
        """Test response generation with loaded model."""
        # Mock loaded model and tokenizer
        mock_tokenizer = Mock()
        mock_model = Mock()
        mock_model.device = torch.device('cpu')
        
        # Mock tokenizer methods
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        # Fix: tokenizer should return tensor directly, not dict
        mock_model_inputs = Mock()
        mock_model_inputs.input_ids = torch.tensor([[1, 2, 3]])
        mock_model_inputs.to.return_value = mock_model_inputs
        mock_tokenizer.return_value = mock_model_inputs
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.decode.return_value = "Generated response"
        
        # Mock model generate
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        
        # Set up language model
        self.language_model.model = mock_model
        self.language_model.tokenizer = mock_tokenizer
        
        # Mock context manager
        mock_context = Mock()
        mock_no_grad.return_value.__enter__ = Mock(return_value=mock_context)
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)
        
        response = self.language_model.generate_response("Hello", self.test_pet)
        
        assert isinstance(response, str)
        mock_no_grad.assert_called_once()


class TestAICommunicationIntegration:
    """Test cases for AICommunication integration with LanguageModel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ai_comm = AICommunication(quantization=False)
        
        # Create test DigiPal
        self.test_pet = DigiPal(
            id="test-pet-1",
            user_id="test-user",
            name="TestPal",
            egg_type=EggType.BLUE,
            life_stage=LifeStage.TEEN,
            hp=120,
            happiness=65,
            energy=90,
            discipline=70
        )
    
    def test_initialization(self):
        """Test AICommunication initialization with language model."""
        assert self.ai_comm.model_name == "Qwen/Qwen3-0.6B"
        assert self.ai_comm.quantization == False
        assert isinstance(self.ai_comm.language_model, LanguageModel)
        assert not self.ai_comm.is_model_loaded()
    
    @patch.object(LanguageModel, 'load_model')
    def test_load_model_success(self, mock_load):
        """Test successful model loading through AICommunication."""
        mock_load.return_value = True
        
        success = self.ai_comm.load_model()
        
        assert success == True
        assert self.ai_comm._model_loaded == True
        mock_load.assert_called_once()
    
    @patch.object(LanguageModel, 'load_model')
    def test_load_model_failure(self, mock_load):
        """Test model loading failure through AICommunication."""
        mock_load.return_value = False
        
        success = self.ai_comm.load_model()
        
        assert success == False
        assert self.ai_comm._model_loaded == False
    
    @patch.object(LanguageModel, 'is_loaded')
    @patch.object(LanguageModel, 'generate_response')
    def test_generate_response_with_model(self, mock_generate, mock_is_loaded):
        """Test response generation using language model."""
        mock_is_loaded.return_value = True
        mock_generate.return_value = "AI generated response"
        self.ai_comm._model_loaded = True
        
        response = self.ai_comm.generate_response("Hello!", self.test_pet)
        
        assert response == "AI generated response"
        mock_generate.assert_called_once_with("Hello!", self.test_pet)
    
    @patch.object(LanguageModel, 'is_loaded')
    def test_generate_response_fallback(self, mock_is_loaded):
        """Test response generation fallback to template responses."""
        mock_is_loaded.return_value = False
        self.ai_comm._model_loaded = False
        
        response = self.ai_comm.generate_response("Hello!", self.test_pet)
        
        # Should use fallback response generator
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_process_interaction_complete_flow(self):
        """Test complete interaction processing flow."""
        # Mock language model to avoid actual loading
        with patch.object(self.ai_comm.language_model, 'is_loaded', return_value=False):
            interaction = self.ai_comm.process_interaction("Let's train!", self.test_pet)
        
        assert isinstance(interaction, Interaction)
        assert interaction.user_input == "Let's train!"
        assert interaction.interpreted_command == "train"
        assert interaction.success == True  # Train is available for teen stage
        assert interaction.result == InteractionResult.SUCCESS
        assert len(interaction.pet_response) > 0
    
    def test_get_model_info(self):
        """Test model information retrieval."""
        info = self.ai_comm.get_model_info()
        
        assert info['model_name'] == "Qwen/Qwen3-0.6B"
        assert info['quantization'] == False
        assert info['loaded'] == False
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.empty_cache')
    def test_unload_model(self, mock_empty_cache, mock_cuda_available):
        """Test model unloading."""
        mock_cuda_available.return_value = True
        
        # Set up loaded state
        self.ai_comm._model_loaded = True
        self.ai_comm.language_model.model = Mock()
        self.ai_comm.language_model.tokenizer = Mock()
        
        self.ai_comm.unload_model()
        
        assert self.ai_comm._model_loaded == False
        assert self.ai_comm.language_model.model is None
        assert self.ai_comm.language_model.tokenizer is None
        mock_empty_cache.assert_called_once()


class TestContextAwareGeneration:
    """Test cases for context-aware response generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.language_model = LanguageModel(quantization=False)
        
        # Create DigiPals at different life stages
        self.baby_pet = DigiPal(
            name="BabyPal",
            life_stage=LifeStage.BABY,
            happiness=80,
            energy=60
        )
        
        self.adult_pet = DigiPal(
            name="AdultPal",
            life_stage=LifeStage.ADULT,
            happiness=70,
            energy=85,
            personality_traits={
                'friendliness': 0.9,
                'obedience': 0.8,
                'playfulness': 0.3,
                'curiosity': 0.6
            }
        )
    
    def test_prompt_templates_different_stages(self):
        """Test that different life stages use appropriate prompt templates."""
        baby_prompt = self.language_model._create_prompt("Hello", self.baby_pet)
        adult_prompt = self.language_model._create_prompt("Hello", self.adult_pet)
        
        # Baby prompt should mention basic commands
        assert "eat, sleep, good, bad" in baby_prompt
        assert "baby" in baby_prompt.lower()
        
        # Adult prompt should be more sophisticated
        assert "wise" in adult_prompt.lower() or "mature" in adult_prompt.lower()
        assert "adult" in adult_prompt.lower()
        
        # Prompts should be different
        assert baby_prompt != adult_prompt
    
    def test_personality_integration(self):
        """Test that personality traits are integrated into prompts."""
        prompt = self.language_model._create_prompt("How are you?", self.adult_pet)
        
        # Should include personality description
        assert "very friendly" in prompt
        assert "well-behaved" in prompt
    
    def test_pet_stats_integration(self):
        """Test that pet statistics are included in prompts."""
        prompt = self.language_model._create_prompt("Status check", self.adult_pet)
        
        # Should include current stats
        assert str(self.adult_pet.hp) in prompt
        assert str(self.adult_pet.happiness) in prompt
        assert str(self.adult_pet.energy) in prompt


if __name__ == "__main__":
    pytest.main([__file__])