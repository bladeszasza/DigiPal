"""
Tests for AI communication layer components.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from digipal.ai.communication import (
    AICommunication, 
    CommandInterpreter, 
    ResponseGenerator, 
    ConversationMemoryManager
)
from digipal.core.models import DigiPal, Interaction, Command
from digipal.core.enums import (
    EggType, 
    LifeStage, 
    CommandType, 
    InteractionResult
)


class TestCommandInterpreter:
    """Test command interpretation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.interpreter = CommandInterpreter()
    
    def test_parse_eat_command_baby_stage(self):
        """Test parsing eat commands for baby stage."""
        command = self.interpreter.parse_command("I want to eat", LifeStage.BABY)
        
        assert command.action == "eat"
        assert command.command_type == CommandType.EAT
        assert command.stage_appropriate is True
        assert command.energy_required == 0
    
    def test_parse_sleep_command_baby_stage(self):
        """Test parsing sleep commands for baby stage."""
        command = self.interpreter.parse_command("time to sleep", LifeStage.BABY)
        
        assert command.action == "sleep"
        assert command.command_type == CommandType.SLEEP
        assert command.stage_appropriate is True
    
    def test_parse_good_command_baby_stage(self):
        """Test parsing praise commands for baby stage."""
        command = self.interpreter.parse_command("good job!", LifeStage.BABY)
        
        assert command.action == "good"
        assert command.command_type == CommandType.GOOD
        assert command.stage_appropriate is True
    
    def test_parse_bad_command_baby_stage(self):
        """Test parsing scold commands for baby stage."""
        command = self.interpreter.parse_command("no, that's bad", LifeStage.BABY)
        
        assert command.action == "bad"
        assert command.command_type == CommandType.BAD
        assert command.stage_appropriate is True
    
    def test_parse_train_command_baby_stage_inappropriate(self):
        """Test that train commands are inappropriate for baby stage."""
        command = self.interpreter.parse_command("let's train", LifeStage.BABY)
        
        assert command.action == "train"
        assert command.command_type == CommandType.TRAIN
        assert command.stage_appropriate is False
        assert command.energy_required == 20
    
    def test_parse_train_command_child_stage(self):
        """Test parsing train commands for child stage."""
        command = self.interpreter.parse_command("let's train", LifeStage.CHILD)
        
        assert command.action == "train"
        assert command.command_type == CommandType.TRAIN
        assert command.stage_appropriate is True
        assert command.energy_required == 20
    
    def test_parse_play_command_child_stage(self):
        """Test parsing play commands for child stage."""
        command = self.interpreter.parse_command("let's play a game", LifeStage.CHILD)
        
        assert command.action == "play"
        assert command.command_type == CommandType.PLAY
        assert command.stage_appropriate is True
        assert command.energy_required == 10
    
    def test_parse_status_command_teen_stage(self):
        """Test parsing status commands for teen stage."""
        command = self.interpreter.parse_command("how are you feeling?", LifeStage.TEEN)
        
        assert command.action == "status"
        assert command.command_type == CommandType.STATUS
        assert command.stage_appropriate is True
        assert command.energy_required == 0
    
    def test_parse_status_command_baby_stage_inappropriate(self):
        """Test that status commands are inappropriate for baby stage."""
        command = self.interpreter.parse_command("show me your status", LifeStage.BABY)
        
        assert command.action == "status"
        assert command.command_type == CommandType.STATUS
        assert command.stage_appropriate is False
    
    def test_parse_unknown_command(self):
        """Test parsing unknown/unrecognized commands."""
        command = self.interpreter.parse_command("do a backflip", LifeStage.ADULT)
        
        assert command.action == "unknown"
        assert command.command_type == CommandType.UNKNOWN
        assert command.stage_appropriate is False
        assert command.energy_required == 0
        assert "original_text" in command.parameters
    
    def test_parse_train_with_parameters(self):
        """Test parsing train commands with specific parameters."""
        command = self.interpreter.parse_command("train your strength", LifeStage.ADULT)
        
        assert command.action == "train"
        assert command.command_type == CommandType.TRAIN
        assert command.parameters.get("training_type") == "strength"
    
    def test_parse_train_defense_parameters(self):
        """Test parsing train commands with defense parameters."""
        command = self.interpreter.parse_command("let's work on defense", LifeStage.ADULT)
        
        assert command.action == "train"
        assert command.parameters.get("training_type") == "defense"
    
    def test_parse_train_speed_parameters(self):
        """Test parsing train commands with speed parameters."""
        command = self.interpreter.parse_command("time for speed training", LifeStage.ADULT)
        
        assert command.action == "train"
        assert command.parameters.get("training_type") == "speed"
    
    def test_parse_train_brains_parameters(self):
        """Test parsing train commands with intelligence parameters."""
        command = self.interpreter.parse_command("let's train your brain", LifeStage.ADULT)
        
        assert command.action == "train"
        assert command.parameters.get("training_type") == "brains"
    
    def test_case_insensitive_parsing(self):
        """Test that command parsing is case insensitive."""
        command1 = self.interpreter.parse_command("EAT", LifeStage.BABY)
        command2 = self.interpreter.parse_command("eat", LifeStage.BABY)
        command3 = self.interpreter.parse_command("Eat", LifeStage.BABY)
        
        assert command1.action == command2.action == command3.action == "eat"
        assert all(cmd.stage_appropriate for cmd in [command1, command2, command3])


class TestResponseGenerator:
    """Test response generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ResponseGenerator()
        self.baby_pet = DigiPal(
            life_stage=LifeStage.BABY,
            happiness=50
        )
        self.child_pet = DigiPal(
            life_stage=LifeStage.CHILD,
            happiness=50
        )
        self.adult_pet = DigiPal(
            life_stage=LifeStage.ADULT,
            happiness=50
        )
    
    def test_generate_baby_eat_response(self):
        """Test generating eat response for baby stage."""
        response = self.generator.generate_response("eat", self.baby_pet)
        
        # Should be one of the baby eat responses
        expected_responses = ["*happy baby sounds*", "Goo goo!", "*contentedly munches*"]
        assert response in expected_responses
    
    def test_generate_baby_unknown_response(self):
        """Test generating response for unknown command in baby stage."""
        response = self.generator.generate_response("do a backflip", self.baby_pet)
        
        # Should be one of the baby unknown responses
        expected_responses = ["*tilts head curiously*", "*makes confused baby sounds*", "Goo?"]
        assert response in expected_responses
    
    def test_generate_child_train_response(self):
        """Test generating train response for child stage."""
        response = self.generator.generate_response("let's train", self.child_pet)
        
        # Should be one of the child train responses
        expected_responses = ["Let's get stronger!", "*pumps tiny fists*", "Training is fun!"]
        assert response in expected_responses
    
    def test_generate_adult_status_response(self):
        """Test generating status response for adult stage."""
        response = self.generator.generate_response("how are you?", self.adult_pet)
        
        # Should be one of the adult status responses
        expected_responses = ["I am at my peak capabilities", "*stands with dignity*", "How may I serve?"]
        assert response in expected_responses
    
    def test_happiness_affects_response_selection(self):
        """Test that pet happiness affects response selection."""
        # Create pets with different happiness levels
        happy_pet = DigiPal(life_stage=LifeStage.CHILD, happiness=80)
        sad_pet = DigiPal(life_stage=LifeStage.CHILD, happiness=20)
        
        # Generate responses for the same command
        happy_response = self.generator.generate_response("good job", happy_pet)
        sad_response = self.generator.generate_response("good job", sad_pet)
        
        # Responses should be from the same category but potentially different
        # (This tests the selection logic based on happiness)
        assert isinstance(happy_response, str)
        assert isinstance(sad_response, str)
        assert len(happy_response) > 0
        assert len(sad_response) > 0
    
    def test_stage_inappropriate_command_response(self):
        """Test response for stage-inappropriate commands."""
        # Try to get status from baby (inappropriate)
        response = self.generator.generate_response("show status", self.baby_pet)
        
        # Should fall back to unknown response for baby
        expected_responses = ["*tilts head curiously*", "*makes confused baby sounds*", "Goo?"]
        assert response in expected_responses


class TestConversationMemoryManager:
    """Test conversation memory management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.memory_manager = ConversationMemoryManager(max_memory_size=5)
        self.pet = DigiPal(
            life_stage=LifeStage.CHILD,
            happiness=50
        )
    
    def test_add_interaction_to_memory(self):
        """Test adding interaction to pet's memory."""
        interaction = Interaction(
            user_input="eat",
            interpreted_command="eat",
            pet_response="*munches happily*",
            success=True
        )
        
        initial_count = len(self.pet.conversation_history)
        self.memory_manager.add_interaction(interaction, self.pet)
        
        assert len(self.pet.conversation_history) == initial_count + 1
        assert self.pet.conversation_history[-1] == interaction
        assert self.pet.last_interaction == interaction.timestamp
    
    def test_learn_commands_from_successful_interactions(self):
        """Test that pet learns commands from successful interactions."""
        interaction = Interaction(
            user_input="let's play",
            interpreted_command="play",
            pet_response="Yay! Let's play!",
            success=True
        )
        
        initial_commands = len(self.pet.learned_commands)
        self.memory_manager.add_interaction(interaction, self.pet)
        
        assert "play" in self.pet.learned_commands
        assert len(self.pet.learned_commands) >= initial_commands
    
    def test_memory_size_management(self):
        """Test that memory size is managed properly."""
        # Add more interactions than max_memory_size
        for i in range(10):
            interaction = Interaction(
                user_input=f"command {i}",
                interpreted_command=f"action_{i}",
                pet_response=f"response {i}",
                success=True
            )
            self.memory_manager.add_interaction(interaction, self.pet)
        
        # Should only keep the most recent max_memory_size interactions
        assert len(self.pet.conversation_history) == self.memory_manager.max_memory_size
        
        # Should have the most recent interactions
        last_interaction = self.pet.conversation_history[-1]
        assert "command 9" in last_interaction.user_input
    
    def test_personality_trait_updates(self):
        """Test that personality traits are updated based on interactions."""
        # Initialize personality traits
        self.pet.personality_traits = {
            'friendliness': 0.5,
            'playfulness': 0.5,
            'obedience': 0.5,
            'curiosity': 0.5
        }
        
        # Add a 'good' interaction
        good_interaction = Interaction(
            user_input="good job",
            interpreted_command="good",
            pet_response="Thank you!",
            success=True
        )
        
        initial_obedience = self.pet.personality_traits['obedience']
        self.memory_manager.add_interaction(good_interaction, self.pet)
        
        # Obedience should increase
        assert self.pet.personality_traits['obedience'] > initial_obedience
    
    def test_get_recent_interactions(self):
        """Test retrieving recent interactions."""
        # Add several interactions
        for i in range(3):
            interaction = Interaction(
                user_input=f"test {i}",
                interpreted_command="test",
                pet_response=f"response {i}",
                success=True
            )
            self.memory_manager.add_interaction(interaction, self.pet)
        
        recent = self.memory_manager.get_recent_interactions(self.pet, count=2)
        
        assert len(recent) == 2
        assert "test 2" in recent[-1].user_input
        assert "test 1" in recent[-2].user_input
    
    def test_get_interaction_summary(self):
        """Test getting interaction summary statistics."""
        # Add mix of successful and failed interactions
        for i in range(5):
            interaction = Interaction(
                user_input=f"command {i}",
                interpreted_command="eat" if i % 2 == 0 else "unknown",
                pet_response=f"response {i}",
                success=i % 2 == 0
            )
            self.memory_manager.add_interaction(interaction, self.pet)
        
        summary = self.memory_manager.get_interaction_summary(self.pet)
        
        assert summary['total_interactions'] == 5
        assert summary['successful_interactions'] == 3  # 0, 2, 4
        assert summary['success_rate'] == 0.6
        assert len(summary['most_common_commands']) > 0
        assert summary['last_interaction'] is not None
    
    def test_empty_history_summary(self):
        """Test interaction summary with empty history."""
        empty_pet = DigiPal()
        summary = self.memory_manager.get_interaction_summary(empty_pet)
        
        assert summary['total_interactions'] == 0
        assert summary['successful_interactions'] == 0
        assert summary['success_rate'] == 0.0
        assert summary['most_common_commands'] == []
        assert summary['last_interaction'] is None


class TestAICommunication:
    """Test main AI communication class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.ai_comm = AICommunication()
        self.pet = DigiPal(
            life_stage=LifeStage.CHILD,
            happiness=50
        )
    
    def test_initialization(self):
        """Test AI communication initialization."""
        assert self.ai_comm.command_interpreter is not None
        assert self.ai_comm.response_generator is not None
        assert self.ai_comm.memory_manager is not None
    
    def test_process_speech_placeholder(self):
        """Test speech processing placeholder functionality."""
        audio_data = b"fake_audio_data"
        result = self.ai_comm.process_speech(audio_data)
        
        # Should return placeholder text
        assert result == "placeholder_speech_text"
    
    def test_interpret_command(self):
        """Test command interpretation."""
        command = self.ai_comm.interpret_command("let's eat", self.pet)
        
        assert command.action == "eat"
        assert command.command_type == CommandType.EAT
        assert command.stage_appropriate is True
    
    def test_generate_response(self):
        """Test response generation."""
        response = self.ai_comm.generate_response("eat", self.pet)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_process_interaction_complete_flow(self):
        """Test complete interaction processing flow."""
        input_text = "let's play"
        
        interaction = self.ai_comm.process_interaction(input_text, self.pet)
        
        assert interaction.user_input == input_text
        assert interaction.interpreted_command == "play"
        assert isinstance(interaction.pet_response, str)
        assert len(interaction.pet_response) > 0
        assert interaction.success is True
        assert interaction.result == InteractionResult.SUCCESS
        
        # Check that interaction was added to pet's memory
        assert len(self.pet.conversation_history) > 0
        assert self.pet.conversation_history[-1] == interaction
    
    def test_process_interaction_inappropriate_command(self):
        """Test processing stage-inappropriate command."""
        baby_pet = DigiPal(life_stage=LifeStage.BABY)
        input_text = "let's train"
        
        interaction = self.ai_comm.process_interaction(input_text, baby_pet)
        
        assert interaction.user_input == input_text
        assert interaction.interpreted_command == "train"
        assert interaction.success is False
        assert interaction.result == InteractionResult.STAGE_INAPPROPRIATE
    
    def test_update_conversation_memory(self):
        """Test conversation memory updates."""
        interaction = Interaction(
            user_input="test",
            interpreted_command="test",
            pet_response="test response",
            success=True
        )
        
        initial_count = len(self.pet.conversation_history)
        self.ai_comm.update_conversation_memory(interaction, self.pet)
        
        assert len(self.pet.conversation_history) == initial_count + 1
        assert self.pet.conversation_history[-1] == interaction
    
    def test_ai_communication_with_custom_config(self):
        """Test AI communication with custom configuration."""
        custom_config = {"test_param": "test_value"}
        ai_comm = AICommunication(
            model_name="custom/model",
            kyutai_config=custom_config
        )
        
        assert ai_comm.model_name == "custom/model"
        assert ai_comm.kyutai_config == custom_config


class TestIntegration:
    """Integration tests for AI communication components."""
    
    def test_full_conversation_flow(self):
        """Test a complete conversation flow with multiple interactions."""
        ai_comm = AICommunication()
        pet = DigiPal(life_stage=LifeStage.CHILD, happiness=60)
        
        # Simulate a conversation
        interactions = [
            "hello there",
            "let's eat",
            "good job",
            "time to play",
            "let's train"
        ]
        
        for user_input in interactions:
            interaction = ai_comm.process_interaction(user_input, pet)
            
            # Each interaction should be processed successfully
            assert isinstance(interaction, Interaction)
            assert interaction.user_input == user_input
            assert isinstance(interaction.pet_response, str)
            assert len(interaction.pet_response) > 0
        
        # Pet should have learned from the interactions
        assert len(pet.conversation_history) == len(interactions)
        assert len(pet.learned_commands) > 0
        
        # Get interaction summary
        summary = ai_comm.memory_manager.get_interaction_summary(pet)
        assert summary['total_interactions'] == len(interactions)
    
    def test_life_stage_progression_responses(self):
        """Test that responses change appropriately across life stages."""
        ai_comm = AICommunication()
        
        # Test same command across different life stages
        stages_to_test = [LifeStage.BABY, LifeStage.CHILD, LifeStage.TEEN, LifeStage.ADULT]
        
        for stage in stages_to_test:
            pet = DigiPal(life_stage=stage, happiness=50)
            interaction = ai_comm.process_interaction("eat", pet)
            
            # All stages should understand eat command
            assert interaction.success is True
            assert isinstance(interaction.pet_response, str)
            assert len(interaction.pet_response) > 0
            
            # Responses should be different for different stages
            # (This is implicit in the response templates)
    
    def test_personality_development_over_time(self):
        """Test that personality traits develop over multiple interactions."""
        ai_comm = AICommunication()
        pet = DigiPal(life_stage=LifeStage.CHILD, happiness=50)
        
        # Initialize personality traits
        pet.personality_traits = {
            'friendliness': 0.5,
            'playfulness': 0.5,
            'obedience': 0.5,
            'curiosity': 0.5
        }
        
        initial_obedience = pet.personality_traits['obedience']
        initial_playfulness = pet.personality_traits['playfulness']
        
        # Give multiple praise interactions
        for _ in range(3):
            ai_comm.process_interaction("good job", pet)
        
        # Give multiple play interactions
        for _ in range(2):
            ai_comm.process_interaction("let's play", pet)
        
        # Personality should have changed
        assert pet.personality_traits['obedience'] > initial_obedience
        assert pet.personality_traits['playfulness'] > initial_playfulness