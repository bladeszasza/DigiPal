"""
Tests for DigiPal image generation system.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import torch

from digipal.ai.image_generator import ImageGenerator
from digipal.core.models import DigiPal
from digipal.core.enums import LifeStage, EggType


class TestImageGenerator:
    """Test cases for ImageGenerator class."""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_dir = tempfile.mkdtemp()
        cache_dir = Path(temp_dir) / "cache"
        fallback_dir = Path(temp_dir) / "fallbacks"
        
        yield cache_dir, fallback_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def image_generator(self, temp_dirs):
        """Create ImageGenerator instance for testing."""
        cache_dir, fallback_dir = temp_dirs
        return ImageGenerator(
            cache_dir=str(cache_dir),
            fallback_dir=str(fallback_dir)
        )
    
    @pytest.fixture
    def sample_digipal(self):
        """Create sample DigiPal for testing."""
        digipal = DigiPal(
            id="test-123",
            user_id="user-456",
            name="TestPal",
            egg_type=EggType.RED,
            life_stage=LifeStage.BABY,
            offense=60,
            defense=40,
            speed=70,
            brains=50,
            happiness=80
        )
        return digipal
    
    def test_initialization(self, temp_dirs):
        """Test ImageGenerator initialization."""
        cache_dir, fallback_dir = temp_dirs
        generator = ImageGenerator(
            cache_dir=str(cache_dir),
            fallback_dir=str(fallback_dir)
        )
        
        assert generator.cache_dir == cache_dir
        assert generator.fallback_dir == fallback_dir
        assert cache_dir.exists()
        assert fallback_dir.exists()
        assert not generator._model_loaded
        assert generator.pipe is None
    
    def test_prompt_generation_basic(self, image_generator, sample_digipal):
        """Test basic prompt generation."""
        prompt = image_generator.generate_prompt(sample_digipal)
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "baby" in prompt.lower()
        assert "fire" in prompt.lower()
        assert "red" in prompt.lower()
        assert "digital art" in prompt.lower()
    
    def test_prompt_generation_all_stages(self, image_generator):
        """Test prompt generation for all life stages."""
        for stage in LifeStage:
            digipal = DigiPal(
                life_stage=stage,
                egg_type=EggType.BLUE
            )
            prompt = image_generator.generate_prompt(digipal)
            
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            assert stage.value in prompt.lower()
    
    def test_prompt_generation_all_egg_types(self, image_generator):
        """Test prompt generation for all egg types."""
        for egg_type in EggType:
            digipal = DigiPal(
                life_stage=LifeStage.CHILD,
                egg_type=egg_type
            )
            prompt = image_generator.generate_prompt(digipal)
            
            assert isinstance(prompt, str)
            assert len(prompt) > 0
            # Check that egg type characteristics are included
            if egg_type == EggType.RED:
                assert "fire" in prompt.lower()
            elif egg_type == EggType.BLUE:
                assert "water" in prompt.lower()
            elif egg_type == EggType.GREEN:
                assert "earth" in prompt.lower()
    
    def test_prompt_generation_attribute_modifiers(self, image_generator):
        """Test that DigiPal attributes affect prompt generation."""
        # High offense DigiPal (not egg stage to see attribute effects)
        high_offense_pal = DigiPal(offense=80, defense=20, speed=30, brains=25, life_stage=LifeStage.ADULT)
        prompt_offense = image_generator.generate_prompt(high_offense_pal)
        assert "fierce" in prompt_offense.lower()
        
        # High defense DigiPal
        high_defense_pal = DigiPal(offense=20, defense=80, speed=30, brains=25, life_stage=LifeStage.ADULT)
        prompt_defense = image_generator.generate_prompt(high_defense_pal)
        assert "armored" in prompt_defense.lower()
        
        # High speed DigiPal
        high_speed_pal = DigiPal(offense=20, defense=30, speed=80, brains=25, life_stage=LifeStage.ADULT)
        prompt_speed = image_generator.generate_prompt(high_speed_pal)
        assert "sleek" in prompt_speed.lower()
        
        # High brains DigiPal
        high_brains_pal = DigiPal(offense=20, defense=30, speed=25, brains=80, life_stage=LifeStage.ADULT)
        prompt_brains = image_generator.generate_prompt(high_brains_pal)
        assert "intelligent" in prompt_brains.lower()
    
    def test_prompt_generation_happiness_modifiers(self, image_generator):
        """Test that happiness affects prompt generation."""
        # Happy DigiPal
        happy_pal = DigiPal(happiness=90)
        prompt_happy = image_generator.generate_prompt(happy_pal)
        assert "happy" in prompt_happy.lower() or "cheerful" in prompt_happy.lower()
        
        # Sad DigiPal
        sad_pal = DigiPal(happiness=20)
        prompt_sad = image_generator.generate_prompt(sad_pal)
        assert "sad" in prompt_sad.lower() or "tired" in prompt_sad.lower()
    
    def test_cache_key_generation(self, image_generator):
        """Test cache key generation."""
        prompt = "test prompt"
        params = {"height": 1024, "width": 1024}
        
        key1 = image_generator._get_cache_key(prompt, params)
        key2 = image_generator._get_cache_key(prompt, params)
        
        assert key1 == key2  # Same inputs should produce same key
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length
        
        # Different inputs should produce different keys
        key3 = image_generator._get_cache_key("different prompt", params)
        assert key1 != key3
    
    def test_placeholder_image_creation(self, image_generator):
        """Test placeholder image creation."""
        test_path = image_generator.fallback_dir / "test_placeholder.png"
        
        image_generator._create_placeholder_image(
            test_path, LifeStage.BABY, EggType.RED
        )
        
        assert test_path.exists()
        
        # Verify it's a valid image
        img = Image.open(test_path)
        assert img.size == (512, 512)
        assert img.mode == 'RGB'
    
    def test_fallback_image_initialization(self, image_generator):
        """Test that fallback images are properly initialized."""
        # Check that fallback mappings exist for all combinations
        for stage in LifeStage:
            for egg_type in EggType:
                key = f"{stage.value}_{egg_type.value}"
                assert key in image_generator.fallback_images
                
                fallback_path = Path(image_generator.fallback_images[key])
                assert fallback_path.exists()
    
    def test_get_fallback_image(self, image_generator, sample_digipal):
        """Test fallback image retrieval."""
        fallback_path = image_generator._get_fallback_image(sample_digipal)
        
        assert isinstance(fallback_path, str)
        assert Path(fallback_path).exists()
        assert "baby_red" in fallback_path.lower()
    
    def test_cache_operations(self, image_generator):
        """Test image caching operations."""
        # Create a test image
        test_image = Image.new('RGB', (100, 100), color='red')
        cache_key = "test_cache_key"
        
        # Test saving to cache
        cache_path = image_generator._save_to_cache(test_image, cache_key)
        assert cache_path.exists()
        assert cache_path.name == f"{cache_key}.png"
        
        # Test retrieving from cache
        retrieved_path = image_generator._get_cached_image_path(cache_key)
        assert retrieved_path == cache_path
        
        # Test non-existent cache
        non_existent = image_generator._get_cached_image_path("non_existent_key")
        assert non_existent is None
    
    def test_model_loading(self, image_generator):
        """Test model loading functionality."""
        # Mock the diffusers module and FluxPipeline class
        mock_diffusers = Mock()
        mock_flux_pipeline = Mock()
        mock_diffusers.FluxPipeline = mock_flux_pipeline
        
        with patch.dict('sys.modules', {'diffusers': mock_diffusers}):
            mock_pipe = Mock()
            mock_flux_pipeline.from_pretrained.return_value = mock_pipe
            
            # Test initial state
            assert not image_generator._model_loaded
            
            # Test model loading
            image_generator._load_model()
            
            assert image_generator._model_loaded
            assert image_generator.pipe == mock_pipe
            mock_flux_pipeline.from_pretrained.assert_called_once()
            mock_pipe.enable_model_cpu_offload.assert_called_once()
            
            # Test that subsequent calls don't reload
            mock_flux_pipeline.reset_mock()
            image_generator._load_model()
            mock_flux_pipeline.from_pretrained.assert_not_called()
    
    def test_model_loading_import_error(self, image_generator):
        """Test model loading with missing diffusers library."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'diffusers'")):
            with pytest.raises(ImportError):
                image_generator._load_model()
    
    def test_generate_image_success(self, image_generator, sample_digipal):
        """Test successful image generation."""
        # Mock the diffusers module and FluxPipeline class
        mock_diffusers = Mock()
        mock_flux_pipeline = Mock()
        mock_diffusers.FluxPipeline = mock_flux_pipeline
        
        with patch.dict('sys.modules', {'diffusers': mock_diffusers}):
                # Mock the pipeline
                mock_pipe = Mock()
                mock_image = Mock()
                mock_pipe.return_value.images = [mock_image]
                mock_flux_pipeline.from_pretrained.return_value = mock_pipe
                
                # Mock image saving
                with patch.object(image_generator, '_save_to_cache') as mock_save:
                    mock_save.return_value = Path("test_path.png")
                    
                    result_path = image_generator.generate_image(sample_digipal)
                    
                    assert result_path == "test_path.png"
                    assert sample_digipal.current_image_path == "test_path.png"
                    assert len(sample_digipal.image_generation_prompt) > 0
                    mock_save.assert_called_once()
    
    def test_generate_image_with_cache(self, image_generator, sample_digipal):
        """Test image generation with existing cache."""
        # Create a cached image
        test_image = Image.new('RGB', (100, 100), color='blue')
        prompt = image_generator.generate_prompt(sample_digipal)
        cache_key = image_generator._get_cache_key(prompt, image_generator.generation_params)
        cache_path = image_generator._save_to_cache(test_image, cache_key)
        
        # Generate image (should use cache)
        result_path = image_generator.generate_image(sample_digipal)
        
        assert result_path == str(cache_path)
    
    def test_generate_image_force_regenerate(self, image_generator, sample_digipal):
        """Test forced image regeneration."""
        # Create a cached image
        test_image = Image.new('RGB', (100, 100), color='blue')
        prompt = image_generator.generate_prompt(sample_digipal)
        cache_key = image_generator._get_cache_key(prompt, image_generator.generation_params)
        image_generator._save_to_cache(test_image, cache_key)
        
        # Mock the diffusers module and FluxPipeline class
        mock_diffusers = Mock()
        mock_flux = Mock()
        mock_diffusers.FluxPipeline = mock_flux
        
        with patch.dict('sys.modules', {'diffusers': mock_diffusers}):
                mock_pipe = Mock()
                mock_new_image = Mock()
                mock_pipe.return_value.images = [mock_new_image]
                mock_flux.from_pretrained.return_value = mock_pipe
                
                with patch.object(image_generator, '_save_to_cache') as mock_save:
                    mock_save.return_value = Path("new_test_path.png")
                    
                    result_path = image_generator.generate_image(sample_digipal, force_regenerate=True)
                    
                    assert result_path == "new_test_path.png"
                    mock_save.assert_called_once()
    
    def test_generate_image_failure_fallback(self, image_generator, sample_digipal):
        """Test image generation failure with fallback."""
        # Mock model loading to fail
        with patch.object(image_generator, '_load_model', side_effect=Exception("Model failed")):
            result_path = image_generator.generate_image(sample_digipal)
            
            # Should return fallback image
            assert "baby_red" in result_path.lower()
            assert Path(result_path).exists()
            assert sample_digipal.current_image_path == result_path
            assert "Fallback image" in sample_digipal.image_generation_prompt
    
    def test_update_image_for_evolution(self, image_generator, sample_digipal):
        """Test image update during evolution."""
        # Mock the diffusers module and FluxPipeline class
        mock_diffusers = Mock()
        mock_flux_pipeline = Mock()
        mock_diffusers.FluxPipeline = mock_flux_pipeline
        
        with patch.dict('sys.modules', {'diffusers': mock_diffusers}):
                # Mock the pipeline
                mock_pipe = Mock()
                mock_image = Mock()
                mock_pipe.return_value.images = [mock_image]
                mock_flux_pipeline.from_pretrained.return_value = mock_pipe
                
                # Change life stage to simulate evolution
                sample_digipal.life_stage = LifeStage.CHILD
                
                with patch.object(image_generator, '_save_to_cache') as mock_save:
                    mock_save.return_value = Path("evolution_path.png")
                    
                    result_path = image_generator.update_image_for_evolution(sample_digipal)
                    
                    assert result_path == "evolution_path.png"
                    assert "child" in sample_digipal.image_generation_prompt.lower()
    
    def test_cache_cleanup(self, image_generator):
        """Test cache cleanup functionality."""
        # Create some test cached images
        old_image = Image.new('RGB', (100, 100), color='red')
        recent_image = Image.new('RGB', (100, 100), color='blue')
        
        old_path = image_generator._save_to_cache(old_image, "old_image")
        recent_path = image_generator._save_to_cache(recent_image, "recent_image")
        
        # Manually set old timestamp (simulate old file)
        import os
        import time
        old_timestamp = time.time() - (35 * 24 * 3600)  # 35 days ago
        os.utime(old_path, (old_timestamp, old_timestamp))
        
        # Run cleanup (max age 30 days)
        image_generator.cleanup_cache(max_age_days=30)
        
        # Old image should be deleted, recent should remain
        assert not old_path.exists()
        assert recent_path.exists()
    
    def test_get_cache_info(self, image_generator):
        """Test cache information retrieval."""
        # Create some test cached images
        test_image = Image.new('RGB', (100, 100), color='green')
        image_generator._save_to_cache(test_image, "info_test")
        
        cache_info = image_generator.get_cache_info()
        
        assert isinstance(cache_info, dict)
        assert "cache_dir" in cache_info
        assert "cached_images" in cache_info
        assert "total_size_mb" in cache_info
        assert "model_loaded" in cache_info
        assert cache_info["cached_images"] >= 1
        assert cache_info["total_size_mb"] >= 0  # Allow 0 for very small files
        assert cache_info["model_loaded"] == image_generator._model_loaded
    
    def test_consistent_seed_generation(self, image_generator):
        """Test that same DigiPal ID produces consistent seeds."""
        digipal1 = DigiPal(id="test-123")
        digipal2 = DigiPal(id="test-123")  # Same ID
        digipal3 = DigiPal(id="test-456")  # Different ID
        
        # Test that hash function produces consistent results
        seed1 = hash(digipal1.id) % (2**32)
        seed2 = hash(digipal2.id) % (2**32)
        seed3 = hash(digipal3.id) % (2**32)
        
        # Same ID should produce same seed
        assert seed1 == seed2
        # Different ID should produce different seed
        assert seed1 != seed3
        
        # Test with actual torch.Generator mock
        with patch('torch.Generator') as mock_generator:
            mock_gen_instance = Mock()
            mock_generator.return_value = mock_gen_instance
            
            # Mock the diffusers module and FluxPipeline class
            mock_diffusers = Mock()
            mock_flux = Mock()
            mock_diffusers.FluxPipeline = mock_flux
            
            with patch.dict('sys.modules', {'diffusers': mock_diffusers}):
                mock_pipe = Mock()
                mock_pipe.return_value.images = [Mock()]
                mock_flux.from_pretrained.return_value = mock_pipe
                
                with patch.object(image_generator, '_save_to_cache') as mock_save:
                    mock_save.return_value = Path("test.png")
                    
                    # Generate image - should call manual_seed with consistent seed
                    image_generator.generate_image(digipal1)
                    
                    # Verify that manual_seed was called
                    mock_gen_instance.manual_seed.assert_called_once_with(seed1)


if __name__ == "__main__":
    pytest.main([__file__])