"""
Tests for the CodeGenerator module.
"""

import pytest
import numpy as np
from vidnci.core.code_generator import CodeGenerator


class TestCodeGenerator:
    """Test cases for CodeGenerator class."""
    
    def test_init_with_seed(self):
        """Test initialization with a specific seed."""
        generator = CodeGenerator(seed=42)
        assert generator is not None
    
    def test_init_without_seed(self):
        """Test initialization without seed."""
        generator = CodeGenerator()
        assert generator is not None
    
    def test_generate_gaussian_code(self):
        """Test Gaussian code generation."""
        generator = CodeGenerator(seed=42)
        code = generator.generate_gaussian_code(100, mean=0.0, std=1.0)
        
        assert isinstance(code, np.ndarray)
        assert len(code) == 100
        assert np.abs(np.mean(code)) < 0.5  # Should be close to 0
        assert np.abs(np.std(code) - 1.0) < 0.5  # Should be close to 1
    
    def test_generate_binary_code(self):
        """Test binary code generation."""
        generator = CodeGenerator(seed=42)
        code = generator.generate_binary_code(100, p_positive=0.5)
        
        assert isinstance(code, np.ndarray)
        assert len(code) == 100
        assert set(np.unique(code)) == {-1, 1}
        assert np.abs(np.sum(code == 1) / len(code) - 0.5) < 0.2
    
    def test_generate_bipolar_code(self):
        """Test bipolar code generation."""
        generator = CodeGenerator(seed=42)
        code = generator.generate_bipolar_code(100, p_positive=0.3, p_negative=0.3)
        
        assert isinstance(code, np.ndarray)
        assert len(code) == 100
        assert set(np.unique(code)) == {-1, 0, 1}
    
    def test_generate_uniform_code(self):
        """Test uniform code generation."""
        generator = CodeGenerator(seed=42)
        code = generator.generate_uniform_code(100, low=-1.0, high=1.0)
        
        assert isinstance(code, np.ndarray)
        assert len(code) == 100
        assert np.all(code >= -1.0)
        assert np.all(code < 1.0)
    
    def test_generate_multiple_codes(self):
        """Test multiple code generation."""
        generator = CodeGenerator(seed=42)
        codes = generator.generate_multiple_codes(3, 100, "gaussian")
        
        assert isinstance(codes, np.ndarray)
        assert codes.shape == (3, 100)
    
    def test_invalid_length(self):
        """Test that invalid length raises ValueError."""
        generator = CodeGenerator()
        
        with pytest.raises(ValueError):
            generator.generate_gaussian_code(0)
        
        with pytest.raises(ValueError):
            generator.generate_gaussian_code(-1)
    
    def test_invalid_probabilities(self):
        """Test that invalid probabilities raise ValueError."""
        generator = CodeGenerator()
        
        with pytest.raises(ValueError):
            generator.generate_binary_code(100, p_positive=1.5)
        
        with pytest.raises(ValueError):
            generator.generate_bipolar_code(100, p_positive=0.6, p_negative=0.5)
    
    def test_validate_code_properties(self):
        """Test code property validation."""
        generator = CodeGenerator(seed=42)
        code = generator.generate_gaussian_code(100)
        properties = generator.validate_code_properties(code)
        
        assert "length" in properties
        assert "mean" in properties
        assert "std" in properties
        assert properties["length"] == 100
    
    def test_save_and_load_code(self, tmp_path):
        """Test code saving and loading."""
        generator = CodeGenerator(seed=42)
        code = generator.generate_gaussian_code(100)
        
        # Save code
        filename = tmp_path / "test_code.npy"
        generator.save_code(code, str(filename))
        
        # Load code
        loaded_code = generator.load_code(str(filename))
        
        assert np.array_equal(code, loaded_code)
    
    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises FileNotFoundError."""
        generator = CodeGenerator()
        
        with pytest.raises(FileNotFoundError):
            generator.load_code("nonexistent_file.npy")
    
    def test_empty_filename(self):
        """Test that empty filename raises ValueError."""
        generator = CodeGenerator()
        code = generator.generate_gaussian_code(100)
        
        with pytest.raises(ValueError):
            generator.save_code(code, "")
        
        with pytest.raises(ValueError):
            generator.load_code("")
