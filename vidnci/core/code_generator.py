"""
Code Generator Module for VidNCI

This module provides functionality to generate pseudo-random noise codes
for Noise-Coded Illumination (NCI) technique.
"""

import numpy as np
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class CodeGenerator:
    """
    Generates pseudo-random noise codes for Noise-Coded Illumination (NCI).
    
    This class provides methods to generate various types of noise codes
    that can be embedded into video illumination for forensic analysis.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the CodeGenerator.
        
        Args:
            seed: Random seed for reproducible code generation
        """
        if seed is not None:
            np.random.seed(seed)
            logger.info(f"CodeGenerator initialized with seed: {seed}")
        else:
            logger.info("CodeGenerator initialized with random seed")
    
    def generate_gaussian_code(
        self, 
        length: int, 
        mean: float = 0.0, 
        std: float = 1.0
    ) -> np.ndarray:
        """
        Generate a Gaussian noise code.
        
        Args:
            length: Length of the code sequence
            mean: Mean of the Gaussian distribution
            std: Standard deviation of the Gaussian distribution
            
        Returns:
            numpy.ndarray: Gaussian noise code sequence
            
        Raises:
            ValueError: If length is not positive
        """
        if length <= 0:
            raise ValueError("Length must be positive")
        
        code = np.random.normal(mean, std, length)
        logger.debug(f"Generated Gaussian code: length={length}, mean={mean}, std={std}")
        return code
    
    def generate_binary_code(
        self, 
        length: int, 
        p_positive: float = 0.5
    ) -> np.ndarray:
        """
        Generate a binary noise code (+1, -1).
        
        Args:
            length: Length of the code sequence
            p_positive: Probability of +1 (default: 0.5)
            
        Returns:
            numpy.ndarray: Binary noise code sequence with values +1 and -1
            
        Raises:
            ValueError: If length is not positive or p_positive is not in [0, 1]
        """
        if length <= 0:
            raise ValueError("Length must be positive")
        if not 0 <= p_positive <= 1:
            raise ValueError("p_positive must be in [0, 1]")
        
        # Generate random values in [0, 1)
        random_values = np.random.random(length)
        
        # Convert to binary code: +1 if random < p_positive, -1 otherwise
        code = np.where(random_values < p_positive, 1, -1)
        
        logger.debug(f"Generated binary code: length={length}, p_positive={p_positive}")
        return code
    
    def generate_bipolar_code(
        self, 
        length: int, 
        p_positive: float = 0.33, 
        p_negative: float = 0.33
    ) -> np.ndarray:
        """
        Generate a bipolar noise code (+1, 0, -1).
        
        Args:
            length: Length of the code sequence
            p_positive: Probability of +1 (default: 0.33)
            p_negative: Probability of -1 (default: 0.33)
            
        Returns:
            numpy.ndarray: Bipolar noise code sequence with values +1, 0, and -1
            
        Raises:
            ValueError: If probabilities are invalid or length is not positive
        """
        if length <= 0:
            raise ValueError("Length must be positive")
        if not 0 <= p_positive <= 1 or not 0 <= p_negative <= 1:
            raise ValueError("Probabilities must be in [0, 1]")
        if p_positive + p_negative > 1:
            raise ValueError("Sum of probabilities cannot exceed 1")
        
        # Generate random values in [0, 1)
        random_values = np.random.random(length)
        
        # Initialize code with zeros
        code = np.zeros(length)
        
        # Set +1 values
        code[random_values < p_positive] = 1
        
        # Set -1 values
        code[(random_values >= p_positive) & (random_values < p_positive + p_negative)] = -1
        
        logger.debug(f"Generated bipolar code: length={length}, p_positive={p_positive}, p_negative={p_negative}")
        return code
    
    def generate_uniform_code(
        self, 
        length: int, 
        low: float = -1.0, 
        high: float = 1.0
    ) -> np.ndarray:
        """
        Generate a uniform noise code.
        
        Args:
            length: Length of the code sequence
            low: Lower bound of the uniform distribution
            high: Upper bound of the uniform distribution
            
        Returns:
            numpy.ndarray: Uniform noise code sequence
            
        Raises:
            ValueError: If length is not positive or low >= high
        """
        if length <= 0:
            raise ValueError("Length must be positive")
        if low >= high:
            raise ValueError("low must be less than high")
        
        code = np.random.uniform(low, high, length)
        logger.debug(f"Generated uniform code: length={length}, range=[{low}, {high})")
        return code
    
    def generate_multiple_codes(
        self, 
        num_codes: int, 
        length: int, 
        code_type: str = "gaussian",
        **kwargs
    ) -> np.ndarray:
        """
        Generate multiple noise codes.
        
        Args:
            num_codes: Number of codes to generate
            length: Length of each code sequence
            code_type: Type of code to generate ("gaussian", "binary", "bipolar", "uniform")
            **kwargs: Additional arguments for the specific code generator
            
        Returns:
            numpy.ndarray: Array of shape (num_codes, length) containing the codes
            
        Raises:
            ValueError: If num_codes or length is not positive
            ValueError: If code_type is not supported
        """
        if num_codes <= 0:
            raise ValueError("num_codes must be positive")
        if length <= 0:
            raise ValueError("length must be positive")
        
        # Map code types to generator methods
        generator_methods = {
            "gaussian": self.generate_gaussian_code,
            "binary": self.generate_binary_code,
            "bipolar": self.generate_bipolar_code,
            "uniform": self.generate_uniform_code,
        }
        
        if code_type not in generator_methods:
            raise ValueError(f"Unsupported code_type: {code_type}. Supported types: {list(generator_methods.keys())}")
        
        generator_method = generator_methods[code_type]
        codes = np.array([generator_method(length, **kwargs) for _ in range(num_codes)])
        
        logger.info(f"Generated {num_codes} {code_type} codes of length {length}")
        return codes
    
    def validate_code_properties(
        self, 
        code: np.ndarray
    ) -> dict:
        """
        Validate and analyze the properties of a generated code.
        
        Args:
            code: The noise code to analyze
            
        Returns:
            dict: Dictionary containing code properties and statistics
        """
        properties = {
            "length": len(code),
            "mean": float(np.mean(code)),
            "std": float(np.std(code)),
            "min": float(np.min(code)),
            "max": float(np.max(code)),
            "unique_values": list(np.unique(code)),
            "zero_crossings": int(np.sum(np.diff(np.signbit(code)))),
        }
        
        # Calculate autocorrelation for first few lags
        max_lag = min(10, len(code) // 2)
        autocorr = [1.0]  # lag 0
        for lag in range(1, max_lag + 1):
            if lag < len(code):
                corr = np.corrcoef(code[:-lag], code[lag:])[0, 1]
                autocorr.append(float(corr) if not np.isnan(corr) else 0.0)
        
        properties["autocorrelation"] = autocorr
        
        logger.debug(f"Code properties analyzed: {properties}")
        return properties
    
    def save_code(
        self, 
        code: np.ndarray, 
        filename: str
    ) -> None:
        """
        Save a code to a file.
        
        Args:
            code: The noise code to save
            filename: Output filename
            
        Raises:
            ValueError: If filename is empty
        """
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        np.save(filename, code)
        logger.info(f"Code saved to {filename}")
    
    def load_code(self, filename: str) -> np.ndarray:
        """
        Load a code from a file.
        
        Args:
            filename: Input filename
            
        Returns:
            numpy.ndarray: The loaded noise code
            
        Raises:
            ValueError: If filename is empty
            FileNotFoundError: If file doesn't exist
        """
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        try:
            code = np.load(filename)
            logger.info(f"Code loaded from {filename}")
            return code
        except FileNotFoundError:
            logger.error(f"File not found: {filename}")
            raise
        except Exception as e:
            logger.error(f"Error loading code from {filename}: {e}")
            raise
