"""
Code Extractor Module for VidNCI

This module provides functionality to extract embedded noise codes from videos
using the Noise-Coded Illumination (NCI) technique.
"""

import cv2
import numpy as np
from typing import Optional, Union, Tuple, List
import logging
from pathlib import Path
from scipy import signal
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


class CodeExtractor:
    """
    Extracts embedded noise codes from videos using the NCI technique.
    
    This class implements the core algorithm for code image recovery,
    which is based on the mathematical model from the research paper.
    """
    
    def __init__(self, use_cython: bool = True):
        """
        Initialize the CodeExtractor.
        
        Args:
            use_cython: Whether to use Cython implementation if available
        """
        self.use_cython = use_cython
        
        # Try to import Cython implementation
        if use_cython:
            try:
                from ._code_extractor import extract_code_image_cython
                self._cython_extract = extract_code_image_cython
                logger.info("Cython implementation loaded successfully")
            except ImportError:
                logger.warning("Cython implementation not available, using Python fallback")
                self._cython_extract = None
        else:
            self._cython_extract = None
        
        logger.info(f"CodeExtractor initialized (use_cython={use_cython})")
    
    def extract_code_image(
        self,
        video_path: Union[str, Path],
        code: np.ndarray,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        normalize: bool = True,
        apply_filtering: bool = True
    ) -> np.ndarray:
        """
        Extract the code image from a video.
        
        Args:
            video_path: Path to the video file
            code: Known noise code sequence
            start_frame: Starting frame index (default: 0)
            end_frame: Ending frame index (default: None, process all frames)
            normalize: Whether to normalize the output (default: True)
            apply_filtering: Whether to apply Gaussian filtering (default: True)
            
        Returns:
            numpy.ndarray: Recovered code image
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If parameters are invalid
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if len(code) <= 0:
            raise ValueError("Code must have positive length")
        
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        
        if end_frame is not None and end_frame <= start_frame:
            raise ValueError("end_frame must be greater than start_frame")
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if end_frame is None:
                end_frame = total_frames
            
            if end_frame > total_frames:
                logger.warning(f"end_frame {end_frame} exceeds video length {total_frames}, using {total_frames}")
                end_frame = total_frames
            
            logger.info(f"Extracting code image from {start_frame} to {end_frame} frames")
            logger.info(f"Video dimensions: {width}x{height}")
            
            # Use Cython implementation if available
            if self._cython_extract is not None:
                code_image = self._extract_with_cython(
                    cap, code, start_frame, end_frame, width, height
                )
            else:
                code_image = self._extract_with_python(
                    cap, code, start_frame, end_frame, width, height
                )
            
            # Apply post-processing
            if apply_filtering:
                code_image = self._apply_post_processing(code_image)
            
            if normalize:
                code_image = self._normalize_code_image(code_image)
            
            cap.release()
            
            logger.info("Code image extraction completed successfully")
            return code_image
            
        except Exception as e:
            logger.error(f"Error extracting code image: {e}")
            if 'cap' in locals():
                cap.release()
            raise
    
    def _extract_with_cython(
        self,
        cap: cv2.VideoCapture,
        code: np.ndarray,
        start_frame: int,
        end_frame: int,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Extract code image using Cython implementation.
        
        Args:
            cap: OpenCV video capture object
            code: Noise code sequence
            start_frame: Starting frame index
            end_frame: Ending frame index
            width: Video width
            height: Video height
            
        Returns:
            numpy.ndarray: Recovered code image
        """
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Prepare arrays for Cython function
        frames = []
        frame_count = 0
        
        while frame_count < (end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and normalize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            frames.append(gray)
            frame_count += 1
        
        if not frames:
            raise ValueError("No frames could be read from video")
        
        # Convert to numpy array
        frames_array = np.array(frames)
        
        # Call Cython function
        code_image = self._cython_extract(frames_array, code)
        
        return code_image
    
    def _extract_with_python(
        self,
        cap: cv2.VideoCapture,
        code: np.ndarray,
        start_frame: int,
        end_frame: int,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Extract code image using Python implementation.
        
        Args:
            cap: OpenCV video capture object
            code: Noise code sequence
            start_frame: Starting frame index
            end_frame: Ending frame index
            width: Video width
            height: Video height
            
        Returns:
            numpy.ndarray: Recovered code image
        """
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Initialize code image
        code_image = np.zeros((height, width), dtype=np.float64)
        code_squared_sum = np.sum(code ** 2)
        
        if code_squared_sum == 0:
            raise ValueError("Code cannot be all zeros")
        
        frame_count = 0
        total_frames = end_frame - start_frame
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale and normalize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
            
            # Get code value for current frame
            code_index = frame_count % len(code)
            code_value = code[code_index]
            
            # Apply the NCI recovery equation
            # r_i(x) = Σ(c_i(t) * y_c(x,t)) / Σ(c_i^2(t))
            code_image += code_value * gray
            
            frame_count += 1
            
            # Log progress
            if frame_count % 100 == 0:
                logger.debug(f"Processed {frame_count}/{total_frames} frames")
        
        # Normalize by the sum of squared code values
        code_image /= code_squared_sum
        
        return code_image
    
    def extract_multiple_codes(
        self,
        video_path: Union[str, Path],
        codes: np.ndarray,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        normalize: bool = True,
        apply_filtering: bool = True
    ) -> List[np.ndarray]:
        """
        Extract multiple code images from a video.
        
        Args:
            video_path: Path to the video file
            codes: Array of noise codes, shape (num_codes, code_length)
            start_frame: Starting frame index (default: 0)
            end_frame: Ending frame index (default: None, process all frames)
            normalize: Whether to normalize the output (default: True)
            apply_filtering: Whether to apply Gaussian filtering (default: True)
            
        Returns:
            List[numpy.ndarray]: List of recovered code images
            
        Raises:
            ValueError: If codes array is not 2D
        """
        if codes.ndim != 2:
            raise ValueError("Codes must be 2D array with shape (num_codes, code_length)")
        
        num_codes = codes.shape[0]
        logger.info(f"Extracting {num_codes} code images from video")
        
        code_images = []
        for i, code in enumerate(codes):
            logger.debug(f"Extracting code {i+1}/{num_codes}")
            code_image = self.extract_code_image(
                video_path, code, start_frame, end_frame, normalize, apply_filtering
            )
            code_images.append(code_image)
        
        return code_images
    
    def _apply_post_processing(self, code_image: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to the recovered code image.
        
        Args:
            code_image: Raw recovered code image
            
        Returns:
            numpy.ndarray: Processed code image
        """
        # Apply Gaussian filtering to reduce noise
        sigma = 1.0
        filtered_image = gaussian_filter(code_image, sigma=sigma)
        
        logger.debug(f"Applied Gaussian filtering with sigma={sigma}")
        return filtered_image
    
    def _normalize_code_image(self, code_image: np.ndarray) -> np.ndarray:
        """
        Normalize the code image to [0, 1] range.
        
        Args:
            code_image: Input code image
            
        Returns:
            numpy.ndarray: Normalized code image
        """
        # Get min and max values
        min_val = np.min(code_image)
        max_val = np.max(code_image)
        
        # Avoid division by zero
        if max_val == min_val:
            return np.zeros_like(code_image)
        
        # Normalize to [0, 1]
        normalized = (code_image - min_val) / (max_val - min_val)
        
        logger.debug(f"Normalized code image: range [{min_val:.4f}, {max_val:.4f}] -> [0, 1]")
        return normalized
    
    def calculate_snr(
        self,
        code_image: np.ndarray,
        expected_code: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate Signal-to-Noise Ratio (SNR) of the recovered code image.
        
        Args:
            code_image: Recovered code image
            expected_code: Expected code pattern (if known)
            
        Returns:
            float: Signal-to-Noise Ratio in dB
        """
        if expected_code is not None:
            # Calculate SNR against expected pattern
            signal_power = np.sum(expected_code ** 2)
            noise_power = np.sum((code_image - expected_code) ** 2)
        else:
            # Estimate SNR using variance
            signal_power = np.var(code_image)
            noise_power = np.var(code_image - np.mean(code_image))
        
        if noise_power == 0:
            return float('inf')
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        logger.debug(f"Calculated SNR: {snr_db:.2f} dB")
        return snr_db
    
    def detect_anomalies(
        self,
        code_image: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[np.ndarray, dict]:
        """
        Detect anomalies in the recovered code image.
        
        Args:
            code_image: Recovered code image
            threshold: Threshold for anomaly detection
            
        Returns:
            Tuple[numpy.ndarray, dict]: Anomaly mask and statistics
        """
        # Calculate local statistics
        from scipy.ndimage import uniform_filter
        
        # Local mean
        local_mean = uniform_filter(code_image, size=15)
        
        # Local standard deviation
        local_var = uniform_filter(code_image ** 2, size=15) - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Detect anomalies (pixels that deviate significantly from local mean)
        anomaly_mask = np.abs(code_image - local_mean) > (threshold * local_std)
        
        # Calculate statistics
        total_pixels = code_image.size
        anomaly_pixels = np.sum(anomaly_mask)
        anomaly_ratio = anomaly_pixels / total_pixels
        
        stats = {
            "total_pixels": total_pixels,
            "anomaly_pixels": anomaly_pixels,
            "anomaly_ratio": anomaly_ratio,
            "threshold": threshold,
        }
        
        logger.info(f"Anomaly detection completed: {anomaly_ratio:.2%} pixels flagged as anomalous")
        return anomaly_mask, stats
    
    def save_code_image(
        self,
        code_image: np.ndarray,
        filename: str,
        format: str = "png"
    ) -> None:
        """
        Save the recovered code image to a file.
        
        Args:
            code_image: Code image to save
            filename: Output filename
            format: Image format (png, jpg, tiff, etc.)
            
        Raises:
            ValueError: If filename is empty
        """
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        try:
            # Ensure the image is in [0, 255] range for saving
            if code_image.max() <= 1.0:
                save_image = (code_image * 255).astype(np.uint8)
            else:
                save_image = code_image.astype(np.uint8)
            
            # Save using OpenCV
            cv2.imwrite(filename, save_image)
            logger.info(f"Code image saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving code image to {filename}: {e}")
            raise
    
    def get_extraction_info(self) -> dict:
        """
        Get information about the extraction process.
        
        Returns:
            dict: Extraction information
        """
        info = {
            "use_cython": self.use_cython,
            "cython_available": self._cython_extract is not None,
            "implementation": "cython" if self._cython_extract else "python",
        }
        
        return info
