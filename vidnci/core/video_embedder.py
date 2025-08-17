"""
Video Embedder Module for VidNCI

This module provides functionality to simulate coded illumination in videos
by embedding noise codes into video frames.
"""

import cv2
import numpy as np
from typing import Optional, Union, Tuple, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoEmbedder:
    """
    Embeds noise codes into videos to simulate coded illumination.
    
    This class provides methods to embed noise codes into video frames,
    simulating the effect of Noise-Coded Illumination (NCI) technique.
    """
    
    def __init__(self, output_format: str = "mp4v"):
        """
        Initialize the VideoEmbedder.
        
        Args:
            output_format: Output video codec (default: "mp4v")
        """
        self.output_format = output_format
        self.supported_formats = ["mp4v", "XVID", "MJPG", "H264"]
        
        if output_format not in self.supported_formats:
            logger.warning(f"Output format {output_format} may not be supported on all systems")
        
        logger.info(f"VideoEmbedder initialized with output format: {output_format}")
    
    def embed_single_code(
        self,
        input_video: Union[str, Path],
        output_video: Union[str, Path],
        code: np.ndarray,
        strength: float = 0.1,
        mask: Optional[np.ndarray] = None
    ) -> bool:
        """
        Embed a single noise code into a video.
        
        Args:
            input_video: Path to input video file
            output_video: Path to output video file
            code: Noise code sequence to embed
            strength: Strength of the embedding (0.0 to 1.0)
            mask: Optional mask defining where to apply the code (same size as video frames)
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            ValueError: If parameters are invalid
            FileNotFoundError: If input video doesn't exist
        """
        if not Path(input_video).exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")
        
        if strength < 0.0 or strength > 1.0:
            raise ValueError("Strength must be between 0.0 and 1.0")
        
        if len(code) <= 0:
            raise ValueError("Code must have positive length")
        
        try:
            # Open input video
            cap = cv2.VideoCapture(str(input_video))
            if not cap.isOpened():
                logger.error(f"Could not open input video: {input_video}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Initialize output video writer
            fourcc = cv2.VideoWriter_fourcc(*self.output_format)
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Could not create output video: {output_video}")
                cap.release()
                return False
            
            # Process frames
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get code value for current frame
                code_index = frame_count % len(code)
                code_value = code[code_index]
                
                # Apply the code to the frame
                modified_frame = self._apply_code_to_frame(
                    frame, code_value, strength, mask
                )
                
                # Write modified frame
                out.write(modified_frame)
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    logger.debug(f"Processed {frame_count}/{total_frames} frames")
            
            # Cleanup
            cap.release()
            out.release()
            
            logger.info(f"Successfully embedded code into video: {output_video}")
            return True
            
        except Exception as e:
            logger.error(f"Error embedding code into video: {e}")
            return False
    
    def embed_multiple_codes(
        self,
        input_video: Union[str, Path],
        output_video: Union[str, Path],
        codes: np.ndarray,
        strengths: Optional[List[float]] = None,
        masks: Optional[List[np.ndarray]] = None
    ) -> bool:
        """
        Embed multiple noise codes into a video.
        
        Args:
            input_video: Path to input video file
            output_video: Path to output video file
            codes: Array of noise codes, shape (num_codes, code_length)
            strengths: List of embedding strengths for each code
            masks: List of masks for each code
            
        Returns:
            bool: True if successful, False otherwise
        """
        if codes.ndim != 2:
            raise ValueError("Codes must be 2D array with shape (num_codes, code_length)")
        
        num_codes = codes.shape[0]
        
        # Set default strengths if not provided
        if strengths is None:
            strengths = [0.1] * num_codes
        
        # Set default masks if not provided
        if masks is None:
            masks = [None] * num_codes
        
        # Validate parameters
        if len(strengths) != num_codes:
            raise ValueError("Number of strengths must match number of codes")
        if len(masks) != num_codes:
            raise ValueError("Number of masks must match number of codes")
        
        try:
            # Open input video
            cap = cv2.VideoCapture(str(input_video))
            if not cap.isOpened():
                logger.error(f"Could not open input video: {input_video}")
                return False
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Embedding {num_codes} codes into video: {width}x{height}, {fps} fps, {total_frames} frames")
            
            # Initialize output video writer
            fourcc = cv2.VideoWriter_fourcc(*self.output_format)
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error(f"Could not create output video: {output_video}")
                cap.release()
                return False
            
            # Process frames
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply all codes to the frame
                modified_frame = frame.copy()
                for i, code in enumerate(codes):
                    code_index = frame_count % len(code)
                    code_value = code[code_index]
                    strength = strengths[i]
                    mask = masks[i]
                    
                    modified_frame = self._apply_code_to_frame(
                        modified_frame, code_value, strength, mask
                    )
                
                # Write modified frame
                out.write(modified_frame)
                frame_count += 1
                
                # Log progress
                if frame_count % 100 == 0:
                    logger.debug(f"Processed {frame_count}/{total_frames} frames")
            
            # Cleanup
            cap.release()
            out.release()
            
            logger.info(f"Successfully embedded {num_codes} codes into video: {output_video}")
            return True
            
        except Exception as e:
            logger.error(f"Error embedding multiple codes into video: {e}")
            return False
    
    def _apply_code_to_frame(
        self,
        frame: np.ndarray,
        code_value: float,
        strength: float,
        mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Apply a code value to a single frame.
        
        Args:
            frame: Input frame (BGR format)
            code_value: Code value to apply
            strength: Embedding strength
            mask: Optional mask defining where to apply the code
            
        Returns:
            numpy.ndarray: Modified frame
        """
        # Convert to float for calculations
        frame_float = frame.astype(np.float32)
        
        # Create the embedding effect
        # The code modulates the pixel values
        embedding = code_value * strength * 255.0
        
        if mask is not None:
            # Apply mask if provided
            if mask.shape[:2] != frame.shape[:2]:
                logger.warning("Mask dimensions don't match frame dimensions, resizing mask")
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            
            # Apply embedding only where mask is non-zero
            for c in range(3):  # BGR channels
                frame_float[:, :, c] += embedding * mask[:, :, c] if mask.ndim == 3 else mask
        
        else:
            # Apply embedding to entire frame
            frame_float += embedding
        
        # Clip values to valid range [0, 255]
        frame_float = np.clip(frame_float, 0, 255)
        
        # Convert back to uint8
        return frame_float.astype(np.uint8)
    
    def create_uniform_mask(
        self,
        width: int,
        height: int,
        value: float = 1.0
    ) -> np.ndarray:
        """
        Create a uniform mask for code embedding.
        
        Args:
            width: Mask width
            height: Mask height
            value: Mask value (default: 1.0)
            
        Returns:
            numpy.ndarray: Uniform mask
        """
        mask = np.full((height, width), value, dtype=np.float32)
        logger.debug(f"Created uniform mask: {width}x{height}, value={value}")
        return mask
    
    def create_radial_mask(
        self,
        width: int,
        height: int,
        center: Optional[Tuple[int, int]] = None,
        max_radius: Optional[float] = None
    ) -> np.ndarray:
        """
        Create a radial mask for code embedding.
        
        Args:
            width: Mask width
            height: Mask height
            center: Center point (x, y), defaults to image center
            max_radius: Maximum radius, defaults to distance to corner
            
        Returns:
            numpy.ndarray: Radial mask with values decreasing from center
        """
        if center is None:
            center = (width // 2, height // 2)
        
        if max_radius is None:
            max_radius = np.sqrt(center[0]**2 + center[1]**2)
        
        # Create coordinate grids
        y, x = np.ogrid[:height, :width]
        
        # Calculate distances from center
        distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # Create radial mask (1.0 at center, 0.0 at max_radius)
        mask = np.maximum(0, 1.0 - distances / max_radius)
        
        logger.debug(f"Created radial mask: {width}x{height}, center={center}, max_radius={max_radius}")
        return mask
    
    def create_rectangular_mask(
        self,
        width: int,
        height: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        fade_edges: bool = True
    ) -> np.ndarray:
        """
        Create a rectangular mask for code embedding.
        
        Args:
            width: Mask width
            height: Mask height
            x1, y1: Top-left corner
            x2, y2: Bottom-right corner
            fade_edges: Whether to fade the edges of the rectangle
            
        Returns:
            numpy.ndarray: Rectangular mask
        """
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Ensure coordinates are within bounds
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        
        if fade_edges:
            # Create smooth edges
            edge_width = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            
            for i in range(edge_width):
                alpha = i / edge_width
                # Top and bottom edges
                if y1 + i < y2 - i:
                    mask[y1 + i, x1:x2] = alpha
                    mask[y2 - i - 1, x1:x2] = alpha
                # Left and right edges
                if x1 + i < x2 - i:
                    mask[y1:y2, x1 + i] = alpha
                    mask[y1:y2, x2 - i - 1] = alpha
        else:
            # Sharp edges
            mask[y1:y2, x1:x2] = 1.0
        
        logger.debug(f"Created rectangular mask: {width}x{height}, region=({x1},{y1})-({x2},{y2})")
        return mask
    
    def get_video_info(self, video_path: Union[str, Path]) -> dict:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Video information
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            "fourcc": cap.get(cv2.CAP_PROP_FOURCC),
        }
        
        cap.release()
        return info
