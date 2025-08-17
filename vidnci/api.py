"""
High-Level API for VidNCI

This module provides a simple and intuitive interface for the complete
VidNCI workflow, from code generation to analysis.
"""

import numpy as np
from typing import Optional, Union, Dict, Any, List
import logging
from pathlib import Path

from .core.code_generator import CodeGenerator
from .core.video_embedder import VideoEmbedder
from .core.code_extractor import CodeExtractor
from .core.analyzer import Analyzer

logger = logging.getLogger(__name__)


class VidNCIAnalyzer:
    """
    High-level interface for the complete VidNCI workflow.
    
    This class provides a simple API for performing Noise-Coded Illumination
    analysis on videos, from code generation to final analysis.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        use_cython: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize the VidNCI Analyzer.
        
        Args:
            seed: Random seed for reproducible code generation
            use_cython: Whether to use Cython implementation if available
            log_level: Logging level
        """
        # Set up logging
        logging.basicConfig(level=getattr(logging, log_level.upper()))
        
        # Initialize components
        self.code_generator = CodeGenerator(seed=seed)
        self.video_embedder = VideoEmbedder()
        self.code_extractor = CodeExtractor(use_cython=use_cython)
        self.analyzer = Analyzer(log_level=log_level)
        
        logger.info("VidNCI Analyzer initialized successfully")
    
    def analyze_video(
        self,
        video_path: Union[str, Path],
        code_type: str = "gaussian",
        code_length: Optional[int] = None,
        embedding_strength: float = 0.1,
        analysis_type: str = "comprehensive",
        save_results: bool = False,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Perform complete analysis of a video using the NCI technique.
        
        Args:
            video_path: Path to the video file to analyze
            code_type: Type of noise code to generate ("gaussian", "binary", "bipolar", "uniform")
            code_length: Length of the noise code (default: auto-detect from video)
            embedding_strength: Strength of code embedding for simulation (0.0 to 1.0)
            analysis_type: Type of analysis to perform ("basic", "comprehensive", "ai_detection")
            save_results: Whether to save intermediate results
            output_dir: Directory to save results (default: current directory)
            
        Returns:
            dict: Complete analysis results
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If parameters are invalid
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Set output directory
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Starting complete analysis of video: {video_path}")
        
        # Step 1: Get video information
        video_info = self.video_embedder.get_video_info(video_path)
        logger.info(f"Video properties: {video_info['width']}x{video_info['height']}, "
                   f"{video_info['fps']:.2f} fps, {video_info['frame_count']} frames")
        
        # Step 2: Generate noise code
        if code_length is None:
            code_length = video_info['frame_count']
        
        logger.info(f"Generating {code_type} noise code of length {code_length}")
        code = self.code_generator.generate_gaussian_code(code_length) if code_type == "gaussian" else \
               self.code_generator.generate_binary_code(code_length) if code_type == "binary" else \
               self.code_generator.generate_bipolar_code(code_length) if code_type == "bipolar" else \
               self.code_generator.generate_uniform_code(code_length)
        
        # Step 3: Simulate code embedding (for demonstration/testing)
        if embedding_strength > 0:
            logger.info(f"Simulating code embedding with strength {embedding_strength}")
            embedded_video_path = output_dir / f"embedded_{video_path.name}"
            
            success = self.video_embedder.embed_single_code(
                video_path, embedded_video_path, code, embedding_strength
            )
            
            if not success:
                logger.warning("Code embedding failed, proceeding with original video")
                embedded_video_path = video_path
        else:
            embedded_video_path = video_path
        
        # Step 4: Extract code image
        logger.info("Extracting code image from video")
        code_image = self.code_extractor.extract_code_image(
            embedded_video_path, code, normalize=True, apply_filtering=True
        )
        
        # Step 5: Analyze the recovered code image
        logger.info(f"Performing {analysis_type} analysis")
        analysis_results = self.analyzer.analyze_code_image(
            code_image, expected_code=None, analysis_type=analysis_type
        )
        
        # Step 6: Compile final results
        final_results = {
            "video_info": video_info,
            "code_info": {
                "type": code_type,
                "length": code_length,
                "properties": self.code_generator.validate_code_properties(code),
            },
            "embedding_info": {
                "strength": embedding_strength,
                "embedded_video_path": str(embedded_video_path),
            },
            "extraction_info": self.code_extractor.get_extraction_info(),
            "analysis_results": analysis_results,
            "summary": self._generate_summary(analysis_results),
        }
        
        # Step 7: Save results if requested
        if save_results:
            self._save_analysis_results(final_results, code_image, output_dir)
        
        logger.info("Video analysis completed successfully")
        return final_results
    
    def batch_analyze(
        self,
        video_paths: List[Union[str, Path]],
        code_type: str = "gaussian",
        embedding_strength: float = 0.1,
        analysis_type: str = "comprehensive",
        save_results: bool = False,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform batch analysis of multiple videos.
        
        Args:
            video_paths: List of video file paths to analyze
            code_type: Type of noise code to generate
            embedding_strength: Strength of code embedding
            analysis_type: Type of analysis to perform
            save_results: Whether to save results
            output_dir: Directory to save results
            
        Returns:
            List of analysis results for each video
        """
        if not video_paths:
            raise ValueError("No video paths provided")
        
        logger.info(f"Starting batch analysis of {len(video_paths)} videos")
        
        results = []
        for i, video_path in enumerate(video_paths):
            try:
                logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
                
                # Create subdirectory for each video
                if output_dir is not None:
                    video_name = Path(video_path).stem
                    video_output_dir = Path(output_dir) / video_name
                    video_output_dir.mkdir(exist_ok=True)
                else:
                    video_output_dir = None
                
                result = self.analyze_video(
                    video_path=video_path,
                    code_type=code_type,
                    embedding_strength=embedding_strength,
                    analysis_type=analysis_type,
                    save_results=save_results,
                    output_dir=video_output_dir
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing video {video_path}: {e}")
                results.append({
                    "video_path": str(video_path),
                    "error": str(e),
                    "status": "failed"
                })
        
        logger.info(f"Batch analysis completed: {len(results)} videos processed")
        return results
    
    def compare_videos(
        self,
        video_paths: List[Union[str, Path]],
        code_type: str = "gaussian",
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Compare multiple videos using the NCI technique.
        
        Args:
            video_paths: List of video file paths to compare
            code_type: Type of noise code to generate
            analysis_type: Type of analysis to perform
            
        Returns:
            dict: Comparison results
        """
        if len(video_paths) < 2:
            raise ValueError("At least 2 videos required for comparison")
        
        logger.info(f"Starting comparison of {len(video_paths)} videos")
        
        # Analyze each video
        individual_results = self.batch_analyze(
            video_paths, code_type, 0.0, analysis_type, False
        )
        
        # Extract key metrics for comparison
        comparison_metrics = []
        for result in individual_results:
            if "error" not in result:
                metrics = {
                    "video_path": result["video_info"]["video_path"],
                    "ai_probability": result["analysis_results"].get("ai_detection", {}).get("ai_probability", 0.0),
                    "snr": result["analysis_results"].get("quality_metrics", {}).get("snr_db", 0.0),
                    "anomaly_ratio": result["analysis_results"].get("anomaly_detection", {}).get("statistics", {}).get("anomaly_ratio", 0.0),
                    "entropy": result["analysis_results"].get("basic_stats", {}).get("entropy", 0.0),
                }
                comparison_metrics.append(metrics)
        
        # Calculate comparison statistics
        if comparison_metrics:
            ai_probs = [m["ai_probability"] for m in comparison_metrics]
            snrs = [m["snr"] for m in comparison_metrics]
            anomaly_ratios = [m["anomaly_ratio"] for m in comparison_metrics]
            entropies = [m["entropy"] for m in comparison_metrics]
            
            comparison_stats = {
                "ai_probability": {
                    "mean": float(np.mean(ai_probs)),
                    "std": float(np.std(ai_probs)),
                    "min": float(np.min(ai_probs)),
                    "max": float(np.max(ai_probs)),
                },
                "snr": {
                    "mean": float(np.mean(snrs)),
                    "std": float(np.std(snrs)),
                    "min": float(np.min(snrs)),
                    "max": float(np.max(snrs)),
                },
                "anomaly_ratio": {
                    "mean": float(np.mean(anomaly_ratios)),
                    "std": float(np.std(anomaly_ratios)),
                    "min": float(np.min(anomaly_ratios)),
                    "max": float(np.max(anomaly_ratios)),
                },
                "entropy": {
                    "mean": float(np.mean(entropies)),
                    "std": float(np.std(entropies)),
                    "min": float(np.min(entropies)),
                    "max": float(np.max(entropies)),
                },
            }
        else:
            comparison_stats = {}
        
        comparison_results = {
            "video_count": len(video_paths),
            "individual_results": individual_results,
            "comparison_metrics": comparison_metrics,
            "comparison_stats": comparison_stats,
        }
        
        logger.info("Video comparison completed")
        return comparison_results
    
    def _generate_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of the analysis results.
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            dict: Summary information
        """
        summary = {}
        
        # Basic summary
        if "basic_stats" in analysis_results:
            basic = analysis_results["basic_stats"]
            summary["basic"] = {
                "mean_value": basic.get("mean", 0.0),
                "std_value": basic.get("std", 0.0),
                "value_range": basic.get("range", 0.0),
            }
        
        # Quality summary
        if "quality_metrics" in analysis_results:
            quality = analysis_results["quality_metrics"]
            summary["quality"] = {
                "snr_db": quality.get("snr_db", 0.0),
                "psnr_db": quality.get("psnr_db", 0.0),
            }
        
        # AI detection summary
        if "ai_detection" in analysis_results:
            ai = analysis_results["ai_detection"]
            summary["ai_detection"] = {
                "probability": ai.get("ai_probability", 0.0),
                "confidence": ai.get("confidence", 0.0),
                "flags": ai.get("flags", []),
            }
        
        # Anomaly summary
        if "anomaly_detection" in analysis_results:
            anomaly = analysis_results["anomaly_detection"]["statistics"]
            summary["anomalies"] = {
                "ratio": anomaly.get("anomaly_ratio", 0.0),
                "severity": anomaly.get("anomaly_severity", 0.0),
            }
        
        return summary
    
    def _save_analysis_results(
        self,
        results: Dict[str, Any],
        code_image: np.ndarray,
        output_dir: Path
    ) -> None:
        """
        Save analysis results to files.
        
        Args:
            results: Analysis results to save
            code_image: Recovered code image
            output_dir: Directory to save results
        """
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        results_file = output_dir / f"analysis_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._prepare_for_json(results)
            json.dump(json_results, f, indent=2, default=str)
        
        # Save code image
        image_file = output_dir / f"code_image_{timestamp}.png"
        self.code_extractor.save_code_image(code_image, str(image_file))
        
        # Create visualization
        viz_file = output_dir / f"analysis_visualization_{timestamp}.png"
        self.analyzer.visualize_analysis(code_image, results["analysis_results"], str(viz_file))
        
        logger.info(f"Analysis results saved to {output_dir}")
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """
        Prepare object for JSON serialization.
        
        Args:
            obj: Object to prepare
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        else:
            return obj
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the VidNCI system.
        
        Returns:
            dict: System information
        """
        import sys
        import cv2
        import numpy as np
        
        info = {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "opencv_version": cv2.__version__,
            "vidnci_version": "0.1.0",
            "components": {
                "code_generator": "available",
                "video_embedder": "available",
                "code_extractor": "available",
                "analyzer": "available",
            },
            "extraction_implementation": self.code_extractor.get_extraction_info()["implementation"],
        }
        
        return info
