"""
Analyzer Module for VidNCI

This module provides functionality to analyze recovered code images
for anomalies and detect AI-generated content.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, List, Dict, Any
import logging
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class Analyzer:
    """
    Analyzes recovered code images for anomalies and AI-generated content.
    
    This class provides comprehensive analysis tools for detecting
    manipulations and identifying AI-generated videos.
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize the Analyzer.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        logging.getLogger(__name__).setLevel(getattr(logging, log_level.upper()))
        logger.info("Analyzer initialized")
    
    def analyze_code_image(
        self,
        code_image: np.ndarray,
        expected_code: Optional[np.ndarray] = None,
        analysis_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a recovered code image.
        
        Args:
            code_image: Recovered code image to analyze
            expected_code: Expected code pattern (if known)
            analysis_type: Type of analysis ("basic", "comprehensive", "ai_detection")
            
        Returns:
            dict: Analysis results containing various metrics and flags
            
        Raises:
            ValueError: If analysis_type is not supported
        """
        if analysis_type not in ["basic", "comprehensive", "ai_detection"]:
            raise ValueError(f"Unsupported analysis_type: {analysis_type}")
        
        logger.info(f"Starting {analysis_type} analysis of code image")
        
        # Basic analysis (always performed)
        basic_results = self._basic_analysis(code_image)
        
        if analysis_type == "basic":
            return basic_results
        
        # Comprehensive analysis
        comprehensive_results = self._comprehensive_analysis(code_image, expected_code)
        basic_results.update(comprehensive_results)
        
        if analysis_type == "comprehensive":
            return basic_results
        
        # AI detection analysis
        ai_results = self._ai_detection_analysis(code_image)
        basic_results.update(ai_results)
        
        logger.info("Analysis completed successfully")
        return basic_results
    
    def _basic_analysis(self, code_image: np.ndarray) -> Dict[str, Any]:
        """
        Perform basic statistical analysis of the code image.
        
        Args:
            code_image: Code image to analyze
            
        Returns:
            dict: Basic analysis results
        """
        # Basic statistics
        mean_val = float(np.mean(code_image))
        std_val = float(np.std(code_image))
        min_val = float(np.min(code_image))
        max_val = float(np.max(code_image))
        median_val = float(np.median(code_image))
        
        # Histogram analysis
        hist, bins = np.histogram(code_image, bins=50)
        hist_peaks = self._find_histogram_peaks(hist, bins)
        
        # Entropy (measure of randomness)
        hist_normalized = hist / np.sum(hist)
        entropy = float(-np.sum(hist_normalized * np.log2(hist_normalized + 1e-10)))
        
        results = {
            "basic_stats": {
                "mean": mean_val,
                "std": std_val,
                "min": min_val,
                "max": max_val,
                "median": median_val,
                "range": max_val - min_val,
            },
            "histogram": {
                "peaks": hist_peaks,
                "entropy": entropy,
                "bin_count": len(bins) - 1,
            },
            "image_properties": {
                "shape": code_image.shape,
                "total_pixels": code_image.size,
                "non_zero_pixels": np.count_nonzero(code_image),
            }
        }
        
        logger.debug("Basic analysis completed")
        return results
    
    def _comprehensive_analysis(
        self,
        code_image: np.ndarray,
        expected_code: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis including quality metrics.
        
        Args:
            code_image: Code image to analyze
            expected_code: Expected code pattern (if known)
            
        Returns:
            dict: Comprehensive analysis results
        """
        # Quality metrics
        snr = self._calculate_snr(code_image, expected_code)
        psnr = self._calculate_psnr(code_image, expected_code)
        
        # Spatial analysis
        spatial_stats = self._spatial_analysis(code_image)
        
        # Frequency domain analysis
        freq_stats = self._frequency_analysis(code_image)
        
        # Correlation analysis
        correlation_stats = self._correlation_analysis(code_image, expected_code)
        
        results = {
            "quality_metrics": {
                "snr_db": snr,
                "psnr_db": psnr,
            },
            "spatial_analysis": spatial_stats,
            "frequency_analysis": freq_stats,
            "correlation_analysis": correlation_stats,
        }
        
        logger.debug("Comprehensive analysis completed")
        return results
    
    def _ai_detection_analysis(self, code_image: np.ndarray) -> Dict[str, Any]:
        """
        Perform AI-specific detection analysis.
        
        Args:
            code_image: Code image to analyze
            
        Returns:
            dict: AI detection analysis results
        """
        # Anomaly detection
        anomaly_mask, anomaly_stats = self._detect_anomalies(code_image)
        
        # Pattern consistency analysis
        pattern_consistency = self._analyze_pattern_consistency(code_image)
        
        # Clustering analysis for detecting artificial patterns
        clustering_results = self._clustering_analysis(code_image)
        
        # Texture analysis
        texture_features = self._texture_analysis(code_image)
        
        # AI generation probability
        ai_probability = self._estimate_ai_probability(
            anomaly_stats, pattern_consistency, clustering_results, texture_features
        )
        
        results = {
            "anomaly_detection": {
                "mask": anomaly_mask,
                "statistics": anomaly_stats,
            },
            "pattern_consistency": pattern_consistency,
            "clustering_analysis": clustering_results,
            "texture_analysis": texture_features,
            "ai_detection": {
                "ai_probability": ai_probability,
                "confidence": self._calculate_confidence(ai_probability),
                "flags": self._generate_ai_flags(ai_probability, anomaly_stats),
            }
        }
        
        logger.debug("AI detection analysis completed")
        return results
    
    def _find_histogram_peaks(self, hist: np.ndarray, bins: np.ndarray) -> List[Dict[str, float]]:
        """
        Find peaks in the histogram.
        
        Args:
            hist: Histogram values
            bins: Bin edges
            
        Returns:
            List of peak information
        """
        from scipy.signal import find_peaks
        
        peaks, properties = find_peaks(hist, height=np.max(hist) * 0.1)
        
        peak_info = []
        for i, peak_idx in enumerate(peaks):
            peak_info.append({
                "bin_center": float((bins[peak_idx] + bins[peak_idx + 1]) / 2),
                "height": float(hist[peak_idx]),
                "prominence": float(properties["prominences"][i]) if "prominences" in properties else 0.0,
            })
        
        return peak_info
    
    def _calculate_snr(
        self,
        code_image: np.ndarray,
        expected_code: Optional[np.ndarray]
    ) -> float:
        """
        Calculate Signal-to-Noise Ratio.
        
        Args:
            code_image: Recovered code image
            expected_code: Expected code pattern
            
        Returns:
            float: SNR in dB
        """
        if expected_code is not None:
            signal_power = np.sum(expected_code ** 2)
            noise_power = np.sum((code_image - expected_code) ** 2)
        else:
            # Estimate using variance
            signal_power = np.var(code_image)
            noise_power = np.var(code_image - np.mean(code_image))
        
        if noise_power == 0:
            return float('inf')
        
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    def _calculate_psnr(
        self,
        code_image: np.ndarray,
        expected_code: Optional[np.ndarray]
    ) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            code_image: Recovered code image
            expected_code: Expected code pattern
            
        Returns:
            float: PSNR in dB
        """
        if expected_code is not None:
            mse = np.mean((code_image - expected_code) ** 2)
        else:
            # Estimate using variance
            mse = np.var(code_image)
        
        if mse == 0:
            return float('inf')
        
        max_val = np.max(code_image)
        psnr_db = 20 * np.log10(max_val / np.sqrt(mse))
        return psnr_db
    
    def _spatial_analysis(self, code_image: np.ndarray) -> Dict[str, Any]:
        """
        Perform spatial analysis of the code image.
        
        Args:
            code_image: Code image to analyze
            
        Returns:
            dict: Spatial analysis results
        """
        # Local statistics
        local_mean = uniform_filter(code_image, size=15)
        local_std = np.sqrt(uniform_filter(code_image ** 2, size=15) - local_mean ** 2)
        
        # Spatial gradients
        grad_x = np.gradient(code_image, axis=1)
        grad_y = np.gradient(code_image, axis=0)
        gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Edge density
        edge_density = np.mean(gradient_magnitude > np.std(gradient_magnitude))
        
        results = {
            "local_statistics": {
                "mean_std": float(np.std(local_mean)),
                "std_std": float(np.std(local_std)),
            },
            "gradients": {
                "mean_gradient": float(np.mean(gradient_magnitude)),
                "std_gradient": float(np.std(gradient_magnitude)),
                "edge_density": float(edge_density),
            }
        }
        
        return results
    
    def _frequency_analysis(self, code_image: np.ndarray) -> Dict[str, Any]:
        """
        Perform frequency domain analysis.
        
        Args:
            code_image: Code image to analyze
            
        Returns:
            dict: Frequency analysis results
        """
        # 2D FFT
        fft = np.fft.fft2(code_image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)
        
        # Power spectral density
        psd = magnitude_spectrum ** 2
        
        # Frequency bands
        height, width = code_image.shape
        center_y, center_x = height // 2, width // 2
        
        # Low frequency (center region)
        low_freq_mask = np.zeros_like(psd)
        low_freq_mask[center_y-10:center_y+10, center_x-10:center_x+10] = 1
        low_freq_power = np.sum(psd * low_freq_mask)
        
        # High frequency (edges)
        high_freq_mask = 1 - low_freq_mask
        high_freq_power = np.sum(psd * high_freq_mask)
        
        # Frequency ratio
        freq_ratio = high_freq_power / (low_freq_power + 1e-10)
        
        results = {
            "frequency_bands": {
                "low_freq_power": float(low_freq_power),
                "high_freq_power": float(high_freq_power),
                "freq_ratio": float(freq_ratio),
            },
            "spectral_properties": {
                "total_power": float(np.sum(psd)),
                "mean_power": float(np.mean(psd)),
                "power_std": float(np.std(psd)),
            }
        }
        
        return results
    
    def _correlation_analysis(
        self,
        code_image: np.ndarray,
        expected_code: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Perform correlation analysis.
        
        Args:
            code_image: Recovered code image
            expected_code: Expected code pattern
            
        Returns:
            dict: Correlation analysis results
        """
        if expected_code is not None:
            # Direct correlation
            correlation = np.corrcoef(code_image.flatten(), expected_code.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
            
            # Cross-correlation
            cross_corr = signal.correlate2d(code_image, expected_code, mode='same')
            max_cross_corr = np.max(cross_corr)
            
            results = {
                "direct_correlation": float(correlation),
                "max_cross_correlation": float(max_cross_corr),
                "correlation_available": True,
            }
        else:
            # Autocorrelation
            autocorr = signal.correlate2d(code_image, code_image, mode='same')
            autocorr_normalized = autocorr / np.max(autocorr)
            
            # Measure of pattern consistency
            pattern_consistency = np.std(autocorr_normalized)
            
            results = {
                "autocorrelation_std": float(pattern_consistency),
                "correlation_available": False,
            }
        
        return results
    
    def _detect_anomalies(
        self,
        code_image: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect anomalies in the code image.
        
        Args:
            code_image: Code image to analyze
            threshold: Threshold for anomaly detection
            
        Returns:
            Tuple of anomaly mask and statistics
        """
        # Local statistics
        local_mean = uniform_filter(code_image, size=15)
        local_var = uniform_filter(code_image ** 2, size=15) - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 0))
        
        # Detect anomalies
        anomaly_mask = np.abs(code_image - local_mean) > (threshold * local_std)
        
        # Statistics
        total_pixels = code_image.size
        anomaly_pixels = np.sum(anomaly_mask)
        anomaly_ratio = anomaly_pixels / total_pixels
        
        # Anomaly severity
        anomaly_severity = np.mean(np.abs(code_image[anomaly_mask] - local_mean[anomaly_mask]))
        
        stats = {
            "total_pixels": total_pixels,
            "anomaly_pixels": anomaly_pixels,
            "anomaly_ratio": anomaly_ratio,
            "anomaly_severity": float(anomaly_severity),
            "threshold": threshold,
        }
        
        return anomaly_mask, stats
    
    def _analyze_pattern_consistency(self, code_image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the consistency of patterns in the code image.
        
        Args:
            code_image: Code image to analyze
            
        Returns:
            dict: Pattern consistency analysis results
        """
        # Divide image into blocks and analyze consistency
        block_size = 32
        height, width = code_image.shape
        
        blocks_h = height // block_size
        blocks_w = width // block_size
        
        block_means = []
        block_stds = []
        
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = code_image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_means.append(np.mean(block))
                block_stds.append(np.std(block))
        
        # Consistency metrics
        mean_consistency = np.std(block_means)
        std_consistency = np.std(block_stds)
        
        results = {
            "block_analysis": {
                "block_size": block_size,
                "num_blocks": len(block_means),
                "mean_consistency": float(mean_consistency),
                "std_consistency": float(std_consistency),
            }
        }
        
        return results
    
    def _clustering_analysis(self, code_image: np.ndarray) -> Dict[str, Any]:
        """
        Perform clustering analysis to detect artificial patterns.
        
        Args:
            code_image: Code image to analyze
            
        Returns:
            dict: Clustering analysis results
        """
        # Sample points for clustering
        height, width = code_image.shape
        sample_size = min(1000, height * width)
        
        # Random sampling
        indices = np.random.choice(height * width, sample_size, replace=False)
        y_indices = indices // width
        x_indices = indices % width
        
        # Features: pixel value and position
        features = np.column_stack([
            code_image[y_indices, x_indices],
            x_indices / width,  # Normalized x position
            y_indices / height,  # Normalized y position
        ])
        
        # Normalize features
        features_normalized = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-10)
        
        # DBSCAN clustering
        try:
            clustering = DBSCAN(eps=0.1, min_samples=5).fit(features_normalized)
            labels = clustering.labels_
            
            # Calculate silhouette score (excluding noise points)
            valid_labels = labels[labels != -1]
            if len(np.unique(valid_labels)) > 1:
                silhouette = silhouette_score(features_normalized[labels != -1], valid_labels)
            else:
                silhouette = 0.0
            
            n_clusters = len(np.unique(labels[labels != -1]))
            noise_ratio = np.sum(labels == -1) / len(labels)
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
            silhouette = 0.0
            n_clusters = 0
            noise_ratio = 1.0
        
        results = {
            "clustering": {
                "n_clusters": n_clusters,
                "silhouette_score": float(silhouette),
                "noise_ratio": float(noise_ratio),
                "sample_size": sample_size,
            }
        }
        
        return results
    
    def _texture_analysis(self, code_image: np.ndarray) -> Dict[str, Any]:
        """
        Perform texture analysis of the code image.
        
        Args:
            code_image: Code image to analyze
            
        Returns:
            dict: Texture analysis results
        """
        # Gray-level co-occurrence matrix (GLCM) features
        from skimage.feature import graycomatrix, graycoprops
        
        # Convert to uint8 for GLCM
        if code_image.max() <= 1.0:
            img_uint8 = (code_image * 255).astype(np.uint8)
        else:
            img_uint8 = code_image.astype(np.uint8)
        
        # Calculate GLCM
        distances = [1]
        angles = [0, 45, 90, 135]
        
        glcm = graycomatrix(img_uint8, distances, angles, levels=256, symmetric=True, normed=True)
        
        # Extract texture properties
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        
        results = {
            "texture_features": {
                "contrast": [float(x) for x in contrast],
                "dissimilarity": [float(x) for x in dissimilarity],
                "homogeneity": [float(x) for x in homogeneity],
                "energy": [float(x) for x in energy],
                "correlation": [float(x) for x in correlation],
            }
        }
        
        return results
    
    def _estimate_ai_probability(
        self,
        anomaly_stats: Dict[str, Any],
        pattern_consistency: Dict[str, Any],
        clustering_results: Dict[str, Any],
        texture_features: Dict[str, Any]
    ) -> float:
        """
        Estimate the probability that the content is AI-generated.
        
        Args:
            Various analysis results
            
        Returns:
            float: AI generation probability [0, 1]
        """
        # Initialize probability
        ai_prob = 0.5  # Neutral starting point
        
        # Anomaly-based scoring
        anomaly_ratio = anomaly_stats.get("anomaly_ratio", 0.0)
        if anomaly_ratio > 0.1:  # High anomaly ratio
            ai_prob += 0.2
        elif anomaly_ratio < 0.01:  # Very low anomaly ratio (suspicious)
            ai_prob += 0.1
        
        # Pattern consistency scoring
        mean_consistency = pattern_consistency.get("block_analysis", {}).get("mean_consistency", 0.0)
        if mean_consistency < 0.01:  # Very consistent patterns (suspicious)
            ai_prob += 0.15
        
        # Clustering scoring
        silhouette_score = clustering_results.get("clustering", {}).get("silhouette_score", 0.0)
        if silhouette_score > 0.7:  # Very clear clusters (suspicious)
            ai_prob += 0.1
        
        # Texture scoring
        energy_values = texture_features.get("texture_features", {}).get("energy", [0.0])
        avg_energy = np.mean(energy_values)
        if avg_energy > 0.8:  # Very high energy (suspicious)
            ai_prob += 0.05
        
        # Clamp to [0, 1]
        ai_prob = np.clip(ai_prob, 0.0, 1.0)
        
        return float(ai_prob)
    
    def _calculate_confidence(self, ai_probability: float) -> float:
        """
        Calculate confidence in the AI probability estimate.
        
        Args:
            ai_probability: Estimated AI probability
            
        Returns:
            float: Confidence level [0, 1]
        """
        # Higher confidence for extreme probabilities
        if ai_probability < 0.2 or ai_probability > 0.8:
            confidence = 0.8
        elif ai_probability < 0.4 or ai_probability > 0.6:
            confidence = 0.6
        else:
            confidence = 0.4
        
        return confidence
    
    def _generate_ai_flags(
        self,
        ai_probability: float,
        anomaly_stats: Dict[str, Any]
    ) -> List[str]:
        """
        Generate flags based on analysis results.
        
        Args:
            ai_probability: Estimated AI probability
            anomaly_stats: Anomaly detection statistics
            
        Returns:
            List of flag strings
        """
        flags = []
        
        if ai_probability > 0.7:
            flags.append("HIGH_AI_PROBABILITY")
        elif ai_probability > 0.5:
            flags.append("MODERATE_AI_PROBABILITY")
        else:
            flags.append("LOW_AI_PROBABILITY")
        
        anomaly_ratio = anomaly_stats.get("anomaly_ratio", 0.0)
        if anomaly_ratio > 0.15:
            flags.append("HIGH_ANOMALY_RATIO")
        elif anomaly_ratio > 0.05:
            flags.append("MODERATE_ANOMALY_RATIO")
        
        return flags
    
    def visualize_analysis(
        self,
        code_image: np.ndarray,
        analysis_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create visualizations of the analysis results.
        
        Args:
            code_image: Original code image
            analysis_results: Analysis results
            save_path: Optional path to save the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("VidNCI Code Image Analysis", fontsize=16)
        
        # Original code image
        axes[0, 0].imshow(code_image, cmap='viridis')
        axes[0, 0].set_title("Recovered Code Image")
        axes[0, 0].axis('off')
        
        # Histogram
        axes[0, 1].hist(code_image.flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_title("Pixel Value Distribution")
        axes[0, 1].set_xlabel("Pixel Value")
        axes[0, 1].set_ylabel("Frequency")
        
        # Anomaly detection
        if "anomaly_detection" in analysis_results:
            anomaly_mask = analysis_results["anomaly_detection"]["mask"]
            axes[0, 2].imshow(anomaly_mask, cmap='Reds')
            axes[0, 2].set_title("Anomaly Detection")
            axes[0, 2].axis('off')
        
        # Spatial analysis
        if "spatial_analysis" in analysis_results:
            grad_x = np.gradient(code_image, axis=1)
            grad_y = np.gradient(code_image, axis=0)
            gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
            axes[1, 0].imshow(gradient_magnitude, cmap='hot')
            axes[1, 0].set_title("Gradient Magnitude")
            axes[1, 0].axis('off')
        
        # Frequency analysis
        if "frequency_analysis" in analysis_results:
            fft = np.fft.fft2(code_image)
            fft_shifted = np.fft.fftshift(fft)
            magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
            axes[1, 1].imshow(magnitude_spectrum, cmap='gray')
            axes[1, 1].set_title("Frequency Spectrum (log)")
            axes[1, 1].axis('off')
        
        # AI detection summary
        if "ai_detection" in analysis_results:
            ai_info = analysis_results["ai_detection"]
            axes[1, 2].text(0.1, 0.8, f"AI Probability: {ai_info['ai_probability']:.3f}", 
                           transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.6, f"Confidence: {ai_info['confidence']:.3f}", 
                           transform=axes[1, 2].transAxes, fontsize=12)
            axes[1, 2].text(0.1, 0.4, f"Flags: {', '.join(ai_info['flags'])}", 
                           transform=axes[1, 2].transAxes, fontsize=10)
            axes[1, 2].set_title("AI Detection Summary")
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
