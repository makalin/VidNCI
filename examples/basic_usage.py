#!/usr/bin/env python3
"""
Basic Usage Example for VidNCI

This example demonstrates the basic workflow of using VidNCI
to analyze a video for AI-generated content detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the parent directory to the path to import vidnci
sys.path.insert(0, str(Path(__file__).parent.parent))

from vidnci import VidNCIAnalyzer


def main():
    """Main function demonstrating basic VidNCI usage."""
    
    print("=== VidNCI Basic Usage Example ===\n")
    
    # Initialize the VidNCI analyzer
    print("1. Initializing VidNCI Analyzer...")
    analyzer = VidNCIAnalyzer(seed=42, log_level="INFO")
    
    # Get system information
    system_info = analyzer.get_system_info()
    print(f"   Python version: {system_info['python_version']}")
    print(f"   NumPy version: {system_info['numpy_version']}")
    print(f"   OpenCV version: {system_info['opencv_version']}")
    print(f"   Extraction implementation: {system_info['extraction_implementation']}")
    
    # Check if we have a video file to analyze
    # For this example, we'll create a synthetic video
    print("\n2. Creating synthetic test video...")
    test_video_path = create_synthetic_video()
    print(f"   Test video created: {test_video_path}")
    
    # Analyze the video
    print("\n3. Analyzing video with VidNCI...")
    try:
        results = analyzer.analyze_video(
            video_path=test_video_path,
            code_type="gaussian",
            embedding_strength=0.1,  # Simulate code embedding
            analysis_type="ai_detection",
            save_results=True,
            output_dir="results"
        )
        
        # Display results
        print("\n4. Analysis Results:")
        print("   Video Info:")
        print(f"     Dimensions: {results['video_info']['width']}x{results['video_info']['height']}")
        print(f"     FPS: {results['video_info']['fps']:.2f}")
        print(f"     Frame count: {results['video_info']['frame_count']}")
        
        print("\n   Code Info:")
        print(f"     Type: {results['code_info']['type']}")
        print(f"     Length: {results['code_info']['length']}")
        
        print("\n   AI Detection:")
        ai_info = results['analysis_results']['ai_detection']
        print(f"     AI Probability: {ai_info['ai_probability']:.3f}")
        print(f"     Confidence: {ai_info['confidence']:.3f}")
        print(f"     Flags: {', '.join(ai_info['flags'])}")
        
        print("\n   Quality Metrics:")
        quality = results['analysis_results']['quality_metrics']
        print(f"     SNR: {quality['snr_db']:.2f} dB")
        print(f"     PSNR: {quality['psnr_db']:.2f} dB")
        
        print("\n   Anomaly Detection:")
        anomaly = results['analysis_results']['anomaly_detection']['statistics']
        print(f"     Anomaly ratio: {anomaly['anomaly_ratio']:.3f}")
        print(f"     Anomaly severity: {anomaly['anomaly_severity']:.3f}")
        
        # Create a simple visualization
        print("\n5. Creating visualization...")
        create_visualization(results)
        
        print("\n=== Analysis Complete ===")
        print("Results have been saved to the 'results' directory.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test video
        if test_video_path.exists():
            test_video_path.unlink()
            print(f"\nTest video cleaned up: {test_video_path}")


def create_synthetic_video():
    """Create a synthetic test video for demonstration."""
    import cv2
    
    # Video parameters
    width, height = 320, 240
    fps = 30
    duration = 3  # seconds
    total_frames = fps * duration
    
    # Create output directory
    output_dir = Path("test_data")
    output_dir.mkdir(exist_ok=True)
    
    # Video file path
    video_path = output_dir / "synthetic_test_video.mp4"
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    # Generate synthetic frames
    for frame_idx in range(total_frames):
        # Create a frame with some moving content
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a moving circle
        t = frame_idx / total_frames
        center_x = int(width * 0.5 + width * 0.3 * np.sin(2 * np.pi * t))
        center_y = int(height * 0.5 + height * 0.3 * np.cos(2 * np.pi * t))
        radius = 30
        
        # Draw the circle
        cv2.circle(frame, (center_x, center_y), radius, (255, 255, 255), -1)
        
        # Add some noise to make it more realistic
        noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()
    
    return video_path


def create_visualization(results):
    """Create a simple visualization of the results."""
    try:
        # Create a summary plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("VidNCI Analysis Results", fontsize=16)
        
        # AI Detection probability
        ai_prob = results['analysis_results']['ai_detection']['ai_probability']
        axes[0, 0].bar(['AI Probability'], [ai_prob], color='red' if ai_prob > 0.5 else 'green')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_title("AI Generation Probability")
        axes[0, 0].set_ylabel("Probability")
        
        # Quality metrics
        quality = results['analysis_results']['quality_metrics']
        metrics = ['SNR', 'PSNR']
        values = [quality['snr_db'], quality['psnr_db']]
        axes[0, 1].bar(metrics, values, color=['blue', 'orange'])
        axes[0, 1].set_title("Quality Metrics")
        axes[0, 1].set_ylabel("dB")
        
        # Anomaly detection
        anomaly = results['analysis_results']['anomaly_detection']['statistics']
        axes[1, 0].pie([anomaly['anomaly_ratio'], 1 - anomaly['anomaly_ratio']], 
                       labels=['Anomalous', 'Normal'], autopct='%1.1f%%')
        axes[1, 0].set_title("Anomaly Distribution")
        
        # Basic statistics
        basic = results['analysis_results']['basic_stats']
        stats = ['Mean', 'Std', 'Range']
        values = [basic['mean'], basic['std'], basic['range']]
        axes[1, 1].bar(stats, values, color=['purple', 'brown', 'pink'])
        axes[1, 1].set_title("Basic Statistics")
        
        plt.tight_layout()
        
        # Save the visualization
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        viz_path = output_dir / "results_summary.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   Visualization saved to: {viz_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"   Warning: Could not create visualization: {e}")


if __name__ == "__main__":
    main()
