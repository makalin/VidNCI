#!/usr/bin/env python3
"""
Command-line interface for VidNCI library.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from . import VidNCIAnalyzer


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="VidNCI - Video Noise-Coded Illumination Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single video
  vidnci analyze video.mp4

  # Analyze with specific settings
  vidnci analyze video.mp4 --code-type binary --analysis-type ai_detection

  # Batch analyze multiple videos
  vidnci batch video1.mp4 video2.mp4 video3.mp4

  # Compare multiple videos
  vidnci compare video1.mp4 video2.mp4

  # Get system information
  vidnci info
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single video')
    analyze_parser.add_argument('video_path', help='Path to video file')
    analyze_parser.add_argument('--code-type', choices=['gaussian', 'binary', 'bipolar', 'uniform'],
                               default='gaussian', help='Type of noise code to generate')
    analyze_parser.add_argument('--embedding-strength', type=float, default=0.1,
                               help='Strength of code embedding (0.0 to 1.0)')
    analyze_parser.add_argument('--analysis-type', choices=['basic', 'comprehensive', 'ai_detection'],
                               default='comprehensive', help='Type of analysis to perform')
    analyze_parser.add_argument('--save-results', action='store_true',
                               help='Save analysis results to files')
    analyze_parser.add_argument('--output-dir', type=str, default='results',
                               help='Directory to save results')
    analyze_parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    analyze_parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               default='INFO', help='Logging level')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Analyze multiple videos')
    batch_parser.add_argument('video_paths', nargs='+', help='Paths to video files')
    batch_parser.add_argument('--code-type', choices=['gaussian', 'binary', 'bipolar', 'uniform'],
                             default='gaussian', help='Type of noise code to generate')
    batch_parser.add_argument('--embedding-strength', type=float, default=0.1,
                             help='Strength of code embedding (0.0 to 1.0)')
    batch_parser.add_argument('--analysis-type', choices=['basic', 'comprehensive', 'ai_detection'],
                             default='comprehensive', help='Type of analysis to perform')
    batch_parser.add_argument('--save-results', action='store_true',
                             help='Save analysis results to files')
    batch_parser.add_argument('--output-dir', type=str, default='results',
                             help='Directory to save results')
    batch_parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    batch_parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                             default='INFO', help='Logging level')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare multiple videos')
    compare_parser.add_argument('video_paths', nargs='+', help='Paths to video files')
    compare_parser.add_argument('--code-type', choices=['gaussian', 'binary', 'bipolar', 'uniform'],
                               default='gaussian', help='Type of noise code to generate')
    compare_parser.add_argument('--analysis-type', choices=['basic', 'comprehensive', 'ai_detection'],
                               default='comprehensive', help='Type of analysis to perform')
    compare_parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    compare_parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                               default='INFO', help='Logging level')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'analyze':
            return analyze_video(args)
        elif args.command == 'batch':
            return batch_analyze(args)
        elif args.command == 'compare':
            return compare_videos(args)
        elif args.command == 'info':
            return show_info()
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        return 1


def analyze_video(args) -> int:
    """Analyze a single video."""
    print(f"Analyzing video: {args.video_path}")
    
    # Initialize analyzer
    analyzer = VidNCIAnalyzer(
        seed=args.seed,
        log_level=args.log_level
    )
    
    # Perform analysis
    results = analyzer.analyze_video(
        video_path=args.video_path,
        code_type=args.code_type,
        embedding_strength=args.embedding_strength,
        analysis_type=args.analysis_type,
        save_results=args.save_results,
        output_dir=args.output_dir
    )
    
    # Display results
    display_analysis_results(results)
    
    return 0


def batch_analyze(args) -> int:
    """Analyze multiple videos."""
    print(f"Batch analyzing {len(args.video_paths)} videos...")
    
    # Initialize analyzer
    analyzer = VidNCIAnalyzer(
        seed=args.seed,
        log_level=args.log_level
    )
    
    # Perform batch analysis
    results = analyzer.batch_analyze(
        video_paths=args.video_paths,
        code_type=args.code_type,
        embedding_strength=args.embedding_strength,
        analysis_type=args.analysis_type,
        save_results=args.save_results,
        output_dir=args.output_dir
    )
    
    # Display batch results
    display_batch_results(results)
    
    return 0


def compare_videos(args) -> int:
    """Compare multiple videos."""
    print(f"Comparing {len(args.video_paths)} videos...")
    
    # Initialize analyzer
    analyzer = VidNCIAnalyzer(
        seed=args.seed,
        log_level=args.log_level
    )
    
    # Perform comparison
    comparison_results = analyzer.compare_videos(
        video_paths=args.video_paths,
        code_type=args.code_type,
        analysis_type=args.analysis_type
    )
    
    # Display comparison results
    display_comparison_results(comparison_results)
    
    return 0


def show_info() -> int:
    """Show system information."""
    analyzer = VidNCIAnalyzer()
    info = analyzer.get_system_info()
    
    print("VidNCI System Information")
    print("=" * 40)
    print(f"Python Version: {info['python_version']}")
    print(f"NumPy Version: {info['numpy_version']}")
    print(f"OpenCV Version: {info['opencv_version']}")
    print(f"VidNCI Version: {info['vidnci_version']}")
    print(f"Extraction Implementation: {info['extraction_implementation']}")
    
    print("\nComponents:")
    for component, status in info['components'].items():
        print(f"  {component}: {status}")
    
    return 0


def display_analysis_results(results: dict):
    """Display analysis results in a formatted way."""
    print("\n" + "=" * 50)
    print("VIDNCI ANALYSIS RESULTS")
    print("=" * 50)
    
    # Video info
    print(f"Video: {results['video_info']['width']}x{results['video_info']['height']}, "
          f"{results['video_info']['fps']:.2f} fps, {results['video_info']['frame_count']} frames")
    
    # Code info
    print(f"Code Type: {results['code_info']['type']}, Length: {results['code_info']['length']}")
    
    # Analysis summary
    if 'summary' in results:
        summary = results['summary']
        
        if 'ai_detection' in summary:
            ai = summary['ai_detection']
            print(f"\nAI Detection:")
            print(f"  Probability: {ai['probability']:.3f}")
            print(f"  Confidence: {ai['confidence']:.3f}")
            print(f"  Flags: {', '.join(ai['flags'])}")
        
        if 'quality' in summary:
            quality = summary['quality']
            print(f"\nQuality Metrics:")
            print(f"  SNR: {quality['snr_db']:.2f} dB")
            print(f"  PSNR: {quality['psnr_db']:.2f} dB")
        
        if 'anomalies' in summary:
            anomalies = summary['anomalies']
            print(f"\nAnomaly Detection:")
            print(f"  Ratio: {anomalies['ratio']:.3f}")
            print(f"  Severity: {anomalies['severity']:.3f}")
    
    print("\n" + "=" * 50)


def display_batch_results(results: list):
    """Display batch analysis results."""
    print(f"\nBatch Analysis Complete: {len(results)} videos processed")
    print("=" * 50)
    
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]
    
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print("\nSuccessful Analyses:")
        for i, result in enumerate(successful):
            video_name = Path(result['video_info']['video_path']).name
            ai_prob = result['analysis_results']['ai_detection']['ai_probability']
            print(f"  {i+1}. {video_name}: AI Probability = {ai_prob:.3f}")
    
    if failed:
        print("\nFailed Analyses:")
        for i, result in enumerate(failed):
            video_name = Path(result['video_path']).name
            error = result['error']
            print(f"  {i+1}. {video_name}: {error}")


def display_comparison_results(comparison_results: dict):
    """Display video comparison results."""
    print(f"\nVideo Comparison Results: {comparison_results['video_count']} videos")
    print("=" * 50)
    
    if comparison_results['comparison_metrics']:
        print("\nIndividual Metrics:")
        for metrics in comparison_results['comparison_metrics']:
            video_name = Path(metrics['video_path']).name
            print(f"\n  {video_name}:")
            print(f"    AI Probability: {metrics['ai_probability']:.3f}")
            print(f"    SNR: {metrics['snr']:.2f} dB")
            print(f"    Anomaly Ratio: {metrics['anomaly_ratio']:.3f}")
            print(f"    Entropy: {metrics['entropy']:.3f}")
        
        if comparison_results['comparison_stats']:
            print("\nComparison Statistics:")
            stats = comparison_results['comparison_stats']
            
            for metric_name, metric_stats in stats.items():
                print(f"\n  {metric_name.replace('_', ' ').title()}:")
                print(f"    Mean: {metric_stats['mean']:.3f}")
                print(f"    Std: {metric_stats['std']:.3f}")
                print(f"    Range: [{metric_stats['min']:.3f}, {metric_stats['max']:.3f}]")


if __name__ == '__main__':
    sys.exit(main())
