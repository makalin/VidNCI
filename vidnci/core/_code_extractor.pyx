# cython: language_level=3
# distutils: language=c++

"""
Cython implementation of the code image extraction algorithm.

This module provides a high-performance implementation of the core
NCI algorithm for extracting embedded noise codes from videos.
"""

import numpy as np
cimport numpy as np
from libcpp cimport bool
from libc.math cimport sqrt, fabs

# Define numpy data types
np.import_array()
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def extract_code_image_cython(
    np.ndarray[DTYPE_t, ndim=3] frames,
    np.ndarray[DTYPE_t, ndim=1] code
):
    """
    Extract code image from video frames using Cython.
    
    Args:
        frames: Video frames array of shape (num_frames, height, width)
        code: Noise code sequence of length num_frames
        
    Returns:
        numpy.ndarray: Recovered code image of shape (height, width)
    """
    cdef:
        Py_ssize_t num_frames = frames.shape[0]
        Py_ssize_t height = frames.shape[1]
        Py_ssize_t width = frames.shape[2]
        Py_ssize_t i, j, t
        np.ndarray[DTYPE_t, ndim=2] code_image
        DTYPE_t code_value, frame_value
        DTYPE_t code_squared_sum = 0.0
    
    # Calculate sum of squared code values
    for t in range(num_frames):
        code_squared_sum += code[t] * code[t]
    
    if code_squared_sum == 0:
        raise ValueError("Code cannot be all zeros")
    
    # Initialize code image
    code_image = np.zeros((height, width), dtype=DTYPE)
    
    # Main extraction loop
    for t in range(num_frames):
        code_value = code[t]
        
        for i in range(height):
            for j in range(width):
                frame_value = frames[t, i, j]
                code_image[i, j] += code_value * frame_value
    
    # Normalize by the sum of squared code values
    code_image /= code_squared_sum
    
    return code_image


def extract_multiple_codes_cython(
    np.ndarray[DTYPE_t, ndim=3] frames,
    np.ndarray[DTYPE_t, ndim=2] codes
):
    """
    Extract multiple code images from video frames using Cython.
    
    Args:
        frames: Video frames array of shape (num_frames, height, width)
        codes: Noise codes array of shape (num_codes, num_frames)
        
    Returns:
        numpy.ndarray: Array of recovered code images of shape (num_codes, height, width)
    """
    cdef:
        Py_ssize_t num_codes = codes.shape[0]
        Py_ssize_t num_frames = frames.shape[0]
        Py_ssize_t height = frames.shape[1]
        Py_ssize_t width = frames.shape[2]
        Py_ssize_t c, t, i, j
        np.ndarray[DTYPE_t, ndim=3] code_images
        np.ndarray[DTYPE_t, ndim=2] code_image
        DTYPE_t code_value, frame_value
        DTYPE_t code_squared_sum
    
    # Initialize output array
    code_images = np.zeros((num_codes, height, width), dtype=DTYPE)
    
    # Process each code
    for c in range(num_codes):
        # Calculate sum of squared code values for this code
        code_squared_sum = 0.0
        for t in range(num_frames):
            code_squared_sum += codes[c, t] * codes[c, t]
        
        if code_squared_sum == 0:
            continue  # Skip zero codes
        
        # Extract code image for this code
        code_image = np.zeros((height, width), dtype=DTYPE)
        
        for t in range(num_frames):
            code_value = codes[c, t]
            
            for i in range(height):
                for j in range(width):
                    frame_value = frames[t, i, j]
                    code_image[i, j] += code_value * frame_value
        
        # Normalize and store
        code_image /= code_squared_sum
        code_images[c] = code_image
    
    return code_images


def apply_gaussian_filter_cython(
    np.ndarray[DTYPE_t, ndim=2] image,
    DTYPE_t sigma
):
    """
    Apply Gaussian filtering to an image using Cython.
    
    Args:
        image: Input image
        sigma: Gaussian filter standard deviation
        
    Returns:
        numpy.ndarray: Filtered image
    """
    cdef:
        Py_ssize_t height = image.shape[0]
        Py_ssize_t width = image.shape[1]
        Py_ssize_t i, j, di, dj
        np.ndarray[DTYPE_t, ndim=2] filtered_image
        DTYPE_t kernel_sum, pixel_sum
        DTYPE_t kernel_value, distance
        int kernel_size
    
    # Determine kernel size based on sigma
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    filtered_image = np.zeros((height, width), dtype=DTYPE)
    
    # Apply Gaussian filter
    for i in range(height):
        for j in range(width):
            pixel_sum = 0.0
            kernel_sum = 0.0
            
            for di in range(-kernel_size//2, kernel_size//2 + 1):
                for dj in range(-kernel_size//2, kernel_size//2 + 1):
                    # Check bounds
                    if (0 <= i + di < height and 0 <= j + dj < width):
                        # Calculate distance from center
                        distance = sqrt(di * di + dj * dj)
                        
                        # Gaussian kernel
                        kernel_value = exp(-(distance * distance) / (2 * sigma * sigma))
                        
                        pixel_sum += image[i + di, j + dj] * kernel_value
                        kernel_sum += kernel_value
            
            if kernel_sum > 0:
                filtered_image[i, j] = pixel_sum / kernel_sum
    
    return filtered_image


def calculate_snr_cython(
    np.ndarray[DTYPE_t, ndim=2] code_image,
    np.ndarray[DTYPE_t, ndim=2] expected_code
):
    """
    Calculate Signal-to-Noise Ratio using Cython.
    
    Args:
        code_image: Recovered code image
        expected_code: Expected code pattern
        
    Returns:
        float: SNR in dB
    """
    cdef:
        Py_ssize_t height = code_image.shape[0]
        Py_ssize_t width = code_image.shape[1]
        Py_ssize_t i, j
        DTYPE_t signal_power = 0.0
        DTYPE_t noise_power = 0.0
        DTYPE_t diff
    
    # Calculate signal and noise power
    for i in range(height):
        for j in range(width):
            signal_power += expected_code[i, j] * expected_code[i, j]
            diff = code_image[i, j] - expected_code[i, j]
            noise_power += diff * diff
    
    if noise_power == 0:
        return float('inf')
    
    return 10.0 * log10(signal_power / noise_power)


def detect_anomalies_cython(
    np.ndarray[DTYPE_t, ndim=2] code_image,
    DTYPE_t threshold
):
    """
    Detect anomalies in code image using Cython.
    
    Args:
        code_image: Input code image
        threshold: Threshold for anomaly detection
        
    Returns:
        tuple: (anomaly_mask, anomaly_stats)
    """
    cdef:
        Py_ssize_t height = code_image.shape[0]
        Py_ssize_t width = code_image.shape[1]
        Py_ssize_t i, j
        np.ndarray[DTYPE_t, ndim=2] local_mean
        np.ndarray[DTYPE_t, ndim=2] local_std
        np.ndarray[bool, ndim=2] anomaly_mask
        DTYPE_t local_sum, local_sum_sq
        DTYPE_t local_var, local_std_val
        Py_ssize_t window_size = 15
        Py_ssize_t half_window = window_size // 2
        Py_ssize_t start_i, end_i, start_j, end_j
        Py_ssize_t count
        DTYPE_t anomaly_sum = 0.0
        Py_ssize_t anomaly_count = 0
    
    # Initialize arrays
    local_mean = np.zeros((height, width), dtype=DTYPE)
    local_std = np.zeros((height, width), dtype=DTYPE)
    anomaly_mask = np.zeros((height, width), dtype=bool)
    
    # Calculate local statistics
    for i in range(height):
        for j in range(width):
            # Define window boundaries
            start_i = max(0, i - half_window)
            end_i = min(height, i + half_window + 1)
            start_j = max(0, j - half_window)
            end_j = min(width, j + half_window + 1)
            
            # Calculate local mean and variance
            local_sum = 0.0
            local_sum_sq = 0.0
            count = 0
            
            for di in range(start_i, end_i):
                for dj in range(start_j, end_j):
                    local_sum += code_image[di, dj]
                    local_sum_sq += code_image[di, dj] * code_image[di, dj]
                    count += 1
            
            if count > 0:
                local_mean[i, j] = local_sum / count
                local_var = (local_sum_sq / count) - (local_mean[i, j] * local_mean[i, j])
                local_std[i, j] = sqrt(max(local_var, 0.0))
    
    # Detect anomalies
    for i in range(height):
        for j in range(width):
            if local_std[i, j] > 0:
                if fabs(code_image[i, j] - local_mean[i, j]) > (threshold * local_std[i, j]):
                    anomaly_mask[i, j] = True
                    anomaly_sum += fabs(code_image[i, j] - local_mean[i, j])
                    anomaly_count += 1
    
    # Calculate statistics
    anomaly_stats = {
        "total_pixels": height * width,
        "anomaly_pixels": int(anomaly_count),
        "anomaly_ratio": float(anomaly_count) / (height * width),
        "anomaly_severity": float(anomaly_sum / anomaly_count) if anomaly_count > 0 else 0.0,
        "threshold": threshold,
    }
    
    return anomaly_mask, anomaly_stats
