"""
Reinhard Color Transfer Implementation
Implementation of "Color Transfer between Images" by Reinhard et al. (2001)
Uses orthogonal basis transformation (lαβ space) for decorrelated color manipulation.
"""

import numpy as np
import cv2
import os
from pathlib import Path


# transfomation matrix

# RGB to LMS Matrix (simulates human cone cell response)
# Based on Ruderman et al. psychophysical measurements
M_RGB_TO_LMS = np.array([
    [0.3811, 0.5783, 0.0402],
    [0.1967, 0.7244, 0.0782],
    [0.0241, 0.1288, 0.8444]
])

# LMS to lαβ Matrix (orthogonal basis transformation)
# Coefficients ensure orthogonality: dot product between basis vectors = 0
# l: achromatic (luminance), α: yellow-blue, β: red-green
r3, r6, r2 = np.sqrt(3), np.sqrt(6), np.sqrt(2)
M_LMS_TO_LAB = np.array([
    [1/r3,  1/r3,  1/r3],   # l  = (L + M + S) / sqrt(3)
    [1/r6,  1/r6, -2/r6],   # α  = (L + M - 2S) / sqrt(6)
    [1/r2, -1/r2,  0.0 ]    # β  = (L - M) / sqrt(2)
])

# Pre-compute inverse matrices for efficiency
M_LMS_TO_RGB = np.linalg.inv(M_RGB_TO_LMS)
M_LAB_TO_LMS = np.linalg.inv(M_LMS_TO_LAB)


# convert rgb 
def rgb_to_lab(img_rgb):
    """
    Convert RGB image to lαβ space through logarithmic LMS.
    
    Algorithm:
    1. RGB → LMS (cone response simulation)
    2. Apply log transform (normalize distribution)
    3. LMS → lαβ (change to orthogonal basis)
    
    Args:
        img_rgb: RGB image in range [0, 1], shape (H, W, 3)
    
    Returns:
        img_lab: Image in lαβ space, shape (H, W, 3)
    """
    h, w, c = img_rgb.shape
    img_flat = img_rgb.reshape((-1, 3))  # Flatten to (H*W, 3) for matrix ops
    
    # Step 1: Linear transformation RGB → LMS
    img_lms = img_flat @ M_RGB_TO_LMS.T
    
    # Step 2: Logarithmic transformation (avoid log(0) with epsilon)
    img_lms = np.log10(img_lms + 1e-8)
    
    # Step 3: Change of basis to orthogonal lαβ space
    img_lab = img_lms @ M_LMS_TO_LAB.T
    
    return img_lab.reshape((h, w, 3))


def lab_to_rgb(img_lab):
    """
    Convert lαβ image back to RGB space.
    
    Algorithm (inverse of rgb_to_lab):
    1. lαβ → LMS (inverse orthogonal transformation)
    2. Apply exponential (inverse log)
    3. LMS → RGB (inverse cone response)
    
    Args:
        img_lab: Image in lαβ space, shape (H, W, 3)
    
    Returns:
        img_rgb: RGB image in range [0, 1], shape (H, W, 3)
    """
    h, w, c = img_lab.shape
    img_flat = img_lab.reshape((-1, 3))
    
    # Step 1: Inverse orthogonal transformation lαβ → LMS
    img_lms = img_flat @ M_LAB_TO_LMS.T
    
    # Step 2: Inverse logarithmic transformation
    img_lms = np.power(10, img_lms)
    
    # Step 3: Inverse linear transformation LMS → RGB
    img_rgb = img_lms @ M_LMS_TO_RGB.T
    
    return img_rgb.reshape((h, w, 3))


# validate orthogonality
def analyze_covariance(img, space_name):
    """Calculate and display covariance matrix to validate orthogonality."""
    flat = img.reshape((-1, 3))
    cov_matrix = np.cov(flat, rowvar=False)
    
    print(f"\nCovariance Matrix ({space_name}):")
    print(np.array2string(cov_matrix, precision=4, suppress_small=True))
    
    off_diag = cov_matrix.copy()
    np.fill_diagonal(off_diag, 0)
    avg_corr = np.mean(np.abs(off_diag))
    
    print(f"Off-Diagonal Avg: {avg_corr:.6f}", end="")
    if avg_corr < 0.001:
        print(" -> DECORRELATED ✓")
    else:
        print(" -> CORRELATED")
    
    return cov_matrix


# color transfer
def global_color_transfer(source, target, verbose=True):
    """
    Perform global color statistics transfer from target to source.
    
    Algorithm (Reinhard Method):
    1. Convert both images to lαβ orthogonal space
    2. Calculate mean (μ) and standard deviation (σ) for each channel
    3. Transfer statistics: normalize source, then scale to target distribution
    4. Convert result back to RGB
    
    Mathematical formula for each pixel in each channel:
        result = (source - μ_source) * (σ_target / σ_source) + μ_target
    
    Args:
        source: Source image (RGB, [0,1]), whose colors will be modified
        target: Target image (RGB, [0,1]), reference for color statistics
    
    Returns:
        result: Transferred image (RGB, [0,1])
    """
    # Step 1: Transform both images to orthogonal lαβ space
    src_lab = rgb_to_lab(source)
    tgt_lab = rgb_to_lab(target)
    
    # Step 2: Calculate statistics per channel (l, α, β)
    # axis=(0,1) means calculate across all spatial dimensions (all pixels)
    src_mean = np.mean(src_lab, axis=(0, 1))  # Shape: (3,)
    src_std  = np.std(src_lab, axis=(0, 1))   # Shape: (3,)
    
    tgt_mean = np.mean(tgt_lab, axis=(0, 1))
    tgt_std  = np.std(tgt_lab, axis=(0, 1))
    
    if verbose:
        print(f"\nStatistics [l, α, β]:")
        print(f"  Source: μ={src_mean}, σ={src_std}")
        print(f"  Target: μ={tgt_mean}, σ={tgt_std}")
    
    # Step 3: Linear statistical transfer
    # Normalize source to zero mean and unit variance, then scale to target distribution
    res_lab = (src_lab - src_mean) * (tgt_std / (src_std + 1e-6)) + tgt_mean
    
    # Step 4: Reconstruct to RGB space
    res_rgb = lab_to_rgb(res_lab)
    
    # Clip to valid range [0, 1]
    return np.clip(res_rgb, 0, 1)


# helper i/o img
def load_image(path):
    """Load image and convert to RGB float [0, 1]."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR
    return img.astype(np.float32) / 255.0


def save_image(img, path):
    """Save RGB float [0, 1] image to file."""
    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)
    print(f"Saved: {path}")


# batch helper
def batch_process(source_dir, target_dir, output_dir):
    """Process all combinations of source and target images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    
    source_files = sorted([f for f in source_path.iterdir() 
                          if f.suffix.lower() in image_extensions])
    target_files = sorted([f for f in target_path.iterdir() 
                          if f.suffix.lower() in image_extensions])
    
    if not source_files:
        print(f"No source images found in {source_dir}")
        return
    if not target_files:
        print(f"No target images found in {target_dir}")
        return
    
    print(f"Sources: {len(source_files)}")
    print(f"Targets: {len(target_files)}")
    print("=" * 70)
    
    count = 0
    for src_file in source_files:
        for tgt_file in target_files:
            try:
                count += 1
                print(f"\n[{count}] {src_file.name} + {tgt_file.name}")
                
                source_img = load_image(src_file)
                target_img = load_image(tgt_file)
                
                # Show covariance analysis for every pair
                analyze_covariance(source_img, "RGB Space")
                src_lab = rgb_to_lab(source_img)
                analyze_covariance(src_lab, "lαβ Space")
                
                # Show statistics for every pair
                result_img = global_color_transfer(source_img, target_img, verbose=True)
                
                src_name = src_file.stem
                tgt_name = tgt_file.stem
                output_filename = f"result_{src_name}_TO_{tgt_name}.jpg"
                output_file = output_path / output_filename
                
                save_image(result_img, output_file)
                print("=" * 70)
                
            except Exception as e:
                print(f"ERROR: {e}")
    
    print(f"Process complete. {count} images saved to: {output_dir}")


# main
if __name__ == "__main__":
    # config
    SOURCE_DIR = "source"
    TARGET_DIR = "target"
    OUTPUT_DIR = "output"
    
    print("Reinhard Color Transfer - Batch Processing")
    print("=" * 70)
    
    #run
    batch_process(SOURCE_DIR, TARGET_DIR, OUTPUT_DIR)
