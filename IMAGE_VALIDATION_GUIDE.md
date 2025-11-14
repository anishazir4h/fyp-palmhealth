# Image Validation Guide

## Overview
The system now includes automatic image quality validation to prevent processing of unsuitable images.

## Validation Filters

### 1. **Resolution Check**
- **Minimum:** 400x400 pixels
- **Maximum:** 8000x8000 pixels
- **Reason:** Too small = not enough detail; Too large = satellite view (too zoomed out)

### 2. **Blur Detection**
- Uses Laplacian variance to measure image sharpness
- **Threshold:** Variance < 50 = Too blurry
- **Reason:** Blurry images cannot accurately detect individual palm crowns

### 3. **Contrast Check**
- Measures standard deviation of pixel values
- **Threshold:** Std < 20 = Insufficient contrast
- **Reason:** Low contrast images lack distinguishable features

### 4. **Vegetation Density**
- Analyzes green color ratio in HSV color space
- **Too zoomed out:** Green ratio > 85% (uniform vegetation)
- **Not a plantation:** Green ratio < 5% (no palm trees)
- **Reason:** Ensures image is at appropriate zoom level with visible individual palms

## Examples of REJECTED Images

### ❌ Too Zoomed Out
- Satellite imagery where individual palms are not distinguishable
- Uniform green canopy with no individual tree crowns
- **Your example images appear to fall in this category**

### ❌ Too Blurry
- Out of focus drone/aerial images
- Motion blur from camera movement
- Low resolution images upscaled

### ❌ Wrong Content
- Ground-level photos
- Images with no vegetation
- Non-palm tree images

## Examples of ACCEPTED Images

### ✅ Ideal Images
- Clear aerial/drone imagery
- Individual palm crowns are visible
- Moderate zoom level (not too close, not too far)
- Good lighting and contrast
- Resolution between 400-8000px

## How to Fix Rejected Images

1. **For "too zoomed out":** Use higher resolution drone imagery or zoom in closer
2. **For "too blurry":** 
   - Use better camera equipment
   - Ensure proper focus
   - Avoid motion during capture
3. **For "insufficient contrast":** Adjust camera settings or capture during better lighting conditions
4. **For "wrong resolution":** Resize or re-capture at appropriate resolution

## Technical Implementation

The validation happens in the `validate_image_quality()` function before any detection processing:

```python
def validate_image_quality(image, min_resolution=400, max_resolution=8000):
    # Checks resolution, blur, contrast, and vegetation density
    # Returns (is_valid, reason)
```

## Adjusting Thresholds

You can modify the validation thresholds in `stm.py`:

```python
# Resolution limits
min_resolution=400  # Increase for stricter quality
max_resolution=8000  # Decrease to reject very large images

# Blur threshold
if laplacian_var < 50:  # Increase to be more strict

# Green density thresholds
if green_ratio > 0.85:  # Too uniform (decrease to be less strict)
if green_ratio < 0.05:  # Too little vegetation (increase to be less strict)
```

## User Feedback

When an image is rejected, users see:
- ❌ Clear error message explaining why
- Specific metric values (e.g., "sharpness score: 45.3")
- Actionable guidance on how to fix the issue
