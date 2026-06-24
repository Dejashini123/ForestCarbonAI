import numpy as np
from PIL import Image
import streamlit as st

def extract_bands_from_image(uploaded_file):
    """
    Extract 11 satellite bands from uploaded image
    Supports TIFF, PNG, JPG formats
    """
    try:
        # Read image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        st.write(f"📊 Image shape: {img_array.shape}")
        
        # If RGB image, simulate multi-spectral bands
        if len(img_array.shape) == 2:  # Grayscale
            # Single band - replicate to create 11 bands
            bands = np.tile(img_array, (11, 1, 1))
        elif img_array.shape[2] == 3:  # RGB image
            # Convert RGB to simulated Sentinel-2 bands
            r = img_array[:, :, 0].astype(float)
            g = img_array[:, :, 1].astype(float)
            b = img_array[:, :, 2].astype(float)
            
            # Create 11 bands (Sentinel-2 simulation)
            # B2 (Blue), B3 (Green), B4 (Red), B5, B6, B7, B8 (NIR), B8A, B11, B12, SCL
            bands = np.zeros((11, img_array.shape[0], img_array.shape[1]))
            
            bands[0] = b * 100 + np.random.normal(0, 5, b.shape)  # B2 (Blue)
            bands[1] = g * 100 + np.random.normal(0, 5, g.shape)  # B3 (Green)
            bands[2] = r * 100 + np.random.normal(0, 5, r.shape)  # B4 (Red)
            bands[3] = (r + g) * 50  # B5
            bands[4] = (r + g) * 60  # B6
            bands[5] = (r + g) * 70  # B7
            bands[6] = (r + g + b) * 80  # B8 (NIR)
            bands[7] = (r + g + b) * 75  # B8A
            bands[8] = (r + g) * 120  # B11 (SWIR)
            bands[9] = (r + g) * 100  # B12 (SWIR)
            bands[10] = r * 0.5  # Simulated Biomass
            
        elif img_array.shape[2] >= 11:  # Multi-spectral image (already has bands)
            bands = img_array[:, :, :11]
        else:
            st.error("❌ Image must be RGB or multi-spectral!")
            return None
        
        return bands
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        return None

def calculate_indices_from_bands(bands):
    """
    Calculate vegetation indices from satellite bands
    Returns: NDVI, EVI, SAVI, NBR and other features
    """
    try:
        # Extract key bands (normalized)
        b2 = np.mean(bands[0]) / 2550 if len(bands) > 0 else 0.5  # Blue
        b3 = np.mean(bands[1]) / 2550 if len(bands) > 1 else 0.5  # Green
        b4 = np.mean(bands[2]) / 2550 if len(bands) > 2 else 0.5  # Red
        b8 = np.mean(bands[6]) / 2550 if len(bands) > 6 else 0.8  # NIR
        b11 = np.mean(bands[8]) / 2550 if len(bands) > 8 else 0.7  # SWIR1
        b12 = np.mean(bands[9]) / 2550 if len(bands) > 9 else 0.6  # SWIR2
        
        # Calculate indices (add small epsilon to avoid division by zero)
        eps = 1e-6
        
        # NDVI: (NIR - Red) / (NIR + Red)
        ndvi = (b8 - b4) / (b8 + b4 + eps)
        ndvi = max(0, min(1, ndvi))  # Clip to [0, 1]
        
        # EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        evi = 2.5 * (b8 - b4) / (b8 + 6*b4 - 7.5*b2 + 1 + eps)
        evi = max(0.5, min(3, evi))  # Clip to [0.5, 3]
        
        # SAVI: 1.5 * (NIR - Red) / (NIR + Red + 0.5)
        savi = 1.5 * (b8 - b4) / (b8 + b4 + 0.5 + eps)
        savi = max(0.2, min(1.5, savi))  # Clip to [0.2, 1.5]
        
        # NBR: (NIR - SWIR) / (NIR + SWIR)
        nbr = (b8 - b11) / (b8 + b11 + eps)
        nbr = max(-0.1, min(0.7, nbr))  # Clip to [-0.1, 0.7]
        
        # Simulate biomass from indices
        biomass = (ndvi * 100 + evi * 20) + np.random.uniform(5, 50)
        biomass = max(0, min(250, biomass))
        
        # Convert reflectances to 0-1000 range for bands
        b2_val = b2 * 1000
        b3_val = b3 * 1000
        b4_val = b4 * 1000
        b8_val = b8 * 1000
        b11_val = b11 * 1000
        b12_val = b12 * 1000
        
        return {
            'NDVI': ndvi,
            'EVI': evi,
            'SAVI': savi,
            'NBR': nbr,
            'B2': b2_val,
            'B3': b3_val,
            'B4': b4_val,
            'B8': b8_val,
            'B11': b11_val,
            'B12': b12_val,
            'Biomass': biomass
        }
    
    except Exception as e:
        st.error(f"❌ Error calculating indices: {str(e)}")
        return None