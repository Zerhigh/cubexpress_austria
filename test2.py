import numpy as np

array1 = [np.nan, 0.85, np.nan, 0.67, 0.23, 0.91, np.nan, 0.44, 0.76, 0.33]
array2 = [np.nan, 0.52, 0.88, 0.19, np.nan, 0.74, 0.65, 0.31, 0.98, 0.41]

res = np.corrcoef(array1, array1)[0, 1]


# Set random seed for reproducibility
np.random.seed(42)

# Create random images with values between 0 and 1
img1 = np.random.rand(4, 10, 10)
img2 = np.random.rand(4, 10, 10)

# Introduce NaN values (set some pixels to 0 as nodata)
nan_indices = np.random.choice(100, 10, replace=False)  # 10 random NaNs per band

for b in range(4):
    img1[b].flat[nan_indices] = 0  # Set NaN as 0 in img1
    img2[b].flat[nan_indices] = 0  # Set NaN as 0 in img2

img2=img1.copy()
img2[0, 0, 0] = 1

# my method is valid: the correlation of these images is 1 if they are the same
# its 0.976 if only one value is different
#its 0.975 if one extra cell is attributed as 0==nodata, this means there are less values in total
# which reduces the amount of possible correlations and the images are less correlated
# => no data values are excluded from the correlation calculation
img2[0, 0, 1] = 0
img1[0, 0, 1] = 0
# Define nodata value
none_value = 0

bands, height, width = img1.shape
correlations = np.full(bands, np.nan)  # Initialize with NaN

for b in range(bands):
    band1 = img1[b].ravel()
    band2 = img2[b].ravel()

    # Mask no-data values (convert them to NaN)
    valid_mask = (band1 != none_value) & (band2 != none_value)
    valid_band1 = band1[valid_mask]
    valid_band2 = band2[valid_mask]

    # If no valid data left, return NaN
    if valid_band1.size == 0 or valid_band2.size == 0:
        correlations[b] = np.nan
    else:
        correlations[b] = np.corrcoef(valid_band1, valid_band2)[0, 1]

pass