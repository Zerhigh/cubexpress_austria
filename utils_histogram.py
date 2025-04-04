import numpy as np

def fast_block_correlation(img1: np.ndarray, img2: np.ndarray, none_value: float = 0, block_size: int = 16) -> np.ndarray:
    """
    Compute the Pearson correlation coefficient for non-overlapping blocks of two images.
    
    This function divides the input images into non-overlapping blocks of size `block_size`.
    For each corresponding pair of blocks, it calculates the Pearson correlation coefficient.
    If the arrays are 3D (e.g., multiple channels), the computation is performed separately for each channel.
    
    Args:
        img1 (np.ndarray): The first image array.
        img2 (np.ndarray): The second image array.
        block_size (int, optional): The size of each non-overlapping block. Defaults to 16.
    
    Returns:
        np.ndarray: A 2D array of correlation coefficients. If the input arrays are 3D,
            it returns a 3D array where each slice corresponds to one channel's correlation map.
    
    Raises:
        AssertionError: If `img1` and `img2` do not share the same shape.
    """
    # Ensure both images have identical shapes
    assert img1.shape == img2.shape, "Images must have the same shape"
    
    # If the images have multiple channels, compute the correlation per channel
    if img1.ndim == 3:
        correlations = []
        for channel_index in range(img1.shape[0]):
            correlations.append(
                fast_block_correlation(img1[channel_index], img2[channel_index], block_size)
            )
        return np.array(correlations)
    
    # For 2D images, get dimensions and compute number of blocks
    height, width = img1.shape
    h_blocks = height // block_size
    w_blocks = width // block_size
    
    # Initialize an array to store correlation coefficients for each block
    block_correlations = np.zeros((h_blocks, w_blocks))
    
    # Calculate the correlation coefficient for each non-overlapping block
    for i in range(h_blocks):
        for j in range(w_blocks):
            block1 = img1[i * block_size : (i + 1) * block_size,
                          j * block_size : (j + 1) * block_size].ravel()
            block2 = img2[i * block_size : (i + 1) * block_size,
                          j * block_size : (j + 1) * block_size].ravel()

            valid_mask = (block1 != none_value) & (block2 != none_value)
            valid_band1 = block1[valid_mask]
            valid_band2 = block2[valid_mask]

            if valid_band1.size == 0 or valid_band2.size == 0:
                block_correlations[i, j] = np.nan
            else:
                # Flatten the blocks and compute the Pearson correlation using np.corrcoef
                block_correlations[i, j] = np.corrcoef(valid_band1, valid_band2)[0, 1]
    
    return block_correlations


def base_fast_block_correlation(img1: np.ndarray, img2: np.ndarray, none_value: float = 0,
                           block_size: int = 16) -> np.ndarray:
    """
    Compute the Pearson correlation coefficient for non-overlapping blocks of two images.

    This function divides the input images into non-overlapping blocks of size `block_size`.
    For each corresponding pair of blocks, it calculates the Pearson correlation coefficient.
    If the arrays are 3D (e.g., multiple channels), the computation is performed separately for each channel.

    Args:
        img1 (np.ndarray): The first image array.
        img2 (np.ndarray): The second image array.
        block_size (int, optional): The size of each non-overlapping block. Defaults to 16.

    Returns:
        np.ndarray: A 2D array of correlation coefficients. If the input arrays are 3D,
            it returns a 3D array where each slice corresponds to one channel's correlation map.

    Raises:
        AssertionError: If `img1` and `img2` do not share the same shape.
    """
    # Ensure both images have identical shapes
    assert img1.shape == img2.shape, "Images must have the same shape"

    # If the images have multiple channels, compute the correlation per channel
    if img1.ndim == 3:
        correlations = []
        for channel_index in range(img1.shape[0]):
            correlations.append(
                base_fast_block_correlation(img1[channel_index], img2[channel_index], block_size)
            )
        return np.array(correlations)

    # For 2D images, get dimensions and compute number of blocks
    height, width = img1.shape
    h_blocks = height // block_size
    w_blocks = width // block_size

    # Initialize an array to store correlation coefficients for each block
    block_correlations = np.zeros((h_blocks, w_blocks))

    # Calculate the correlation coefficient for each non-overlapping block
    for i in range(h_blocks):
        for j in range(w_blocks):
            block1 = img1[i * block_size: (i + 1) * block_size,
                     j * block_size: (j + 1) * block_size].ravel()
            block2 = img2[i * block_size: (i + 1) * block_size,
                     j * block_size: (j + 1) * block_size].ravel()

            # Flatten the blocks and compute the Pearson correlation using np.corrcoef
            block_correlations[i, j] = np.corrcoef(block1, block2)[0, 1]

    return block_correlations


def own_bandwise_correlation(img1, img2, none_value=0):
    """
    Compute Pearson correlation per band between two multi-band images,
    while handling no-data values.

    Args:
        img1, img2 (np.ndarray): Arrays of shape (bands, height, width).
        nodata_value (int or float): No-data value to be ignored.

    Returns:
        np.ndarray: Correlation coefficient for each band.
    """
    assert img1.shape == img2.shape, "Input images must have the same shape"

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

    return correlations
