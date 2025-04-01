import numpy as np

def fast_block_correlation(img1: np.ndarray, img2: np.ndarray, block_size: int = 16) -> np.ndarray:
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
                          j * block_size : (j + 1) * block_size]
            block2 = img2[i * block_size : (i + 1) * block_size,
                          j * block_size : (j + 1) * block_size]
            
            # Flatten the blocks and compute the Pearson correlation using np.corrcoef
            block_correlations[i, j] = np.corrcoef(block1.ravel(), block2.ravel())[0, 1]
    
    return block_correlations
