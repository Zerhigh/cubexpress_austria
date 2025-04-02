import numpy as np


new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,  # np.float128 ; doesn't exist on windows
    'G': np.complex128,  # np.complex256 ; doesn't exist on windows
}


def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : np.dtype or tuple of np.dtype
        The input dtype. If a tuple of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, tuple):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def _match_cumulative_cdf(source, template, ignore_none=False, none_value=0):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    # remove nones
    s2nans = np.count_nonzero(template == 0)
    hrnans = np.count_nonzero(source == 0)

    src_size, tmpl_size = source.size, template.size

    src_lookup = source.reshape(-1)
    template_lookup = template.reshape(-1)

    if source.dtype.kind == 'u':
        src_counts = np.bincount(src_lookup)
        tmpl_counts = np.bincount(template_lookup)

        # omit values where the count was 0
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        src_values, src_lookup, src_counts = np.unique(
            src_lookup, return_inverse=True, return_counts=True
        )
        tmpl_values, tmpl_counts = np.unique(template_lookup, return_counts=True)

        # modify counts and remove the 0 values to change impact of noData values
        if ignore_none:
            try:
                # index of nan value in src_values
                src_nan_index = np.where(src_values == none_value)[0]
                tmpl_nan_index = np.where(src_values == none_value)[0]

                # reduce amount of nans in the template (s2) by the amount of nans in the masked HR Orthofoto
                # this maintains the naTural s2 nones while Removing the impact of those introduced by the binary masking
                #nan_count = int(src_counts[src_nan_index][0] / 16)
                tmpl_counts[tmpl_nan_index] = 0

                # set thE number of nans in the Orthofoto to 0, this will 'neglect' natural nans in the orthofoto, but
                # these are already mapped onto the s2 so their impact is also reduced from the s2
                src_counts[src_nan_index] = 0

            except Exception as e:
                print(e)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / src_size
    tmpl_quantiles = np.cumsum(tmpl_counts) / tmpl_size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_lookup].reshape(source.shape)


def real_match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    if source.dtype.kind == 'u':
        src_lookup = source.reshape(-1)
        src_counts = np.bincount(src_lookup)
        tmpl_counts = np.bincount(template.reshape(-1))

        # omit values where the count was 0
        tmpl_values = np.nonzero(tmpl_counts)[0]
        tmpl_counts = tmpl_counts[tmpl_values]
    else:
        src_values, src_lookup, src_counts = np.unique(
            source.reshape(-1), return_inverse=True, return_counts=True
        )
        tmpl_values, tmpl_counts = np.unique(template.reshape(-1), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_lookup].reshape(source.shape)


def match_histograms(image, reference, *, channel_axis=None, ignore_none=False, none_value=0):
    """Adjust an image so that its cumulative histogram matches that of another.

    The adjustment is applied separately for each channel.

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    matched : ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.

    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/

    """

    # modified to work with channel x width x heigth

    if image.ndim != reference.ndim:
        raise ValueError(
            'Image and reference must have the same number ' 'of channels.'
        )

    if channel_axis is not None:
        if image.shape[0] != reference.shape[0]:
            raise ValueError(
                'Number of channels in the input image and '
                'reference image must match!'
            )

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[0]):
            matched_channel = _match_cumulative_cdf(
                image[channel], reference[channel], ignore_none=ignore_none, none_value=none_value
            )
            matched[channel] = matched_channel
    else:
        # _match_cumulative_cdf will always return float64 due to np.interp
        matched = _match_cumulative_cdf(image, reference, ignore_none=ignore_none, none_value=none_value)

    if matched.dtype.kind == 'f':
        # output a float32 result when the input is float16 or float32
        out_dtype = _supported_float_type(image.dtype)
        matched = matched.astype(out_dtype, copy=False)
    return matched


def real_match_histograms(image, reference, *, channel_axis=None):
    """Adjust an image so that its cumulative histogram matches that of another.

    The adjustment is applied separately for each channel.

    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

    Returns
    -------
    matched : ndarray
        Transformed input image.

    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.

    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/

    """

    # modified to work with channel x width x heigth

    if image.ndim != reference.ndim:
        raise ValueError(
            'Image and reference must have the same number ' 'of channels.'
        )

    if channel_axis is not None:
        if image.shape[0] != reference.shape[0]:
            raise ValueError(
                'Number of channels in the input image and '
                'reference image must match!'
            )

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[0]):
            matched_channel = real_match_cumulative_cdf(
                image[channel], reference[channel]
            )
            matched[channel] = matched_channel
    else:
        # _match_cumulative_cdf will always return float64 due to np.interp
        matched = real_match_cumulative_cdf(image, reference)

    if matched.dtype.kind == 'f':
        # output a float32 result when the input is float16 or float32
        out_dtype = _supported_float_type(image.dtype)
        matched = matched.astype(out_dtype, copy=False)
    return matched


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
                          j * block_size : (j + 1) * block_size]
            block2 = img2[i * block_size : (i + 1) * block_size,
                          j * block_size : (j + 1) * block_size]
            
            # Flatten the blocks and compute the Pearson correlation using np.corrcoef
            block_correlations[i, j] = np.corrcoef(block1.ravel(), block2.ravel())[0, 1]
    
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
