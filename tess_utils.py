#!/usr/bin/env python3
""" 16 Nov 2023, author Oryna Ivashtenko"""

import numpy as np
import copy
from scipy.optimize import minimize
from scipy.signal.windows import tukey


def unb_var(arr):
    """
    Calculates the unbiased variance estimate (numpy.var is biased!).
    Is related to the biased estimation as
    var_u = N/(N-1) * var_b,
    where N is the size of the sample.
    Parameters
    ----------
    arr : numpy.array 1d or list of float, may contain numpy.nan
        Sample of numbers to calculate their variance

    Returns
    ---------
    float
        Unbiased variance estimate of arr
    """
    arr = np.array(arr[~np.isnan(arr)])
    return 1 / (len(arr) - 1) * np.sum(np.abs(arr - np.mean(arr)) ** 2)



def get_dt(tim):
    """determining time step"""
    dts = (tim - np.roll(tim, -1))[:-1]
    sigma_dts = np.nanstd(dts)
    dts = dts[np.abs(dts - np.nanmedian(dts)) < sigma_dts * 0.1]
    dt = np.abs(np.nanmean(dts))
    return dt


def fill_holes_time(time):
    """assuming uniform time step, fill missed time steps"""
    dt = get_dt(time)
    min_time = time[0]
    time = time - min_time
    len_new = int(np.round((np.max(time) - np.min(time)) / dt)) + 1
    mask_filling = np.zeros(len_new).astype(bool)
    for t in time:
        mask_filling[int(np.round(t / dt))] = True
    return np.arange(len(mask_filling)) * dt + min_time, mask_filling


def mask_outliers(data, mask, n_sigma=3, n_iterations=3, detrend=True, detr_length=50):
    mask_1 = np.array(mask)
    data_1 = np.array(data)
    if detrend:
        data_1 = detrend_mov_ave(data_1, length=detr_length, mask=mask_1, return_trend=False, keep_overall_level=False,
                                    remove_outliers=False)
    for _ in range(n_iterations):
        outlier_mask = np.abs(data_1 - np.nanmean(data_1[mask_1])) < n_sigma * np.nanstd(data_1[mask_1])
        mask_1 = mask_1 & outlier_mask
    return mask_1


def measure_psd(data, mask, window='hann', nperseg=64):
    noverlap = nperseg // 2
    windowdict = {'boxcar': np.ones(nperseg) / np.linalg.norm(np.ones(nperseg)),
                  'hann': 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg)) / np.linalg.norm(
                      0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / nperseg)))}
    psds = [np.abs(np.fft.rfft(windowdict[window] * data[ind1:ind1 + nperseg])
                   ) ** 2 for ind1 in range(0, len(data) - nperseg, nperseg - noverlap
                                            ) if np.sum((~mask)[ind1:ind1 + nperseg]) == 0]
    return psds


def moving_average(data, mask, detr_len, gap_len=0):
    d = detr_len // 2
    if gap_len == 0:
        return [np.nanmean(data[max(0, i - d): min(i + d, len(data))][mask[max(0, i - d): min(i + d, len(data))]]) for i in
                range(len(data))]
    g = gap_len // 2
    return [np.nanmean(np.concatenate((data[max(0, i - d - g): i - g], data[i + g: min(i + d + g, len(data))]))[
                           np.concatenate((mask[max(0, i - d - g): i - g], mask[i + g: min(i + d + g, len(data))]))]) for i in
            range(len(data))]


def moving_std(data, mask, detr_len, gap_len=0):
    d = detr_len // 2
    if gap_len == 0:
        return [np.nanstd(data[max(0, i - d): min(i + d, len(data))][mask[max(0, i - d): min(i + d, len(data))]])
                if len(np.where(mask[max(0, i - d): min(i + d, len(data))])[0]) > 1 else np.inf for i in range(len(data))]
    g = gap_len // 2
    return [np.nanstd(np.concatenate((data[max(0, i - d - g): i - g], data[i + g: min(i + d + g, len(data))]))[
                          np.concatenate((mask[max(0, i - d - g): i - g], mask[i + g: min(i + d + g, len(data))]))])
            if len(np.where(np.concatenate((mask[max(0, i - d - g): i - g], mask[i + g: min(i + d + g, len(data))])))[0]) > 1
            else np.inf for i in range(len(data))]


def outlier_free_mean_array(array, nsig=4):
    axis = 0
    means = np.mean(array, axis=axis)
    stds = np.std(array, axis=axis)
    masks = np.abs(array - means) / stds < nsig
    means = np.array([np.mean(array[:, i][masks[:, i]]) for i in range(array.shape[1])])
    for _ in range(4):
        stds = np.array([np.std(array[:, i][masks[:, i]]) for i in range(array.shape[1])])
        masks = np.abs(array - means) / stds < nsig
        means = np.array([np.mean(array[:, i][masks[:, i]]) for i in range(array.shape[1])])
    return means


def clean_small_holes(mask, max_hole=3):
    """
    Deletes from the mask holes of size <=max_small_hole,
    replacing False with True.

    Parameters
    ----------
    mask : numpy.array 1d of bool
        True is good data, False is bad data
    max_hole : int
        Maximum size of hole which is considered to be small

    Returns
    -------
    mask : np.array 1d of bool
        Mask in which small holes are removed, i.e. made True
    """
    mask = np.array(mask).astype(bool)
    ind1 = (np.where(mask)[0])[0]
    ind2 = (np.where(mask)[0])[-1]
    mask1 = np.concatenate((np.ones(ind1), np.array(mask[ind1:ind2 + 1]), np.ones(len(mask) - ind2 - 1))).astype(bool)
    begholes = np.where(mask1 * np.roll(~mask1, -1))[0] + 1
    endholes = np.where(mask1 * np.roll(~mask1, 1))[0]
    lenholes = endholes - begholes
    small_holes = np.where(lenholes <= max_hole)[0]
    for i in range(len(small_holes)):
        mask[begholes[small_holes[i]]:endholes[small_holes[i]]] = True  # mask with only big holes left
    return mask


def clean_small_non_holes(mask, max_non_hole=3):
    """
    Deletes from the mask non-holes of size <=max_small_hole,
    replacing True with False.

    Parameters
    ----------
    mask : numpy.array 1d of bool
        True is good data, False is bad data
    max_non_hole : int
        Maximum size of non-hole which is considered to be small

    Returns
    -------
    mask : np.array 1d of bool
        Mask in which small non-holes are removed, i.e. made False
    """
    mask = np.array(mask).astype(bool)
    mask_1 = clean_small_holes(mask, max_hole=max_non_hole)
    mask_1_inv = ~mask_1
    mask_1_inv = clean_small_holes(mask_1_inv, max_hole=max_non_hole)
    mask_1 = ~mask_1_inv
    return mask & mask_1


def detrend_mov_ave(data, length=128, mask=None, method='median', endmethod='mean', ends='symmetric', endlen=None,
                    return_trend=True, keep_overall_level=False, remove_outliers=False, nsig_outliers=4,
                    detr_length_outliers=128, remove_outliers_light=False):
    """
    Amateur detrending function. Subtracting the moving mean or median from the data.

    Parameters
    --------------------------
    data : np.array 1d or list. Data sample from which the moving average has to be subtracted
    length : int. The length of sample to calculate moving average. At a given point,
                    half length to the left and to the right are taken.
    method : str. Kind of moving value to subtract: 'median' or 'mean'
                    The default is 'median', because it is less sensitive to outliers.
    endmethod: str. Kind of moving value to subtract at the ends of the data (see below): 'median' or 'mean'.
                    The default is 'mean', because it is less sensitive to small
                    amount of data samples available in the ends.
    ends : str. How to treat the points closer than length/2 to the ends of the data:
                'symmetric' :  endlen is length/4
                               1) if point<endlen, take endlen samples from the right, and up to zero from the left
                               2) if endlen<point<half length, takethe same number of points from left and right
                               3) if point>half length, take usual half length from both sides
                               for the second end, the same in opposite direction
                'asymmetric' : endlen is length/16
                               1) if point<half length, take up to zero from the left, and the length to the
                                    right linearly increases from endlen for point=0 to length/2 at point=length/2
                               2) if point>half length, take usual half length from both sides
                               for the second end, the same in opposite direction
                'cut' : not to calculate the result at this point. The size of the output will be len(data)-length
    endlen : int, default is None. Minimal length to take from the end for determining the average in the
        first (last) point. If None, will be assigned depending on the method (see description of ends)
    return_trend : bool. Whether to return the trend in addition to the detrended data
    keep_overall_level : bool. If true, the total mean or median will be added to the data (that is preserved)

    Returns
    --------------------------
    if not return_trend: data with subtracted mean, np.array
    if return_trend: data with subtracted mean, np.array; trend (moving mean), np.array
    """
    assert ((method == 'mean') or (method == 'median')), 'Incorrect method specified'
    data = np.array(data)

    if mask is None:
        mask = ~np.isnan(data)

    mask = np.array(mask)
    assert len(data) == len(mask), 'Data and mask must be of the same length!'

    if remove_outliers:
        data1 = detrend_mov_ave(np.array(data), length=detr_length_outliers, mask=mask, method='median',
                                endmethod='mean',
                                ends='symmetric', endlen=None, return_trend=False, keep_overall_level=False,
                                remove_outliers=False)
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask

    elif remove_outliers_light:
        outlier_mask = np.abs(data - np.nanmean(data[mask])) < nsig_outliers * np.nanstd(data[mask])
        mask *= outlier_mask

    if ends == 'asymmetric':
        trend = []
        res = []
        if endlen is None:
            l4 = int(np.ceil(length / 16))
        elif endlen > int(length / 2):
            l4 = int(length / 2)
        else:
            l4 = endlen
        l2 = int(np.ceil(length / 2))
        if endmethod == 'median':
            for i in np.arange(0, l2):
                if (mask[0:i + int(l4 + (l2 - l4) / l2 * i)]).any():
                    m = np.nanmedian((data[0:i + int(l4 + (l2 - l4) / l2 * i)]
                                     )[mask[0:i + int(l4 + (l2 - l4) / l2 * i)]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[0:i + int(l4 + (l2 - l4) / l2 * i)])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod == 'mean':
            for i in np.arange(0, l2):
                if (mask[0:i + int(l4 + (l2 - l4) / l2 * i)]).any():
                    m = np.nanmean((data[0:i + int(l4 + (l2 - l4) / l2 * i)]
                                   )[mask[0:i + int(l4 + (l2 - l4) / l2 * i)]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[0:i + int(l4 + (l2 - l4) / l2 * i)])
                    trend.append(m)
                    res.append(data[i] - m)
        if method == 'median':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmedian((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
        elif method == 'mean':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmean((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
        if endmethod == 'median':
            for i in np.arange(len(data) - l2, len(data)):
                if (mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]).any():
                    m = np.nanmedian((data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]
                                     )[mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod == 'mean':
            for i in np.arange(len(data) - l2, len(data)):
                if (mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]).any():
                    m = np.nanmean((data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]
                                    )[mask[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i - int(l4 + (l2 - l4) / l2 * (len(data) - i)):])
                    trend.append(m)
                    res.append(data[i] - m)
    elif ends == 'symmetric':
        trend = []
        res = []
        if endlen is None:
            l4 = int(np.ceil(length / 4))
        elif endlen > int(length / 2):
            l4 = int(length / 2)
        else:
            l4 = endlen
        l2 = int(np.ceil(length / 2))
        if endmethod == 'median':
            for i in np.arange(0, l4):
                if (mask[0:i + l4]).any():
                    m = np.nanmedian((data[0:i + l4])[mask[0:i + l4]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[0:i + l4])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(l4, l2):
                if (mask[0:2 * i]).any():
                    m = np.nanmedian((data[0:2 * i])[mask[0:2 * i]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[0:2 * i])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod == 'mean':
            for i in np.arange(0, l4):
                if (mask[0:i + l4]).any():
                    m = np.nanmean((data[0:i + l4])[mask[0:i + l4]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[0:i + l4])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(l4, l2):
                if (mask[0:2 * i]).any():
                    m = np.nanmean((data[0:2 * i])[mask[0:2 * i]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[0:2 * i])
                    trend.append(m)
                    res.append(data[i] - m)
        if method == 'median':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmedian((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    # trend.append(np.nan)
                    # res.append(np.nan)
                    m = np.nanmedian(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
        elif method == 'mean':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmean((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
        if endmethod == 'median':
            for i in np.arange(len(data) - l2, len(data) - l4):
                if (mask[2 * i - len(data):]).any():
                    m = np.nanmedian((data[2 * i - len(data):])[mask[2 * i - len(data):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[2 * i - len(data):])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(len(data) - l4, len(data)):
                if (mask[i - l4:]).any():
                    m = np.nanmedian((data[i - l4:])[mask[i - l4:]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i - l4:])
                    trend.append(m)
                    res.append(data[i] - m)
        elif endmethod == 'mean':
            for i in np.arange(len(data) - l2, len(data) - l4):
                if (mask[2 * i - len(data):]).any():
                    m = np.nanmean((data[2 * i - len(data):])[mask[2 * i - len(data):]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[2 * i - len(data):])
                    trend.append(m)
                    res.append(data[i] - m)
            for i in np.arange(len(data) - l4, len(data)):
                if (mask[i - l4:]).any():
                    m = np.nanmean((data[i - l4:])[mask[i - l4:]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i - l4:])
                    trend.append(m)
                    res.append(data[i] - m)
    elif ends == 'cut':
        trend = []
        res = []
        l2 = int(np.ceil(length / 2))
        if method == 'median':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmedian((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmedian(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
        if method == 'mean':
            for i in np.arange(l2, len(data) - l2):
                if (mask[i - l2:i + l2]).any():
                    m = np.nanmean((data[i - l2:i + l2])[mask[i - l2:i + l2]])
                    trend.append(m)
                    res.append(data[i] - m)
                else:
                    m = np.nanmean(data[i - l2:i + l2])
                    trend.append(m)
                    res.append(data[i] - m)
    else:
        raise AssertionError('Incorrect way to treat ends specified!')
    res = np.array(res)
    if keep_overall_level:
        if method == 'median':
            res = res - np.nanmedian(res) + np.nanmedian(data[mask])
        if method == 'mean':
            res = res - np.nanmean(res) + np.nanmean(data[mask])
    if return_trend:
        return res, trend
    else:
        return res


def interpolate_over_mask(flux, mask=None, interp_method='linear', ends='continue', mitigate_hole_ends=False,
                          endhole_method='median', mit_length=10):
    """
    Takes the data and the mask of holes, interpolates the holes. If the mask is not
    provided, finds numpy.nan in the data and interpolates them.

    Parameters
    ----------
    flux : np.array 1d of floats
        data to be interpolated, may contain nans
    mask : np.array 1d of bool
        False entries mean there is a hole to be interpolated. No matter what
        is the value of data at this point.
    interp_method : str : 'mean' or 'median' or 'iterative_fourier' or ' linear' (default).
        How to fill the holes. By mean of the data, by median of the data,
        by doing Fourier transform many times, cuting high frequencies,
        going back and substituting to the holes,
        or by linearly interpolating over the holes.
    ends : str : 'continue' or 'zeros' or 'keep'.
        What to do to the ends of the data in linear method.
        'continue' is to take the first(last) value and fill by it all the rest.
        'zeros' meens to replace by zeros.
        'keep' means to leave what was in the original data (maybe nans)
    mitigate_hole_ends : bool
        If true, for the linear, mean and median interpolation, not the ends of the holes will be taken,
        but the average of points near the hole
    endhole_method : str : 'mean' or 'median'(default)
        The type of average for mitigate_hole_ends.
    mit_length : int. Default is 10
        The length of the slice for the averaging in the mitigate_hole_ends
    Returns
    -------
    numpy.array 1d of float
        Interpolated flux
    """
    if mask is None:
        mask = ~np.isnan(flux)
    else:
        mask = np.array(mask).astype(bool)

    ind1 = None
    ind2 = None
    if interp_method == 'mean' or interp_method == 'median':
        ind1 = (np.where(mask)[0])[0]
        ind2 = (np.where(mask)[0])[-1]
        # if there is a hole in the beginning or end, it cannot be interpolated
        mask1 = np.array(mask[ind1:ind2 + 1])
        begholes = np.where(mask1 * np.roll(~mask1, -1))[0] + 1
        endholes = np.where(mask1 * np.roll(~mask1, 1))[0]
        lenholes = endholes - begholes
        kernel_size = np.max(lenholes) + 4
        indholes = np.where(~mask1)[0]
        flux[indholes] = nan_average_filt(flux, kernel_size=kernel_size, method=interp_method)[indholes]
    elif interp_method == 'iterative_fourier':
        nrep = 6
        flux1 = np.array(flux)
        indholes = np.where(~ mask)[0]
        flux1[indholes] = 0
        for i in range(nrep):
            fflux1 = np.fft.fft(flux1)
            fflux1[len(flux1) // 10 + 4 ** i: -(len(flux1) // 10 + 4 ** i)] = 0
            flux1[indholes] = (np.fft.ifft(fflux1))[indholes]
        flux = flux1
    elif interp_method == 'linear':
        ind1 = (np.where(mask)[0])[0]
        ind2 = (np.where(mask)[0])[-1]
        # if there is a hole in the beginning or end, it cannot be interpolated
        flux1 = np.array(flux[ind1:ind2 + 1])
        mask1 = np.array(mask[ind1:ind2 + 1])
        begholes = np.where(mask1 * np.roll(~mask1, -1))[0]
        endholes = np.where(mask1 * np.roll(~mask1, 1))[0]
        if not mitigate_hole_ends:
            for j in range(len(begholes)):
                flux1[begholes[j] + 1:endholes[j]] = np.interp(
                    np.arange(begholes[j], endholes[j] + 1), np.array([begholes[j], endholes[j]]),
                    np.array([flux1[begholes[j]], flux1[endholes[j]]]))[1:-1]
        else:
            if endhole_method == 'median':
                for j in range(len(begholes)):
                    i1 = max(begholes[j] - mit_length + 1, 0)
                    i2 = min(endholes[j] + mit_length, len(flux1))
                    flux1[begholes[j] + 1:endholes[j]] = np.interp(
                        np.arange(begholes[j], endholes[j] + 1), np.array([begholes[j], endholes[j]]),
                        np.array([np.nanmedian((flux1[i1:begholes[j] + 1])[np.where(mask1[i1:begholes[j] + 1])[0]]),
                                  np.nanmedian((flux1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]])]))[1:-1]
            elif endhole_method == 'mean':
                for j in range(len(begholes)):
                    i1 = max(begholes[j] - mit_length + 1, 0)
                    i2 = min(endholes[j] + mit_length, len(flux1))
                    flux1[begholes[j] + 1:endholes[j]] = np.interp(
                        np.arange(begholes[j], endholes[j] + 1), np.array([begholes[j], endholes[j]]),
                        np.array([np.nanmean((flux1[i1:begholes[j] + 1])[np.where(mask1[i1:begholes[j] + 1])[0]]),
                                  np.nanmean((flux1[endholes[j]:i2])[np.where(mask1[endholes[j]:i2])[0]])]))[1:-1]
            else:
                raise AssertionError('Incorrect endhole_method specified!')
        flux[ind1:ind2 + 1] = flux1
    if interp_method == 'linear' or interp_method == 'mean' or interp_method == 'median':
        if ends == 'continue':
            if ind1 != 0:
                flux[0:ind1] = flux[ind1]
            if ind2 != len(flux) - 1:
                flux[ind2:] = flux[ind2]
        elif ends == 'zeros':
            if ind1 != 0:
                flux[0:ind1] = 0
            if ind2 != len(flux) - 1:
                flux[ind2:] = 0
        elif ends == 'keep':
            pass
        else:
            raise AssertionError('Unlefined ends parameter specified!')
    return flux


def psd2wf(psd, support=None, rfft=True, tukey_steps_len=0,
           conditional_zero_zeroth_freq=True, always_zero_zeroth_freq=False):
    """
    Generates a whitening filter of the needed length (support) from the PSD.
    Makes sinc-extrapolation, applies tuckey window to the end and to the low frequencies if they drop too quickly
    :param psd: measured PSD in coarse resolution
    :type psd: ndarray
    :param support: the desired length of the extrapolated filter (time domain). If None, take the length corresponding to the psd
    :type support: int
    :param rfft: whether to use rfft (or just fft). If true, the signal is treated as real,
                 and all ffts are rffts. The length of the frequency samples are the nperseg//2+1.
    :type rfft: bool
    :param tukey_steps_len: length taken for zeroing in the tukey windowing
    :type tukey_steps_len: int or float
    :param conditional_zero_zeroth_freq: whether to put the low frequencies to zero if they deviate too much
    :type conditional_zero_zeroth_freq: bool
    :param always_zero_zeroth_freq: whether to always put the lowest frequencies to zero
    :type always_zero_zeroth_freq: bool
    :return: wf_td : whitening filter in time domain; wf_fd : whitening filter in Fourier domain.
    :rtype: tuple(ndarray, ndarray)
    """
    wf_fd_meas = 1 / np.array(psd)**0.5
    was_vector = False
    if not hasattr(wf_fd_meas[0], '__len__'):
        wf_fd_meas = np.array([wf_fd_meas])
        was_vector = True
    if rfft:
        ifftfun = np.fft.irfft
        dtype_res = np.float64
        if support is None:
            support = 2 * (len(wf_fd_meas) - 1)
    else:
        if support == 'none':
            support = len(wf_fd_meas)
        ifftfun = np.fft.ifft
        dtype_res = np.complex128

    assert support % 2 == 0, 'self.support assumed to be even!'
    wf_tds = []
    wf_fds = []
    for wpsd in wf_fd_meas:
        wf_td = ifftfun(wpsd)
        meas_len = len(wf_td)
        wf_td = np.fft.ifftshift(np.fft.fftshift(wf_td) * tukey(meas_len, alpha=tukey_steps_len / meas_len, sym=True))
        mid_point = len(wf_td) // 2
        result = np.zeros(support, dtype=dtype_res)
        result[:mid_point] = wf_td[:mid_point]
        result[-mid_point:] = wf_td[-mid_point:]

        wf_td = np.fft.fftshift(result)
        wf_fd = np.abs(np.fft.rfft(result))  # It is complex, which is weird, but what can we do
        if ((conditional_zero_zeroth_freq and (
                np.mean(np.abs(wf_fd[0:len(wf_fd) // 50])) / np.mean(np.abs(wf_fd[5 * len(wf_fd) // 10:7 * len(wf_fd) // 10])) < 0.3))
                or always_zero_zeroth_freq):
            first_measured_bin = int(np.ceil(support / meas_len))
            # wf_fd[:first_measured_bin] = 0.
            lentuk = 2 * first_measured_bin
            wf_fd[:lentuk // 2] = wf_fd[:lentuk // 2] * tukey(lentuk, alpha=3 * first_measured_bin / lentuk,
                                                              sym=True)[:lentuk // 2]
            wf_td = np.fft.fftshift(ifftfun(wf_fd))
        wf_tds.append(wf_td)
        wf_fds.append(wf_fd)

    if was_vector:
        wf_tds = wf_tds[0]
        wf_fds = wf_fds[0]

    return wf_tds, wf_fds


def moving_var(signal, mask=None, nwind=64, ngap=10,
               remove_outliers=False, detrend_remove_outliers=False,
               detr_length_outliers=128,
               nsig_outliers=4, return_mask=False, nsamples_min=20,
               return_total_var=False):
    """
    Calculates the moving variance of the signal. At every point, takes the window of length
    nwind symmetrically from the two sides, excludes from the middle the window of length
    ngap and calculates the unbiased variance of the remaining. It is needed to
    avoid the impact of the entries correlated with the considered point to the
    variance estimation. For example, if there is a transit in this point, the
    variance is taken around the transit, so its presence does not alter the measurement.
    At the ends of the singal, the data is taken only from one side of the
    considered point up to the available length.

    Parameters
    ----------
    signal : numpy.array 1d of float, NO NANS
        Data from which the moving variance is taken.
    nwind : int
        Length of the segment to calculate local variance for each bin
    ngap : int
        Length of the gap to exclude around each bin when calculating the
        variance in this bin in order to avoid influence of these
        points on the variance estimation.
    Returns
    -------
    mov_var : numpy.array 1d of float, len(signal)
        Moving variance estimate at each point of the signal
    """
    signal = np.array(signal)
    mov_var = None

    if mask is None:
        mask = ~np.isnan(signal)
    else:
        mask = np.array(mask)
        assert len(signal) == len(mask), 'Data and mask have to be of the same length!'

    if remove_outliers:
        if detrend_remove_outliers:
            data1 = detrend_mov_ave(np.array(signal), length=detr_length_outliers, mask=mask,
                                    method='median', endmethod='mean', ends='symmetric', endlen=None,
                                    return_trend=False, keep_overall_level=False, remove_outliers=True,
                                    nsig_outliers=nsig_outliers)
        else:
            data1 = signal
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask
        outlier_mask = np.abs(data1 - np.nanmean(data1[mask])) < nsig_outliers * np.nanstd(data1[mask])
        mask *= outlier_mask

    total_var = unb_var(signal[mask])

    all_done = False
    if (nwind > len(signal) / 2) or (nsamples_min * 4 > len(np.where(mask)[0])):
        mov_var = np.ones(len(signal)) * total_var
        all_done = True

    if not all_done:
        signal[~mask] = 0.
        sum1 = np.nancumsum(signal)
        sum1 = np.concatenate(([0], sum1))
        sum2 = np.nancumsum(signal ** 2)
        sum2 = np.concatenate(([0], sum2))
        summ = np.nancumsum(mask)
        summ = np.concatenate(([0], summ))
        mov_var = []

        hnw = nwind // 2
        hng = ngap // 2

        for i in np.arange(0, hng):
            counts = summ[i + hnw] - summ[i + hng]
            mov_var.append(((sum2[i + hnw] - sum2[i + hng]) / counts - (
                    (sum1[i + hnw] - sum1[i + hng]) / counts) ** 2) * counts / (counts - 1))
        for i in np.arange(hng, hnw):
            counts = summ[i - hng] - summ[0] + summ[i + hnw] - summ[i + hng]
            mov_var.append(((sum2[i - hng] - sum2[0] + sum2[i + hnw] - sum2[i + hng]) / counts
                            - ((sum1[i - hng] - sum1[0] + sum1[i + hnw] - sum1[i + hng]) / counts) ** 2) * counts / (counts - 1))
        for i in range(hnw, len(signal) - hnw):
            counts = summ[i + hnw] - summ[i + hng] + summ[i - hng] - summ[i - hnw]
            mov_var.append(((sum2[i + hnw] - sum2[i + hng] + sum2[i - hng] - sum2[i - hnw]) / counts
                            - (((sum1[i + hnw] - sum1[i + hng] + sum1[i - hng] - sum1[i - hnw]) / counts) ** 2)) * counts / (counts - 1))
        for i in np.arange(len(signal) - hnw, len(signal) - hng):
            counts = summ[i - hng] - summ[i - hnw] + summ[-1] - summ[i + hng]
            mov_var.append(((sum2[i - hng] - sum2[i - hnw] + sum2[-1] - sum2[i + hng]) / counts
                            - ((sum1[i - hng] - sum1[i - hnw] + sum1[-1] - sum1[i + hng]) / counts) ** 2) * counts / (counts - 1))
        for i in np.arange(len(signal) - hng, len(signal)):
            counts = summ[i - hng] - summ[i - hnw]
            mov_var.append(((sum2[i - hng] - sum2[i - hnw]) / counts
                            - ((sum1[i - hng] - sum1[i - hnw]) / counts) ** 2) * counts / (counts - 1))
    if not return_total_var:
        if not return_mask:
            return np.abs(np.array(mov_var))  # because small fluctuations around zero can occur
        else:
            return np.abs(np.array(mov_var)), mask
    else:
        if not return_mask:
            return np.abs(np.array(mov_var)), total_var  # because small fluctuations around zero can occur
        else:
            return np.abs(np.array(mov_var)), mask, total_var


def tanhstep(x, center=0., hwhm=1., start_from_zero=False):
    if type(x) == list or type(x) == np.ndarray:
        res = (1. + np.tanh((np.array(x) - center) / hwhm * 1 / 2 * np.log(3))) / 2.
        if start_from_zero:
            m = np.nanmax(res)
            res -= np.nanmin(res)
            res *= m / np.nanmax(res)
        return res
    else:
        return (1. + np.tanh((np.array(x) - center) / hwhm * 1 / 2 * np.log(3))) / 2.


def trend_fft(data, nperseg, endlen=None, mask=None, data_low=None, fit_order='cubic'):
    """
    Calculates low-frequency trend of the `data` by zeroing its high-frequency
    components (correspondinh to wavelength >`nperseg`) in the Fourier transform.
    After that, treats the ends of the data (size of the end is `endlen`)
    by fitting quadratic polynomials to them in order to avoid bad behaviour
    of the trend from Fourier components which tend to repeat oscillatory behaviour
    in the ends and do not follow the data.

    Parameters
    ----------
    data : numpy.array or list of float
        Data vector to be detrended
    nperseg : int
        Wavelength below which the frequencies should be included in the trend.
    endlen : int
        Length of the data in the ends to be fitted by quadratic polynomial
    mask : array of bool
        Masking good data points with `True`, bad data points with `False`.
    data_low : numpy.ndarray or None
        The low-frequency trend. If None, it is found from the zeroing Fourier components,
        and it is the usual scenario. In rare cases, the required procedure is only to
        fix the ends with the fit. In this case, data_low can be specified.
        The length should be the same as the length of `data`.
    fit_order : str : 'quadratic' or 'cubic'
    Returns
    -------
    data_low : numpy.array or list of float
        Smooth low-frequency trend of the `data`

    """
    assert fit_order == 'quadratic' or fit_order == 'cubic', 'Undefined `fit_order` specified!'

    data = np.array(data)
    time = np.arange(len(data))
    nanmask = ~np.isnan(data)
    if not (nanmask).all():
        data_i = interpolate_over_mask(data, mask=nanmask)
    else:
        data_i = data
    if mask is None:
        mask = nanmask
    if len(data) == 1:
        warnings.warn('WHY YOU GIVE ME DATA OF LENGTH 1 TO FIT???')
    if (len(data) <= nperseg) or (len(data) // 2 <= endlen):
        # parabolic fit or the entire data
        if fit_order == 'quadratic':
            msk1 = np.array(mask).astype(bool)
            dat1 = (data_i)[msk1]
            tim1 = (time)[msk1]
            x00 = np.nanmean(dat1)
            cent = dat1[len(dat1) // 2]
            x10 = (tim1[-1] + tim1[0]) / 2
            x20 = ((dat1[0] - cent) / (tim1[0] - x10) ** 2 + (dat1[-1] - cent) / (tim1[-1] - x10) ** 2) / 2
            sol = minimize(lambda x: np.nansum((dat1 - (x[2] * (tim1 - x[1]) ** 2 + x[0])) ** 2),
                           np.array([x00, x10, x20]))
            data_low = (sol.x[2] * (np.arange(len(data)) - sol.x[1]) ** 2 + sol.x[0])
        else:
            msk1 = np.array(mask).astype(bool)
            dat1 = (data_i)[msk1]
            tim1 = (time)[msk1]
            tim10 = tim1[0]
            tim1 = tim1 - tim10
            x00 = (dat1[0] * tim1[-1] - dat1[-1] * tim1[0]) / (tim1[-1] - tim1[0])
            x10 = (dat1[-1] - dat1[0]) / (tim1[-1] - tim1[0])
            x20 = 0.
            x30 = 0.
            sol = minimize(
                lambda x: np.nansum((dat1 - (x[3] * tim1 ** 3 + x[2] * tim1 ** 2 + x[1] * tim1 + x[0])) ** 2),
                np.array([x00, x10, x20, x30]))
            data_low = (sol.x[3] * (np.arange(len(data))) ** 3
                        + sol.x[2] * (np.arange(len(data))) ** 2
                        + sol.x[1] * (np.arange(len(data))) + sol.x[0])

    else:
        if endlen is None:
            endlen = nperseg
        assert endlen < len(data) // 2, '`endlen` is too big!'

        if data_low is None:
            data_rf = np.fft.rfft(np.array(data_i))
            data_rf[int(len(data_i) / nperseg):] = 0.
            data_low = np.real(np.fft.irfft(data_rf, n=len(data_i)))
        else:
            assert len(data_low) == len(data), 'The lengths of `data_low` and `data` must be equal!'

        w1 = tanhstep(np.arange(endlen), center=endlen / 4 * 3, hwhm=endlen / 16, start_from_zero=True)
        w2 = 1 - w1
        w3 = w1[::-1]
        w4 = w2[::-1]
        data_low[:endlen] *= w1
        data_low[-endlen:] *= w3

        # fit the beginning
        if fit_order == 'quadratic':
            msk1 = np.array(mask[:endlen]).astype(bool)
            dat1 = (data_i[:endlen])[msk1]
            tim1 = (time[:endlen])[msk1]
            x00 = np.nanmean(dat1)
            cent = dat1[len(dat1) // 2]
            x10 = (tim1[-1] + tim1[0]) / 2
            x20 = ((dat1[0] - cent) / (tim1[0] - x10) ** 2 + (dat1[-1] - cent) / (tim1[-1] - x10) ** 2) / 2
            sol = minimize(lambda x: np.nansum((dat1 - (x[2] * (tim1 - x[1]) ** 2 + x[0])) ** 2),
                           np.array([x00, x10, x20]))
            data_low[:endlen] += (sol.x[2] * (np.arange(endlen) - sol.x[1]) ** 2 + sol.x[0]) * w2
        else:
            msk1 = np.array(mask[:endlen]).astype(bool)
            dat1 = (data_i[:endlen])[msk1]
            tim1 = (time[:endlen])[msk1]
            tim10 = tim1[0]
            tim1 = tim1 - tim10
            x00 = (dat1[0] * tim1[-1] - dat1[-1] * tim1[0]) / (tim1[-1] - tim1[0])
            x10 = (dat1[-1] - dat1[0]) / (tim1[-1] - tim1[0])
            x20 = 0.
            x30 = 0.
            sol = minimize(
                lambda x: np.nansum((dat1 - (x[3] * tim1 ** 3 + x[2] * tim1 ** 2 + x[1] * tim1 + x[0])) ** 2),
                np.array([x00, x10, x20, x30]))
            data_low[:endlen] += (sol.x[3] * (np.arange(endlen)) ** 3
                                  + sol.x[2] * (np.arange(endlen)) ** 2
                                  + sol.x[1] * (np.arange(endlen)) + sol.x[0]) * w2

        # fit the end:
        if fit_order == 'quadratic':
            msk1 = np.array(mask[len(data_i) - endlen:]).astype(bool)
            dat1 = (data[len(data) - endlen:])[msk1]
            tim1 = (time[len(data) - endlen:])[msk1]
            x00 = np.nanmean(dat1)
            cent = dat1[len(dat1) // 2]
            x10 = (tim1[-1] + tim1[0]) / 2
            x20 = ((dat1[0] - cent) / (tim1[0] - x10) ** 2 + (dat1[-1] - cent) / (tim1[-1] - x10) ** 2) / 2
            sol = minimize(lambda x: np.nansum((dat1 - (x[2] * (tim1 - x[1]) ** 2 + x[0])) ** 2),
                           np.array([x00, x10, x20]))
            data_low[-endlen:] += (sol.x[2] * (np.arange(endlen) + time[len(data) - endlen] - sol.x[1]) ** 2 + sol.x[
                0]) * w4
        else:
            msk1 = np.array(mask[len(data_i) - endlen:]).astype(bool)
            dat1 = (data[len(data) - endlen:])[msk1]
            tim1 = (time[len(data) - endlen:])[msk1]
            tim10 = tim1[0]
            tim1 = tim1 - tim10
            x00 = (dat1[0] * tim1[-1] - dat1[-1] * tim1[0]) / (tim1[-1] - tim1[0])
            x10 = (dat1[-1] - dat1[0]) / (tim1[-1] - tim1[0])
            x20 = 0.
            x30 = 0.
            sol = minimize(
                lambda x: np.nansum((dat1 - (x[3] * tim1 ** 3 + x[2] * tim1 ** 2 + x[1] * tim1 + x[0])) ** 2),
                np.array([x00, x10, x20, x30]))
            data_low[-endlen:] += (sol.x[3] * (np.arange(endlen) + time[len(data) - endlen] - tim10) ** 3
                                   + sol.x[2] * (np.arange(endlen) + time[len(data) - endlen] - tim10) ** 2
                                   + sol.x[1] * (np.arange(endlen) + time[len(data) - endlen] - tim10) + sol.x[0]) * w4

    return data_low


def detrend_piecewise(data, big_mask=None, mask=None, nperseg=None, endlen=None, nsig_step=1.5,
                      checklen=64, residual_threshold_step_in_sigma=2.,
                      slice_for_residual_step=5, return_trend=False):
    """
    Detrends the data using `trend_fft` function in data which can contain
    several pieces separated by a step. Different pieces are separated by holes in
    `mask`. If the difference between the ends of the two subsequent pieces
    exceeds high-frequency standard deviation of the data multiplied by `nsig_step`,
    then the pieces are detrended separately.
    Attention: trend will contain nans in the places of mask.

    Parameters
    ----------
    data : Union[ndarray, List[float]] # numpy.array or list of float
        Data vector to be detrended
    nperseg : int
        Wavelength below which the frequencies should be included in the trend.
    endlen : int
        Length of the data in the ends to be fitted by quadratic polynomial
    mask : array of bool
        Masking good data points with `True`, bad data points with `False`.
    nsig_step : float
        Minimal step between the edges of t wo subsequent pieces so that they
        will be detrended separately (in units of high-frequency std of `data`)
    checklen : int
        Length of gap extension data from the edges used to calculate the
        residuals to determine where are the steps.
    residual_threshold_step_in_sigma : float
        In the end, the trend is verified for having steps around holes in
        the residuals. If the residuals before and after the hole differ more than
        by `residual_threshold_step_in_sigma` sigma, the piece will be cut, and
        its ends will be fitted separately.
    slice_for_residual_step : int
        Length of slice used to calculate the median values around the hole
        which are needed to identify steps after the first trend attempt.
    return_trend : bool
        Whether to return trend in addition to detrended data

    Returns
    -------
    if return_trend:
        detrended, trend : numpy.arrays
        Detrended data and trend
    else:
        detrended : numpy.array
        `detrended` is detrended data containing zeros in the bad places of masks.
        `trend` is the low-frequency trend of the data containinhg nans
        in the bad places of masks

    """
    threshold_step_size_std = residual_threshold_step_in_sigma  # redefining names
    medlen = slice_for_residual_step

    data = np.array(data)  # copying and going to numpy

    nanmask = ~np.isnan(data)
    if not nanmask.all():  # interpolating data if it contains nans
        data_i = interpolate_over_mask(data, mask=nanmask)  # will need for fft
    else:
        data_i = data

    if mask is None:
        mask = nanmask
    else:
        mask = np.array(mask).astype(bool)
        mask *= nanmask

    if big_mask is None:  # big mask shows potential jumps. If there is no big mask, no jumps will be checked for.
        big_mask = clean_small_holes(mask, max_hole=5)  # np.ones(len(data)).astype(bool)
    else:
        big_mask = np.array(big_mask).astype(bool)

    # remove cases when there is only one unmasked point surrounded by masked because these pieces are hard to detrend
    (big_mask[1:-1])[(big_mask & ~np.roll(big_mask, 1) & ~np.roll(big_mask, -1))[1:-1]] = False

    if nperseg is None:
        nperseg = len(data) // 10
    if endlen is None:
        endlen = nperseg

    detrended = np.zeros(len(data))  # will collect here the result

    ind_i = (np.where(mask)[0])[0]  # start where there is the first valid point
    ind_f = (np.where(mask)[0])[-1] + 1  # end with the last valid point
    mask0 = np.array(big_mask[ind_i:ind_f])  # mask of the working region
    endpieces = np.where((~mask0) * np.roll(mask0, 1))[0] + ind_i  # ends of pieces split by holes
    endpieces = np.concatenate((endpieces, [ind_f]))
    begpieces = np.where(mask0 * np.roll(~mask0, 1))[0] + ind_i  # beginnings of pieces split by holes
    begpieces = np.concatenate(([ind_i], begpieces))

    fdat = np.abs(np.fft.rfft(data_i))
    std_fast_0 = (np.mean((fdat[-len(data) // 4:]) ** 2) / (len(data))) ** 0.5  # standard deviation of fase modes
    std_fast = std_fast_0 + np.max(fdat) / 20000  # I don't remember why exactly this was made but it worked

    # now we will select the bad pieces which have to be detrended separately:
    beg_big_pieces = [ind_i]
    end_big_pieces = []
    for i in range(1, len(begpieces)):
        i1 = max(endpieces[i - 1] - checklen,
                 begpieces[i - 1])  # if the piece before the hole is too small, take it all
        i2 = min(begpieces[i] + checklen, endpieces[i])  # if the piece after the hole is too small, take it all
        mask1 = big_mask[i1: i2]
        dat1 = (data_i[i1: i2])[mask1]
        tim1 = np.arange(len(mask1))[mask1]

        x00 = np.nanmean(dat1)  # define initial fit parameters
        cent = dat1[len(dat1) // 2]
        x10 = (tim1[-1] + tim1[0]) / 2
        x20 = ((dat1[0] - cent) / (tim1[0] - x10) ** 2 + (dat1[-1] - cent) / (tim1[-1] - x10) ** 2) / 2
        sol = minimize(lambda x: np.nansum((dat1 - (x[2] * (tim1 - x[1]) ** 2 + x[0])) ** 2), np.array([x00, x10, x20]))
        # tr_end = sol.x[2]*(tim1-sol.x[1])**2 + sol.x[0]
        mres = (sol.fun / len(dat1)) ** 0.5  # mean residual

        if mres > nsig_step * std_fast:  # if the fit does not fit the two pieces together, detrend them separately
            end_big_pieces.append(endpieces[i - 1])
            beg_big_pieces.append(begpieces[i])
    end_big_pieces.append(ind_f)

    # make one detrending and check it is ok inside the pieces:
    for i in range(len(beg_big_pieces)):
        piece = np.array(data_i[beg_big_pieces[i]:end_big_pieces[i]])
        mask_p = mask[beg_big_pieces[i]:end_big_pieces[i]]
        big_mask_p = big_mask[beg_big_pieces[i]:end_big_pieces[i]]
        trend_local = trend_fft(piece, nperseg=nperseg, endlen=endlen, mask=mask_p)
        # TODO: make here something to avoid repeating detrending if all is fine
        begholes = np.where(big_mask_p * np.roll(~big_mask_p, -1))[0] + 1  # define begholes inside the piece
        endholes = np.where(big_mask_p * np.roll(~big_mask_p, 1))[0]

        beg_small_pieces = [0]
        end_small_pieces = []
        for j in range(len(begholes)):
            if ((len(np.where(~np.isnan(piece[begholes[j] - medlen:begholes[j]]))[0]) >= 3)
                    and (len(np.where(~np.isnan(piece[endholes[j]:endholes[j] + medlen]))[0]) >= 3)
                    and ((np.abs(np.nanmedian(piece[begholes[j] - medlen:begholes[j]]
                                              - trend_local[begholes[j] - medlen:begholes[j]])
                                 - np.nanmedian(piece[endholes[j]:endholes[j] + medlen]
                                                - trend_local[endholes[j]:endholes[j] + medlen])
                                 ) / std_fast) > threshold_step_size_std)):
                # inds_step.append((beg_big_pieces[i]+begholes[j], beg_big_pieces[i]+endholes[j]))
                end_small_pieces.append(begholes[j])
                beg_small_pieces.append(endholes[j])
        end_small_pieces.append(len(piece))
        if len(end_small_pieces) == 1:
            detrended[beg_big_pieces[i]: end_big_pieces[i]] = (
                    np.array(data[beg_big_pieces[i]:end_big_pieces[i]]) - trend_local)
        else:
            for k in range(len(beg_small_pieces)):
                small_piece = np.array(piece[beg_small_pieces[k]:end_small_pieces[k]])
                mask_s = np.array(mask_p[beg_small_pieces[k]:end_small_pieces[k]])
                trend_s_0 = np.array(trend_local[beg_small_pieces[k]:end_small_pieces[k]])
                trend_s = trend_fft(small_piece, nperseg=nperseg, endlen=endlen, mask=mask_s, data_low=trend_s_0)
                detrended[beg_big_pieces[i] + beg_small_pieces[k]:beg_big_pieces[i] + end_small_pieces[k]] = (
                        np.array(data[beg_big_pieces[i] + beg_small_pieces[k]:beg_big_pieces[i] + end_small_pieces[
                            k]]) - trend_s)
                # trend[beg_big_pieces[i]:end_big_pieces[i]] = trend_local

    trend = np.array(data_i - detrended)
    if return_trend:
        return detrended, trend
    else:
        return detrended