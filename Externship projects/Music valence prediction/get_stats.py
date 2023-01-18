def get_stats(array):
    """
    Calculates the mean, variance, minimum and maximum values of an array.
    """
    from numpy import array, mean, var, min, max

    if len(array.shape) > 1:
        feature = np.concatenate(
            (
                np.mean(array, axis=1),
                np.var(array, axis=1),
                np.min(array, axis=1),
                np.max(array, axis=1),
            )
        )
    else:
        feature = np.array(
            (np.mean(array), np.var(array), np.min(array), np.max(array))
        )
    return feature
