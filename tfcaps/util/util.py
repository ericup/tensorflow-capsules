from typing import Union


def force_tuple(tup: Union[int, float, tuple, list], dimensions: int = None) -> tuple:
    """
    Expand single number to a tuple of specified 'dimensions' or return 'tup' if it is a tuple.
    :param tup:
    :param dimensions: Makes len(tup) == dimensions
    :return: tuple of length 'dimensions'
    """
    if isinstance(tup, (tuple, list)):
        if dimensions is None:
            return tuple(tup)
        elif len(tup) == dimensions:
            return tuple(tup)
        raise Warning("Specified dimensions not matched! Found %d, needed %d." % (len(tup), dimensions))
    else:
        return (tup,) * dimensions if dimensions is not None else 1
