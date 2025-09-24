import cupy as cp
import cupy
from cupy._core import internal

####### backport from cupy 14.0.x to support put_along_axis ########
def _make_along_axis_idx(arr_shape, indices, axis):
    # compute dimensions to iterate over

    if not cupy.issubdtype(indices.dtype, cupy.integer):
        raise IndexError('`indices` must be an integer array')
    if len(arr_shape) != indices.ndim:
        raise ValueError(
            "`indices` and `arr` must have the same number of dimensions")

    shape_ones = (1, ) * indices.ndim
    dest_dims = list(range(axis)) + [None] + \
        list(range(axis + 1, indices.ndim))

    # build a fancy index, consisting of orthogonal aranges, with the
    # requested index inserted at the right location
    fancy_index = []
    for dim, n in zip(dest_dims, arr_shape):
        if dim is None:
            fancy_index.append(indices)
        else:
            ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim+1:]
            fancy_index.append(cupy.arange(n).reshape(ind_shape))

    return tuple(fancy_index)


def put_along_axis(arr, indices, values, axis):
    """
    Put values into the destination array by matching 1d index and data slices.
    This iterates over matching 1d slices oriented along the specified axis in
    the index and data arrays, and uses the former to place values into the
    latter. These slices can be different lengths.
    Functions returning an index along an axis, like `argsort` and
    `argpartition`, produce suitable indices for this function.
    Args:
        arr : cupy.ndarray (Ni..., M, Nk...)
            Destination array.
        indices : cupy.ndarray (Ni..., J, Nk...)
            Indices to change along each 1d slice of `arr`. This must match the
            dimension of arr, but dimensions in Ni and Nj may be 1 to broadcast
            against `arr`.
        values : array_like (Ni..., J, Nk...)
            values to insert at those indices. Its shape and dimension are
            broadcast to match that of `indices`.
        axis : int
            The axis to take 1d slices along. If axis is None, the destination
            array is treated as if a flattened 1d view had been created of it.
    .. seealso:: :func:`numpy.put_along_axis`
    """

    # normalize inputs
    if axis is None:
        if indices.ndim != 1:
            raise NotImplementedError(
                "Tuple setitem isn't supported for flatiter.")
        # put is roughly equivalent to a.flat[ind] = values
        cupy.put(arr, indices, values)
    else:
        axis = internal._normalize_axis_index(axis, arr.ndim)
        arr_shape = arr.shape

        # use the fancy index
        arr[_make_along_axis_idx(arr_shape, indices, axis)] = values

##############

def _de_step(func, recombination, mutation, strategy_fun, npop, nperpop, ndim, positions, statistics, generator):
        new_positions = _propose_new_positions(generator, positions, statistics, recombination,  mutation, strategy_fun, npop, nperpop, ndim)
        new_statistics = func(new_positions)
        positions, statistics = _accept_reject(positions, statistics, new_positions, new_statistics)
        return positions, statistics

def _propose_new_positions(generator, positions, statistics, recombination, mutation, strategy_fun, npop, nperpop, ndim):

        avec, bvec, cvec = strategy_fun(generator, positions, statistics, npop, nperpop)
        
        R = generator.integers(ndim, size=(npop, nperpop),)
        r_i = generator.random(size=(npop, nperpop, ndim))

        retain = r_i < recombination[:,None,None]
        put_along_axis(retain, R[:, :, None], True, axis=2)

        new_positions = cp.where(retain, (avec + mutation[:,None,None] * (bvec - cvec)), positions)

        return new_positions

def _accept_reject(positions, statistics, new_positions, new_statistics):
    repl_mask = (new_statistics <= statistics)
    positions = cp.where(repl_mask[:,:,None], new_positions, positions)
    statistics = cp.where(repl_mask, new_statistics, statistics)
    return positions, statistics

def _check_if_tol_exceeded(statistics, tolerance):
    return cp.all(cp.std(statistics, axis=1) < tolerance * cp.abs(cp.mean(statistics, axis=1)))
