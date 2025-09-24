import cupy as cp

from numpy.typing import ArrayLike
from typing import Optional

def latin_hypercube(
            generator: cp.random.Generator,
            ndim: int,
            nsamp: int,
            nperinterval: int, 
            bounds: Optional[ArrayLike] = None,
        ) -> ArrayLike:
        """
        Generate samples in n-dimensions using latin hypercube sampling.

        Args:
            ndim: Number of dimensions.
            nsamp: Number of samples to generate (number of intervals).
            nperinterval: Number of samples per interval (number of sets of samples to produce).
            bounds: Array of shape (nperinterval, ndim, 2) specifying the 
                lower and upper bounds for each dimension. If None, samples are generated in [0, 1).
        
        Returns:
            Array: Array of shape (nperinterval, nsamp, ndim) with samples in bounds.
        """

        samples = generator.uniform(0, 1, size=(nperinterval, nsamp, ndim))

        perms = cp.tile(cp.arange(1, nsamp + 1)[None,:,None], (nperinterval, 1, ndim))
        perms = cp.take_along_axis(perms, generator.standard_normal(perms.shape).argsort(axis=1), axis=1)

        if bounds is not None:
            samples = bounds[:,None,:,0] + (perms - samples) * (bounds[:,None,:,1] - bounds[:,None,:,0]) / nsamp
        else:
            samples = (perms - samples) / nsamp

        return samples
