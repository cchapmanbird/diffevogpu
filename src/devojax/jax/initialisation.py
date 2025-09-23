import jax.numpy as jnp
import jax
from typing import Optional

def latin_hypercube(
            key: jax.Array, 
            ndim: int,
            nsamp: int,
            nperinterval: int, 
            bounds: Optional[jax.Array] = None,
        ) -> jax.Array:
        """
        Generate samples in n-dimensions using latin hypercube sampling.

        Args:
            key (jax.Array): JAX random key.
            ndim (int): Number of dimensions.
            nsamp (int): Number of samples to generate (number of intervals).
            nperinterval (int): Number of samples per interval (number of sets of samples to produce).
            bounds (Optional[jax.Array]): Array of shape (nperinterval, ndim, 2) specifying the 
                lower and upper bounds for each dimension. If None, samples are generated in [0, 1).
        
        Returns:
            jax.Array: Array of shape (nperinterval, nsamp, ndim) with samples in [0, 1).
        """

        samples = jax.random.uniform(key, shape=(nperinterval, nsamp, ndim))

        perms = jnp.tile(jnp.arange(1, nsamp + 1)[None,:,None], (nperinterval, 1, ndim))
        perms = jax.random.permutation(key, perms, axis=1, independent=True)

        if bounds is not None:
            samples = bounds[:,None,:,0] + (perms - samples) * (bounds[:,None,:,1] - bounds[:,None,:,0]) / nsamp
        else:
            samples = (perms - samples) / nsamp

        return samples

        # samples = jax.random.uniform(key, shape=(nsamp, ndim))

        # perms = jnp.tile(jnp.arange(1, nsamp + 1), (ndim, 1))
        # perms = jax.random.permutation(key, perms, axis=1, independent=True).T

        # perms = jnp.tile(jnp.arange(1, nsamp + 1)[:,None], (1, ndim))
        # perms = jax.random.permutation(key, perms, axis=0, independent=True)

        # print(samples.shape, perms.shape, bounds.shape)

        # if bounds is not None:
        #     samples = bounds[:,0] + (perms - samples) * (bounds[:,1] - bounds[:,0]) / nsamp
        # else:
        #     samples = (perms - samples) / nsamp

        # return samples

