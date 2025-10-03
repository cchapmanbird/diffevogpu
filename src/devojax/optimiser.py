
# from .jax.initialisation import latin_hypercube
# from .jax.strategy import rand1, best1

import numpy as np
from typing import Union, Optional, Callable
from numpy.typing import ArrayLike

class DifferentialEvolution:
    def __init__(
            self,
            func: Callable,
            bounds: ArrayLike,
            backend: str = 'jax',
            key_or_generator: Optional[ArrayLike]=None,
            nperpop: int=15,
            npop: int=1,
            recombination: Union[float, ArrayLike]=0.7,
            mutation: Union[float, ArrayLike]=0.5,
            tol: Optional[Union[float, ArrayLike]] = None,
            strategy: str='best',
            maxiter: int=100,
            jittable_func: bool=False,
            disp: bool=False,
        ):
        self.backend = backend

        self.func = func
        self.npop = npop

        if bounds.ndim == 3:
            assert bounds.shape[0] == npop, "First dimension of bounds must match number of populations."

        assert strategy in ['rand', 'best'], "Strategy must be 'rand' or 'best'."
        self.strategy = strategy

        if backend == 'jax':
            self._jax_init(bounds, key_or_generator)
        elif backend == 'cupy':
            self._cupy_init(bounds, key_or_generator)
        else:
            raise ValueError("Backend must be 'jax' or 'cupy'.")
        
        self.ndim = self.bounds.shape[1]
        self.nperpop = nperpop * self.ndim

        self.recombination = self._ensure_control_input_shape(recombination)
        self.mutation = self._ensure_control_input_shape(mutation)
        
        self.tol = tol
        if self.tol is not None:
            self.tol = self._ensure_control_input_shape(tol)
            if self.xp.all(self.tol == 0):
                self.tol = None

        self.maxiter = maxiter
        self.jittable_func = jittable_func
        self.disp = disp
    
        self.initial_positions = None
        self.nfev = 0
        self.niter = 0
        self.pos = []
        self.stat = []

    def _jax_init(self, bounds, key):
        import jax
        import jax.numpy as jnp
        from .jax.initialisation import latin_hypercube
        from .jax.strategy import rand1, best1
        from .jax.steps import _de_step, _propose_new_positions, _accept_reject, _check_if_tol_exceeded

        if bounds.ndim == 2:
            bounds = jnp.tile(bounds[None, :, :], (self.npop, 1, 1))

        self.bounds = bounds

        self.xp = jnp
        
        self.latin_hypercube = latin_hypercube
        self._de_step = _de_step
        self._propose_new_positions = _propose_new_positions
        self._accept_reject = _accept_reject
        self._check_if_tol_exceeded = _check_if_tol_exceeded

        if self.strategy == 'rand':
            self.strategy_fun = rand1
        elif self.strategy == 'best':
            self.strategy_fun = best1

        if key is None:
            key = jax.random.PRNGKey(0)
        
        self.key_or_generator = key

    def _cupy_init(self, bounds, generator):
        import cupy as cp
        from .cupy.initialisation import latin_hypercube
        from .cupy.strategy import rand1, best1
        from .cupy.steps import _de_step, _check_if_tol_exceeded

        if bounds.ndim == 2:
            bounds = cp.tile(bounds[None, :, :], (self.npop, 1, 1))

        self.bounds = bounds

        self.xp = cp
        
        self.latin_hypercube = latin_hypercube
        self._de_step = _de_step
        self._check_if_tol_exceeded = _check_if_tol_exceeded

        if self.strategy == 'rand':
            self.strategy_fun = rand1
        elif self.strategy == 'best':
            self.strategy_fun = best1

        if generator is None:
            generator = cp.random.default_rng(0)
        self.key_or_generator = generator

    def as_numpy(self, obj: ArrayLike):
        if self.backend == 'jax':
            return np.asarray(obj)
        elif self.backend == 'cupy':
            return obj.get()

    def _ensure_control_input_shape(self, input):
        input = self.xp.asarray(input)
        if input.size == 1:
            input = self.xp.full(self.npop, input)
        else:
            assert input.size == self.npop, "Array of control parameters must match number of populations."

        return input
    
    def set_rng(self, key_or_generator: ArrayLike):
        """
        Update the random key (JAX) or generator (cupy).
        """
        if self.backend == 'jax':
            self.key_or_generator = key_or_generator
        elif self.backend == 'cupy':
            self.key_or_generator = key_or_generator

    def _step_nojit_func(
            self,
            positions: ArrayLike,
            statistics: ArrayLike,
    ):

        self.key_or_generator, new_positions = self._propose_new_positions(
            self.key_or_generator,
            positions,
            self.recombination,
            self.mutation,
            self.strategy_fun,
            self.npop,
            self.nperpop,
            self.ndim,
        )

        new_statistics = self.func(new_positions)

        positions, statistics = self._accept_reject(
            positions,
            statistics,
            new_positions,
            new_statistics
        )
        
        return positions, statistics

    def _step_jit_func(
            self,
            positions,
            statistics
        ):
        self.key_or_generator, positions, statistics = self._de_step(
            self.func,
            self.recombination,
            self.mutation,
            self.strategy_fun,
            self.npop,
            self.nperpop,
            self.ndim,
            positions,
            statistics,
            self.key_or_generator
        )
        return positions, statistics

    def _step_cupy_func(
            self,
            positions,
            statistics
        ):
        positions, statistics = self._de_step(
            self.func,
            self.recombination,
            self.mutation,
            self.strategy_fun,
            self.npop,
            self.nperpop,
            self.ndim,
            positions,
            statistics,
            self.key_or_generator
        )

        return positions, statistics

    def step(self,
             positions: ArrayLike, 
             statistics: ArrayLike
        ):
        """
        Perform one iteration of the differential evolution algorithm.

        Args:
            positions: Current positions of shape (npop, nperpop, ndim).
            statistics: Current statistics of shape (npop, nperpop).
        """

        if self.backend == 'cupy':
            return self._step_cupy_func(positions, statistics)
        
        if self.jittable_func:
            return self._step_jit_func(positions, statistics)
        else:
            return self._step_nojit_func(positions, statistics)

    def __call__(
            self,
            initial_positions: Optional[ArrayLike] = None,
    ):

        if initial_positions is not None:
            assert initial_positions.shape == (self.npop, self.nperpop, self.ndim), \
                "Initial positions must have shape (npop, nperpop, ndim)."
            self.initial_positions = initial_positions
        else:
            self.initial_positions = self.latin_hypercube(
                self.key_or_generator,
                ndim=self.ndim,
                nsamp=self.nperpop,
                nperinterval=self.npop,
                bounds=self.bounds,
                )

        positions = self.initial_positions
        statistics = self.func(self.initial_positions)
        
        ii = 1
        self.pos = [positions,]
        self.stat = [statistics,]
        self.nfev = 0

        while ii < self.maxiter:
            positions, statistics = self.step(positions, statistics)

            self.pos.append(positions)
            self.stat.append(statistics)
            
            self.nfev += self.npop * self.nperpop

            ii += 1

            if self.disp:
                print(f"Iteration {ii}, best: {self.xp.min(statistics, axis=1)}")

            if self.tol is not None and self._check_if_tol_exceeded(statistics, self.tol):
                break

        self.niter = ii
        return positions, statistics
