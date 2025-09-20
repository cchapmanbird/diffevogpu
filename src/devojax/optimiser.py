import jax.numpy as jnp
import jax
from functools import partial

from .initialisation import latin_hypercube
from .strategy import rand1, best1

from typing import Union, Optional, Callable

@partial(jax.jit, static_argnums=(0, 3, 4, 5, 6,))
def _de_step(func, recombination, mutation, strategy_fun, npop, nperpop, ndim, positions, statistics, key):
        key, new_positions = _propose_new_positions(key, positions, statistics, recombination,  mutation, strategy_fun, npop, nperpop, ndim)
        new_statistics = func(new_positions)
        positions, statistics = _accept_reject(positions, statistics, new_positions, new_statistics)
        return key, positions, statistics

@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def _propose_new_positions(key, positions, statistics, recombination, mutation, strategy_fun, npop, nperpop, ndim):
        all_keys = jax.random.split(key, nperpop * npop + 3)
        
        avec, bvec, cvec = strategy_fun(all_keys[3:], positions, statistics, npop, nperpop)
        
        R = jax.random.randint(all_keys[1], (npop, nperpop), 0, ndim,)
        r_i = jax.random.uniform(all_keys[2], (npop, nperpop, ndim))

        retain = r_i < recombination[:,None,None]

        retain = jnp.put_along_axis(retain, R[:, :, None], True, axis=2, inplace=False)

        new_positions = jnp.where(retain, (avec + mutation[:,None,None] * (bvec - cvec)), positions)

        return all_keys[0], new_positions

@jax.jit
def _accept_reject(positions, statistics, new_positions, new_statistics):
    repl_mask = (new_statistics <= statistics)
    positions = jnp.where(repl_mask[:,:,None], new_positions, positions)
    statistics = jnp.where(repl_mask, new_statistics, statistics)
    return positions, statistics

@jax.jit
def _check_if_tol_exceeded(statistics, tolerance):
    return jnp.all(jnp.std(statistics, axis=1) < tolerance * jnp.abs(jnp.mean(statistics, axis=1)))


class DifferentialEvolution:
    def __init__(
            self,
            func: Callable,
            bounds: jax.Array,
            key: jax.Array,
            nperpop: int=15,
            npop: int=1,
            recombination: Union[float, jax.Array]=0.7,
            mutation: Union[float, jax.Array]=0.5,
            tol: Optional[Union[float, jax.Array]] = None,
            strategy: str='best',
            maxiter: int=100,
            jittable_func: bool=False,
        ):

        self.func = func

        if bounds.ndim == 3:
            assert bounds.shape[0] == npop, "First dimension of bounds must match number of populations."
        elif bounds.ndim == 2:
            bounds = jnp.tile(bounds[None, :, :], (npop, 1, 1))

        self.bounds = bounds
        self.ndim = bounds.shape[1]

        self.key = key
        self.nperpop = nperpop * self.ndim
        self.npop = npop
        
        self.recombination = self._ensure_control_input_shape(recombination)
        self.mutation = self._ensure_control_input_shape(mutation)
        
        self.tol = tol
        if self.tol is not None:
            self.tol = self._ensure_control_input_shape(tol)
            if jnp.all(self.tol == 0):
                self.tol = None

        assert strategy in ['rand', 'best'], "Strategy must be 'rand' or 'best'."
        self.strategy = strategy
        if strategy == 'rand':
            self.strategy_fun = rand1
        elif strategy == 'best':
            self.strategy_fun = best1

        self.maxiter = maxiter
        self.initial_positions = None

        self.jittable_func = jittable_func

    def _ensure_control_input_shape(self, input):
        input = jnp.asarray(input)
        if input.size == 1:
            input = jnp.full(self.npop, input)
        else:
            assert input.size == self.npop, "Array of control parameters must match number of populations."

        return input
    
    def set_rng(self, key: jax.Array):
        """
        Update the random key.
        """
        self.key = key

    def _step_nojit_func(
            self,
            positions: jax.Array,
            statistics: jax.Array,
    ):

        self.key, new_positions = _propose_new_positions(
            self.key,
            positions,
            self.recombination,
            self.mutation,
            self.strategy_fun,
            self.npop,
            self.nperpop,
            self.ndim,
        )

        new_statistics = self.func(new_positions)

        positions, statistics = _accept_reject(
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
        self.key, positions, statistics = _de_step(
            self.func,
            self.recombination,
            self.mutation,
            self.strategy_fun,
            self.npop,
            self.nperpop,
            self.ndim,
            positions,
            statistics,
            self.key
        )
        return positions, statistics

    def step(self,
             positions: jax.Array, 
             statistics: jax.Array
        ):
        """
        Perform one iteration of the differential evolution algorithm.

        Args:
            positions: Current positions of shape (npop, nperpop, ndim).
            statistics: Current statistics of shape (npop, nperpop).
        """

        if self.jittable_func:
            return self._step_jit_func(positions, statistics)
        else:
            return self._step_nojit_func(positions, statistics)

    def __call__(
            self,
            initial_positions: Optional[jax.Array] = None,
    ):

        if initial_positions is not None:
            assert initial_positions.shape == (self.npop, self.nperpop, self.ndim), \
                "Initial positions must have shape (npop, nperpop, ndim)."
            self.initial_positions = initial_positions
        else:
            self.initial_positions = latin_hypercube(
                key=self.key,
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

            if self.tol is not None and _check_if_tol_exceeded(statistics, self.tol):
                break

        self.niter = ii
        return positions, statistics
