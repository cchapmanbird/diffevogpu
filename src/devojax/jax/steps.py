import jax
import jax.numpy as jnp
from functools import partial

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
