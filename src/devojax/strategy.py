import jax.numpy as jnp
import jax
from functools import partial

@partial(jax.jit, static_argnums=(1, 3, 4))
def draw_from_range_except_ind(
        key: jax.Array,
        total:int, 
        except_ind:int, 
        ndraws:int,
        replace:bool=False
    ):

    initial_draw = jax.random.choice(
        key, 
        jnp.arange(total - 1),
        shape=(ndraws,),
        replace=replace
    )

    inds = total - ((initial_draw - except_ind) % total + 1)
    return inds

draw_from_ranges_except_ind = jax.vmap(
    draw_from_range_except_ind,
    in_axes=(0, None, 0, None, None)
)

@partial(jax.jit, static_argnums = (3, 4))
def rand1(keys, positions, statistics, npop, nperpop):

    indices = draw_from_ranges_except_ind(
        keys, 
        nperpop, 
        jnp.arange(nperpop * npop) % nperpop, 
        3, 
        False
    ).reshape(npop, nperpop, 3)  # 3 unique indices excluding current sample for all particles

    abcvec = jnp.take_along_axis(positions[:,:,None,:], indices[:, :, :, None], axis=1) # shape (npop, nperpop, 3, ndim)
    return abcvec[:,:,0], abcvec[:,:,1], abcvec[:,:,2]

@partial(jax.jit, static_argnums = (3, 4))
def best1(keys, positions, statistics, npop, nperpop):
    indices = draw_from_ranges_except_ind(
        keys, 
        nperpop, 
        jnp.arange(nperpop * npop) % nperpop, 
        2, 
        False
    ).reshape(npop, nperpop, 2)  # 2 unique indices excluding current sample for all particles

    # get the best sample
    best_ind = jnp.argmin(statistics, axis=1, keepdims=True)
   
    avec = jnp.take_along_axis(positions, best_ind[:,:,None], axis=1)
    bcvec = jnp.take_along_axis(positions[:,:,None,:], indices[:, :, :, None], axis=1) # shape (npop, nperpop, 2, ndim)
    return avec, bcvec[:,:,0], bcvec[:,:,1]