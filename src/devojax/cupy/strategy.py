import cupy as cp

def draw_from_ranges_except_ind(
        generator,
        total, 
        except_ind, 
        ndraws,
    ):
    # this isn't correct but it's the best I can do at the moment with cupy

    initial_draw = generator.integers(0, total - 1, size=(except_ind.size, ndraws))
    inds = total - ((initial_draw - except_ind[:,None]) % total + 1)
    return inds

def rand1(generator, positions, statistics, npop, nperpop):

    indices = draw_from_ranges_except_ind(
        generator, 
        nperpop, 
        cp.arange(nperpop * npop) % nperpop, 
        3, 
    ).reshape(npop, nperpop, 3)  # 3 unique indices excluding current sample for all particles

    abcvec = cp.take_along_axis(positions[:,:,None,:], indices[:, :, :, None], axis=1) # shape (npop, nperpop, 3, ndim)
    return abcvec[:,:,0], abcvec[:,:,1], abcvec[:,:,2]

def best1(generator, positions, statistics, npop, nperpop):
    indices = draw_from_ranges_except_ind(
        generator, 
        nperpop, 
        cp.arange(nperpop * npop) % nperpop, 
        2, 
    ).reshape(npop, nperpop, 2)  # 2 unique indices excluding current sample for all particles

    # get the best sample
    best_ind = cp.argmin(statistics, axis=1, keepdims=True)
   
    avec = cp.take_along_axis(positions, best_ind[:,:,None], axis=1)
    bcvec = cp.take_along_axis(positions[:,:,None,:], indices[:, :, :, None], axis=1) # shape (npop, nperpop, 2, ndim)
    return avec, bcvec[:,:,0], bcvec[:,:,1]