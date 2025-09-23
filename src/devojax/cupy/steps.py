import cupy as cp

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

        retain = cp.put_along_axis(retain, R[:, :, None], True, axis=2, inplace=False)

        new_positions = cp.where(retain, (avec + mutation[:,None,None] * (bvec - cvec)), positions)

        return new_positions

def _accept_reject(positions, statistics, new_positions, new_statistics):
    repl_mask = (new_statistics <= statistics)
    positions = cp.where(repl_mask[:,:,None], new_positions, positions)
    statistics = cp.where(repl_mask, new_statistics, statistics)
    return positions, statistics

def _check_if_tol_exceeded(statistics, tolerance):
    return cp.all(cp.std(statistics, axis=1) < tolerance * cp.abs(cp.mean(statistics, axis=1)))
