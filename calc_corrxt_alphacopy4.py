import numpy as np

def calc_Cxt_optimized(Cxt_, steps, spin, alpha):
    """
    Version of calc_Cxt that avoids using np.roll
    The temporal (axis=0) and spatial (axis=1) shifts are independent
    """
    spin = spin[0:steps:fine_res]
    if alpha < 1:
        alpha_ = 1/alpha
    else:
        alpha_ = alpha
    
    T = spin.shape[0]
    safe_dist = max(L//2, L*np.pow(alpha,-x))
    
    for ti, t in enumerate(range(0, steps, fine_res)):
        # Handle temporal shift first - this applies to the entire array
        spin_t = spin[:-ti] if ti > 0 else spin
        spin_t_shifted = spin[ti:] #if ti > 0 else spin
        
        # For negative x values
        for x in range(-safe_dist + 1, -1):
            # Handle spatial shift
            x_abs = np.abs(x)
            spin_x = spin_t[:, :L-x_abs, :]
            spin_x_shifted = spin_t_shifted[:, x_abs:, :]
            
            # Calculate correlation
            Cxt_[ti, x] = np.sum(np.sum(spin_x * spin_x_shifted, axis=2)) / ((L - x_abs) * (T - ti))
        
        # For positive x values (including 0)
        for x in range(0, safe_dist + 1):
            # Handle spatial shift
            spin_x = spin_t[:, :-x, :] if x > 0 else spin_t
            spin_x_shifted = spin_t_shifted[:, x:, :] #if x > 0 else spin_t_shifted
            
            # Calculate correlation
            Cxt_[ti, x] = np.sum(np.sum(spin_x * spin_x_shifted, axis=2)) / ((L - x) * (T - ti))
    
    return Cxt_
