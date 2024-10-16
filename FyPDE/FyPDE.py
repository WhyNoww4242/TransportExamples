"""

"""

# Modules 
import numpy as np
from scipy.sparse import diags
from scipy.integrate import odeint

# Define the spatial coordinate
class Space:
    """
    Spacial coordinate with an start, stop and number of steps
    """
    def __init__(self, initial, final, steps):
        self.initial = initial
        self.final = final
        self.steps = steps
        # Discretizes the spacial coordinate
        self.grid = np.linspace(initial, final, steps, endpoint = True)
    
    def find_space_index(self, value):
        # Find the closest index
        idx = (np.abs(self.grid - value)).argmin()
        # Check if it's an exact match
        if abs(self.grid[idx] - value) <= 1e-5:
            return idx
        else:
            print(f'Error index not found, finding closest match', {self.grid[idx]})
            return idx

# Defines the time coordinate
class Time:
    """
    Temporal coordinate with an start, stop and number of steps
    """
    def __init__(self, initialT, finalT, stepsT):
        self.initialT = initialT
        self.finalT = finalT
        self.stepsT = stepsT
        # Discretizes the time coordinate
        self.gridT = np.linspace(initialT, finalT, stepsT, endpoint = True)
    
    def find_time_index(self, time):
        # Find the closest index
        idx = (np.abs(self.gridT - time)).argmin()
        
        # Check if it's an exact match
        if abs(self.gridT[idx] - time) <= 1e-5:
            return idx
        else:
            print(f'Error index not found, finding closest match',
                  {self.gridT[idx]})
            return idx



def finitediff(diff,bounds,z_Space):
    """
    The boundary condition is taken as an array with the first element is the 
    lower boundary and the second is the upper boundary. Each equation should 
    be write in the form: a*du/dx = b*u+c 
    where the respective array is [[a1, b1, c1], [a2, b2, c2]].
    
    """
    # Extracts the number of spacial steps
    n = z_Space.steps
    # Calcuates distance between eachpoint
    delta_x = abs(z_Space.grid[1]-z_Space.grid[0])
    #resizing for the purpose of index
    n=n-1
    n = int(n)
    # generates sparse matrix   
    diagonals = [-2, 1, 1]
    offsets = [0, -1, 1]
    A = diags(diagonals, offsets, shape=(n+1, n+1), format='csr')
    A[0,1]=2
    A[n,-2]=2
    # converts sparse matrix into a full array
    A = A.toarray()
    # creates the boundary vector for the +b section of the diff eq
    # bound_vec is constant added to ui to approximate the derivative
    bound_vec = np.zeros(n+1)
    # ensure the at the constant has the correct sign
    bound_vec[0] = -1
    bound_vec[n] = 1
    
    # loop checks boundary condition for Dirichlet bounds
    # Dirichlet bounds must be treated differently because of an undefined term
    for i in [0,n]:
        # used to fix the index into the bounds
        # this only works because their are only 2 boundary conditions
        j = i//n
        # the first if, checks if the for a Dirichlet boundary conditions
        # aka ui = constant
        if bounds[j,0] == 0:
            # zeros out the row so the is no change at the boundary
            A[i,:] = 0
            # changes the diffusion at the bound so bound_vec stays const
            # D(x) = dx^2 so dx^2/dx^2 = 1
            diff[i] = delta_x**2
            # sets the bound to a constant varible (ui = -ci/bi)
            bound_vec[i] = -1*bounds[j,2]/bounds[j,1]
        # else statement, the bound is treated as mixed
        else:
            # adds the correction factor to the second derivative ui
            A[i,i] = A[i,i] + 2*(delta_x/bounds[j,0])*bounds[j,1]*bound_vec[i]
            # adds the constant correciton factor to the bound_vec
            bound_vec[i] = 2*(delta_x/bounds[j,0])*bounds[j,2]*bound_vec[i]
            
    # multiples A by the diffusion to get the second derivative approximation 
    A = ((diff/(delta_x**2))*A.transpose()).transpose()
    # multiples the constant bound_vec by the diffusion
    bound_vec = bound_vec*(diff/(delta_x**2))
    
    return A, bound_vec

def deriv(U,t,A,b):
    """
    dU/dt = A*U+b
    """

    return np.dot(A,U)+b

def PDEsolver(diff, z_Space, t_Time, inital, bounds):
    """
    Takes the diffusion and inital value (as vector), z_Space and t_Time 
    objects as the coordinate system for calculation. The boundary condition 
    is taken as an array with the first element is the lower boundary and 
    the second is the upper boundary. Each equation should be write in the form 
    a*du/dx = b*u+c where the respective array is 
    [[a1, b1, c1],
     [a2, b2, c2]].
    
    This function uses finite difference method to solve PDEs in the form 
    u_t = D(x)*u_xx. It works by creating a matrice and vector to
    approximate the second derivate with respect to x. Special conditions
    are given to the boundaries, which can be found in the finitemat function.
    After creating the respective matrices, it turns the PDE into a system 
    of ODEs using method of lines. The spacing is done in x and forward steps
    are taken in time using odeint from the Scipy library.
    """
    # calculates the finite difference matrix and boundary vector
    # A is an approximation of the derivative
    # b is the vector for the constant at the boundaries
    A, b = finitediff(diff,bounds,z_Space)
    
    # creates a time vector to discretize time
    time = t_Time.gridT
    # creates a time vector to discretize space
    posit = z_Space.grid
    
    # solves the the PDE as a system of ODEs using odeint from scipy
    U = odeint(deriv, inital, time, args=(A,b))
    
    # returns U matrix [time, position], time vector, position vector
    return U, time, posit