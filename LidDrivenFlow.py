# Navier-Stokes equations using Taylor-Hood elements
# The required modules are first imported:

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, la, log
from dolfinx.fem import (
    Constant,
    Function,
    dirichletbc,
    extract_function_spaces,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner, dot

# We create a {py:class}`Mesh <dolfinx.mesh.Mesh>`, define functions for
# locating geometrically subsets of the boundary, and define a function
# for the  velocity on the lid:

# Create mesh
msh = create_rectangle(
    MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [1000, 1000], CellType.triangle
)

# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)

# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)

# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))
    
# Two {py:class}`function spaces <dolfinx.fem.FunctionSpace>` are
# defined using different finite elements. `P2` corresponds to a
# continuous piecewise quadratic basis (vector) and `P1` to a continuous
# piecewise linear basis (scalar).

P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,))
P1 = element("Lagrange", msh.basix_cell(), 1)
V, Q = functionspace(msh, P2), functionspace(msh, P1)

# Boundary conditions for the velocity field are defined:

# No-slip condition on boundaries where x = 0, x = 1, and y = 0
noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)  # type: ignore
facets = locate_entities_boundary(msh, 1, noslip_boundary)
bc0 = dirichletbc(noslip, locate_dofs_topological(V, 1, facets), V)

# Driving (lid) velocity condition on top boundary (y = 1)
lid_velocity = Function(V)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V, 1, facets))

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]

# Create the Taylot-Hood function space
TH = mixed_element([P2, P1])
W = functionspace(msh, TH)

# No slip boundary condition
W0 = W.sub(0)
Q, _ = W0.collapse()
noslip = Function(Q)
facets = locate_entities_boundary(msh, 1, noslip_boundary)
dofs = locate_dofs_topological((W0, Q), 1, facets)
bc0 = dirichletbc(noslip, dofs, W0)

# Driving velocity condition u = (1, 0) on top boundary (y = 1)
lid_velocity = Function(Q)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
dofs = locate_dofs_topological((W0, Q), 1, facets)
bc1 = dirichletbc(lid_velocity, dofs, W0)

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]

uh = fem.Function(W)
u, p = uh.split()
(v, q) = ufl.TestFunctions(W)

# Constants
rho = PETSc.ScalarType(1)
mu = PETSc.ScalarType(10000)

# Define the residual
R = mu*inner(grad(u),grad(v))*ufl.dx
R = R + inner(dot(u,grad(u)),v)*ufl.dx
R = R - inner(p,div(v))*ufl.dx
R = R - inner(q,div(u))*ufl.dx

Problem = NonlinearProblem(R, uh, bcs = bcs)

solver = NewtonSolver(MPI.COMM_WORLD, Problem) # Uses a newton solver
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

print(log.set_log_level(log.LogLevel.INFO))
n, converged = solver.solve(uh)
assert (converged) # Used in debugging

# Test