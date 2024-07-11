#!/usr/bin/env python

# Navier-Stokes equations using Taylor-Hood elements, including SUPG and PSPG stabilization methods
# The required modules are first imported:

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import gmsh
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, la, log
from dolfinx.fem import (
    Function,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner, dot, transpose, sym, nabla_grad, Identity

# We create a {py:class}`Mesh <dolfinx.mesh.Mesh>`, define functions for
# locating geometrically subsets of the boundary, and define a function
# for the  velocity on the lid:

Reynold = input("Enter the Desired Reynolds Number: ")

# Create mesh
gmsh.initialize()
L = 1
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, L, tag=1)
    
gmsh.model.occ.synchronize()

fluid_marker = 1
if mesh_comm.rank == model_rank:
    volumes = gmsh.model.getEntities(dim=gdim)
    assert (len(volumes) == 1)
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")
    
wall_marker, lid_marker, outlet_marker = 2, 3, 4
wall, lid, outlet = [], [], []

if mesh_comm.rank == model_rank:
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, L / 2, 0]) or np.allclose(center_of_mass, [L, L/2, 0]) or np.allclose(center_of_mass, [L/2, 0, 0]):
            wall.append(boundary[1])
        elif np.allclose(center_of_mass, [L/2, L, 0]):
            lid.append(boundary[1])
        elif np.allclose(center_of_mass, [0, 0, 0]):
            outlet.append(boundary[1])
            
    gmsh.model.addPhysicalGroup(1, wall, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Wall")
    gmsh.model.addPhysicalGroup(1, lid, lid_marker)
    gmsh.model.setPhysicalName(1, lid_marker, "Bottom Wall")
    gmsh.model.addPhysicalGroup(1, outlet, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")

if mesh_comm.rank == model_rank:
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")
    
mesh, _, ft = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
ft.name = "Facet markers"
    

"""    
msh = create_rectangle(
    MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])], [nx, nx], CellType.triangle
)

# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0)

# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)

# Pressure boundary condition, mark the bottom left corner (x = 0 and y = 0)
def outlet(x):
    return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0)) 

# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))
     """
# Two {py:class}`function spaces <dolfinx.fem.FunctionSpace>` are
# defined using different finite elements. `P2` corresponds to a
# continuous piecewise quadratic basis (vector) and `P1` to a continuous
# piecewise linear basis (scalar).

P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
P1 = element("Lagrange", mesh.basix_cell(), 1)
V, Q = functionspace(mesh, P2), functionspace(mesh, P1)

# Create the Taylot-Hood function space
TH = mixed_element([P2, P1])
W = functionspace(mesh, TH)

W0 = W.sub(0)
Q, _ = W0.collapse()

# No Slip Boundary Condition
bc_no_slip = dirichletbc([PETSc.ScalarType(1), PETSc.ScalarType(1)], locate_dofs_topological(Q, 1, ft.find(wall_marker)), Q)

# Lid Boundary Condition
bc_lid = dirichletbc(PETSc.ScalarType(1), locate_dofs_topological(Q, 1, ft.find(lid_marker)), Q)

W0 = W.sub(1)
Q, _ = W0.collapse()

# Pressure Boundary Condition
bc_pressure = dirichletbc(PETSc.ScalarType(0), locate_dofs_topological(Q, 0, ft.find(outlet_marker)), Q)

""" # No slip boundary condition
W0 = W.sub(0)
Q, _ = W0.collapse()
noslip = Function(Q)
facets = locate_entities_boundary(msh, 1, wall)
dofs = locate_dofs_topological((W0, Q), 1, facets)
bc0 = dirichletbc(noslip, dofs, W0)

# Driving velocity condition u = (1, 0) on top boundary (y = 1)
lid_velocity = Function(Q)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
dofs = locate_dofs_topological((W0, Q), 1, facets)
bc1 = dirichletbc(lid_velocity, dofs, W0)

# Define pressure boundary condition
W0 = W.sub(1)
Q, _ = W0.collapse()
Outlet_pressure = Function(Q)
facets = locate_entities_boundary(msh, 1, outlet)
dofs = locate_dofs_topological((W0, Q), 1, facets)
bc2 = dirichletbc(Outlet_pressure, dofs, W0) """

# Collect Dirichlet boundary conditions
bcs = [bc_no_slip, bc_lid, bc_pressure]

uh = fem.Function(W)
u, p = ufl.split(uh)
(v, q) = ufl.TestFunctions(W)

# Constants
rho = PETSc.ScalarType(1.0)
Re = PETSc.ScalarType(Reynold)
mu = PETSc.ScalarType(1/Re)
tauSUPG = PETSc.ScalarType(1.0)

# Define the residual
R = mu*inner(grad(u),grad(v))*ufl.dx
R = R + rho*inner(grad(u)*u,v)*ufl.dx
R = R - inner(p,div(v))*ufl.dx
R = R - inner(q,div(u))*ufl.dx
R = R + inner(tauSUPG*dot(u,grad(v)), inner(u,v))*ufl.dx

J = ufl.derivative(R, uh)

Problem = NonlinearProblem(R, uh, bcs, J)

solver = NewtonSolver(MPI.COMM_WORLD, Problem) # Uses a newton solver
ksp = solver.krylov_solver
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "tfqmr"
opts[f"{option_prefix}ksp_rtol"] = 1e-6
opts[f"{option_prefix}ksp_atol"] = 1e-9
opts[f"{option_prefix}pc_type"] = "none"
ksp.setFromOptions()
print(log.set_log_level(log.LogLevel.INFO))
solver.solve(uh)

u, p = uh.sub(0).collapse(), uh.sub(1).collapse()

# Save the velocity field
VelcoityFileName = str(f"LidDrivenFlowStabilizedRe{Reynold}.xdmf")

with XDMFFile(MPI.COMM_WORLD, VelcoityFileName, "w") as pfile_xdmf:
    u.x.scatter_forward()
    P4 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u2 = Function(functionspace(msh, P4))
    u2.interpolate(u)
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u2)