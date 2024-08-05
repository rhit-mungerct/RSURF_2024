import sys
import numpy as np
import math
import os
from dolfinx import mesh
from dolfinx import fem, la
from dolfinx.io import gmshio
from mpi4py import MPI
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace, dirichletbc, locate_dofs_topological, Constant, Function, form
from petsc4py import PETSc
import ufl
from ufl import div, dx, grad, inner
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
sys.path.append('~/Repo/RSURF_2024/StokesFlow')
from image2gmsh3D import *
from image2gmsh3D import main as meshgen
from image2inlet import solve_inlet_profiles

img_fname = sys.argv[1]
flowrate_ratio = float(sys.argv[2]) # Ratio of the two flow rates must be between 0 and 1, 0.5 is equal flow rate for both
print("Accepted Inputs, Begining Solving Inlet Profiles")
uh_1, msh_1, uh_2, msh_2 = solve_inlet_profiles(img_fname, flowrate_ratio)

print("Inlet Profile Solved, Starting to Make Mesh")
mesh = meshgen(img_fname)
mesh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
# mesh, _, ft = gmshio.read_from_msh("ChannelMesh.msh", MPI.COMM_WORLD, 0, gdim=3)
print("Finished Making Mesh")
ft.name = "Facet markers"

P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
P1 = element("Lagrange", mesh.basix_cell(), 1)
V, Q = functionspace(mesh, P2), functionspace(mesh, P1)

# Create boundary conditions
TH = mixed_element([P2, P1])
W = functionspace(mesh, TH)
# No Slip Wall Boundary Condition
W0 = W.sub(0)
Q, _ = W0.collapse()
noslip = Function(Q)
dofs = fem.locate_dofs_topological((W0, Q), mesh.topology.dim-1, ft.find(4))
bc_wall = fem.dirichletbc(noslip, dofs, W0)

print("Starting to Interpolate uh_1")
uh_1.x.scatter_forward()
inlet_1_velocity = Function(Q)
inlet_1_velocity.interpolate(
    uh_1,
    nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
        inlet_1_velocity.function_space.mesh,
        inlet_1_velocity.function_space.element,
        uh_1.function_space.mesh,
        padding=1.0e-6,
    ),
)
print("Finished Interpolating uh_1")
dofs = fem.locate_dofs_topological((W0, Q), mesh.topology.dim-1, ft.find(1))
bc_inlet_1 = dirichletbc(inlet_1_velocity, dofs, W0)

print("Starting to Interpolate uh_2")
uh_2.x.scatter_forward()
inlet_2_velocity = Function(Q)
inlet_2_velocity.interpolate(
    uh_2,
    nmm_interpolation_data=fem.create_nonmatching_meshes_interpolation_data(
        inlet_2_velocity.function_space.mesh,
        inlet_2_velocity.function_space.element,
        uh_2.function_space.mesh,
        padding=1.0e-6,
    ),
)
print("Finished Interpolating uh_2")

# Inlet 2 Velocity Boundary Condition
dofs = fem.locate_dofs_topological((W0, Q), mesh.topology.dim-1, ft.find(2))
bc_inlet_2 = dirichletbc(inlet_2_velocity, dofs, W0)

W0 = W.sub(1)
Q, _ = W0.collapse()
# Outlet Pressure Condition
dofs = fem.locate_dofs_topological((W0), mesh.topology.dim-1, ft.find(3))
bc_outlet = dirichletbc(PETSc.ScalarType(0), dofs, W0)
print("Start to Combine Boundary Conditions")
bcs = [bc_wall, bc_inlet_1, bc_inlet_2, bc_outlet]
W0 = W.sub(0)
Q, _ = W0.collapse()

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
delta = 0
f = Function(Q)
a = form((inner(grad(u), grad(v)) + inner(p, div(v)) + inner(div(u), q)) * dx)
L = form(inner(f,v) * dx)

# Assemble LHS matrix and RHS vector
print("Start Assembling Stiffness Matrix and Forcing Vector")
A = fem.petsc.assemble_matrix(a, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector(L)

fem.petsc.apply_lifting(b, [a], bcs=[bcs])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Set Dirichlet boundary condition values in the RHS
fem.petsc.set_bc(b, bcs)

# Create and configure solver
ksp = PETSc.KSP().create(mesh.comm)
ksp.setOperators(A)
ksp.setType("preonly")

# Configure MUMPS to handle pressure nullspace
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
pc.setFactorSetUpSolverType()
pc.getFactorMatrix().setMumpsIcntl(icntl=24, ival=1)
pc.getFactorMatrix().setMumpsIcntl(icntl=25, ival=0)
# Compute the solution
U = Function(W)
try:
    ksp.solve(b, U.x.petsc_vec)
except PETSc.Error as e:
    if e.ierr == 92:
        print("The required PETSc solver/preconditioner is not available. Exiting.")
        print(e)
        exit(0)
    else:
        raise e

# Split the mixed solution and collapse
u, p = U.sub(0).collapse(), U.sub(1).collapse()

# Compute norms
norm_u, norm_p = la.norm(u.x), la.norm(p.x)
if MPI.COMM_WORLD.rank == 0:
    print(f"(D) Norm of velocity coefficient vector (monolithic, direct): {norm_u}")
    print(f"(D) Norm of pressure coefficient vector (monolithic, direct): {norm_p}")

print("Finished Solving, Saving Solution Field")
# Save the pressure field
from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement
with XDMFFile(MPI.COMM_WORLD, "StokesChannelPressure.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    P3 = VectorElement("Lagrange", mesh.basix_cell(), 1)
    u1 = Function(functionspace(mesh, P3))
    u1.interpolate(p)
    pfile_xdmf.write_mesh(mesh)
    pfile_xdmf.write_function(u1)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, "StokesChannelVelocity.xdmf", "w") as pfile_xdmf:
    u.x.scatter_forward()
    P4 = VectorElement("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    u2 = Function(functionspace(mesh, P4))
    u2.interpolate(u)
    pfile_xdmf.write_mesh(mesh)
    pfile_xdmf.write_function(u2)

# def inner_velocity_expression(x):
#     # find/create an interpolation function (or just a couple of lines) to replace np.ones with velocities from uh_1
#     # Numpy array of velocities (last resort)
#     # Get coordinates of each velocity field to interpolate onto common grid
#     #coor_1 = V_1.tabulate_dof_coordinates()
#     u_1 = uh_1.x.array.real.astype(np.float32)
#     print(u_1)
#     #coor_2 = V_2.tabulate_dof_coordinates()
#     #u_2 = uh_2.x.array.real.astype(np.float32)
#     return np.stack(())
# # Inlet 1 Velocity Boundary Condition