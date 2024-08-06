import sys
import os
from dolfinx import mesh
from dolfinx import fem, la
from dolfinx.io import gmshio
from mpi4py import MPI
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace, dirichletbc, Function
from petsc4py import PETSc
import ufl
from ufl import div, dx, grad, inner
from image2gmsh3D import *
from image2gmsh3D import main as meshgen
from image2inlet import solve_inlet_profiles




# ------ Inputs ------
if len(sys.argv) == 3:
    img_fname = sys.argv[1] # Input a .png file that is black and white (example on github) to be used as the inlet profile
    flowrate_ratio = float(sys.argv[2]) # Ratio of the two flow rates must be between 0 and 1, 0.5 is equal flow rate for both
    channel_mesh_size = 0.25
elif len(sys.argv) == 4:
    img_fname = sys.argv[1]
    flowrate_ratio = float(sys.argv[2])
    channel_mesh_size = float(sys.argv[3]) # optional third argument for the element size of the 3D mesh

print("Accepted Inputs, Begining Solving Inlet Profiles")



# ------ Create mesh and inlet velocity profiles ------
uh_1, msh_1, uh_2, msh_2 = solve_inlet_profiles(img_fname, flowrate_ratio)
print("Inlet Profile Solved, Starting to Make Mesh")
mesh = meshgen(img_fname, channel_mesh_size)
mesh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
# mesh, _, ft = gmshio.read_from_msh("ChannelMesh.msh", MPI.COMM_WORLD, 0, gdim=3)
print("Finished Making Mesh")
ft.name = "Facet markers"



# ------ Create the different finite element spaces ------
P2 = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
P1 = element("Lagrange", mesh.basix_cell(), 1)
V, Q = functionspace(mesh, P2), functionspace(mesh, P1)



# ------- Create boundary conditions -------
TH = mixed_element([P2, P1]) # Create mixed element
W = functionspace(mesh, TH)
# No Slip Wall Boundary Condition
W0 = W.sub(0)
Q, _ = W0.collapse()
noslip = Function(Q) # Creating a function makes a function of zeros, which is what is needed for a no-slip BC
dofs = fem.locate_dofs_topological((W0, Q), mesh.topology.dim-1, ft.find(4))
bc_wall = fem.dirichletbc(noslip, dofs, W0)

# inlet 1 boundary condition (inlet 1 is the inner channel)
print("Starting to Interpolate uh_1")
'''
In order to interpolate between non-matching meshes in FEniCSx 0.0.8 to set the fully devolped inflow
boundary condition, the interpolate command needs extra information because the 2 meshes are different sizes and dimensions. 
In this code I am interpolating between a 2D fully devolped flow solution onto the 3D mesh of the entire channel,
and the "create_nonmatching_meshes_interpolation_data" is needed.
This community post has more information about using/debugging interpolation between meshes
https://fenicsproject.discourse.group/t/interpolation-data-has-wrong-shape-size/15453
'''
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

# interpolate inlet 2 boundary condition
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

# Inlet 2 Velocity Boundary Condition (inlet 2 is the outer channel)
dofs = fem.locate_dofs_topological((W0, Q), mesh.topology.dim-1, ft.find(2))
bc_inlet_2 = dirichletbc(inlet_2_velocity, dofs, W0)

W0 = W.sub(1)
Q, _ = W0.collapse()
# Outlet Pressure Condition
dofs = fem.locate_dofs_topological((W0), mesh.topology.dim-1, ft.find(3))
bc_outlet = dirichletbc(PETSc.ScalarType(0), dofs, W0)
print("Start to Combine Boundary Conditions")
bcs = [bc_wall, bc_inlet_1, bc_inlet_2, bc_outlet]



# ------ Create/Define weak form
W0 = W.sub(0)
Q, _ = W0.collapse()
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = Function(Q)

# Stabilization parameters per Andre Massing
h = ufl.CellDiameter(mesh)
Beta = 0.2
mu_T = Beta*h*h # Stabilization coefficient
a = inner(grad(u), grad(v)) * dx
a -= inner(p, div(v)) * dx
a += inner(div(u), q) * dx
a += mu_T*inner(grad(p), grad(q)) * dx # Stabilization term

L = inner(f,v) * dx
L -= mu_T * inner(f, grad(q)) * dx # Stabilization  term
'''
For more infromation about stabilizing stokes flow, the papers "Grad-Div Stabilization for Stokes Equations" by Maxim A. Olshanskii
and "On the parameter of choice in grad-div stabilization for the stokes equations by Eleanir W. Jenkins are recommended
'''



# ------ Assemble LHS matrix and RHS vector and solve-------
print("Start Assembling Stiffness Matrix and Forcing Vector")
from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs = bcs, petsc_options={'ksp_type': 'preonly', 'pc_type':'lu'})
U = Function(W)
U = problem.solve() # Solve the problem



# ------ Split the mixed solution and collapse ------
u, p = U.sub(0).collapse(), U.sub(1).collapse()

# Compute norms
norm_u, norm_p = la.norm(u.x), la.norm(p.x)
norm_inf_u, inf_norm_p = la.norm(u.x, type=la.Norm.linf), la.norm(p.x, type=la.Norm.linf)
if MPI.COMM_WORLD.rank == 0:
    print(f"L2 Norm of velocity coefficient vector: {norm_u}")
    print(f"Infinite Norm of velocity coefficient vector: {norm_inf_u}")
    print(f"\nL2 Norm of pressure coefficient vector: {norm_p}")
    print(f"Infinite Norm of pressure coefficient vector: {inf_norm_p}")

print("Finished Solving, Saving Solution Field")



# ------ Save the solutions to both a .xdmf and .h5 file
# Save the pressure field
from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement
with XDMFFile(MPI.COMM_WORLD, "StokesChannelPressure.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    P3 = VectorElement("Lagrange", mesh.basix_cell(), 1)
    u1 = Function(functionspace(mesh, P3))
    u1.interpolate(p)
    u1.name = 'Pressure'
    pfile_xdmf.write_mesh(mesh)
    pfile_xdmf.write_function(u1)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, "StokesChannelVelocity.xdmf", "w") as pfile_xdmf:
    u.x.scatter_forward()
    P4 = VectorElement("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    u2 = Function(functionspace(mesh, P4))
    u2.interpolate(u)
    u2.name = 'Velocity'
    pfile_xdmf.write_mesh(mesh)
    pfile_xdmf.write_function(u2)