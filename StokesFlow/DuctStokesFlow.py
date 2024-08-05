#!/usr/bin/env python

import numpy as np
import math
import sys
import os
import gmsh
from dolfinx import fem, la
from dolfinx.io import gmshio
from mpi4py import MPI
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace, dirichletbc, locate_dofs_topological, Constant, Function, form
from petsc4py import PETSc
import ufl
from ufl import div, dx, grad, inner
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block

gmsh_fname = sys.argv[1]
mesh_lc = float(sys.argv[2])
x_outlet = float(sys.argv[3])
p_idx=1
l_idx=1
loop_idx=1
surf_idx=1

gmsh.initialize()
gmsh.model.add("gmsh_3D_flow")
g = gmsh.model.occ

line_init = l_idx

inlet_surfaces = []
outlet_surfaces = []
wall_surfaces = []

x_inlet = 0.0
lc = mesh_lc

############# CREATE OUTER BOX #############

p_start = p_idx

g.addPoint(0.0, -0.5, -0.5, lc, p_idx)
g.addPoint(0.0, 0.5, -0.5, lc, p_idx+1)
g.addPoint(0.0, 0.5, 0.5, lc, p_idx+2)
g.addPoint(0.0, -0.5, 0.5, lc, p_idx+3)

g.addLine(p_idx, p_idx+1, l_idx)
g.addLine(p_idx+1, p_idx+2, l_idx+1)
g.addLine(p_idx+2, p_idx+3, l_idx+2)
g.addLine(p_idx+3, p_idx, l_idx+3)
g.addCurveLoop(list(range(l_idx, l_idx+4)), loop_idx)

# Inlet
g.addPlaneSurface([loop_idx],surf_idx)
inlet_surfaces.append(surf_idx)

loop_idx += 1    
surf_idx += 1

p_idx += 4
l_idx += 4
loop_idx += 1

p_idx_closure = p_idx + 1

g.addPoint(x_outlet, -0.5, -0.5, lc, p_idx)
g.addPoint(x_outlet, 0.5, -0.5, lc, p_idx+1)
g.addPoint(x_outlet, 0.5, 0.5, lc, p_idx+2)
g.addPoint(x_outlet, -0.5, 0.5, lc, p_idx+3)

g.addLine(p_idx, p_idx+1, l_idx)
g.addLine(p_idx+1, p_idx+2, l_idx+1)
g.addLine(p_idx+2, p_idx+3, l_idx+2)
g.addLine(p_idx+3, p_idx, l_idx+3)
g.addCurveLoop(list(range(l_idx, l_idx+4)), loop_idx)

p_idx += 4
l_idx += 4

# Outlet
g.addPlaneSurface([loop_idx],surf_idx)
outlet_surfaces.append(surf_idx)
loop_idx += 1
surf_idx += 1

# Create walls
print(f'l_idx = {l_idx}')
for i in range(4):
    g.addLine(p_start+i, p_start+i+4, l_idx)
    l_idx += 1

for i in range(1,4):
    print(f'iter {i}')
    line_loop = [i, i+9, i+4, i+8]
    print(line_loop)
    g.addCurveLoop([i, i+8, i+4, i+9], loop_idx)
    print(f'loop_idx = {loop_idx}')
    print(f'surf_idx = {surf_idx}')
    g.addPlaneSurface([loop_idx], surf_idx)
    wall_surfaces.append(surf_idx)
    loop_idx += 2
    surf_idx += 1

l = 4
g.addCurveLoop([l, l+5, l+4, l+8], loop_idx)
g.addPlaneSurface([loop_idx], surf_idx)
wall_surfaces.append(surf_idx)

all_surfaces = inlet_surfaces + wall_surfaces + outlet_surfaces

sl = g.addSurfaceLoop(all_surfaces)
v = g.addVolume([sl])

# g.addPlaneSurface([2],2)
g.synchronize()
inlet_marker, outlet_marker, wall_marker = 3, 4, 5

gmsh.model.addPhysicalGroup(2, inlet_surfaces, inlet_marker)
gmsh.model.setPhysicalName(2, inlet_marker, "inlet")
gmsh.model.addPhysicalGroup(2, outlet_surfaces, outlet_marker)
gmsh.model.setPhysicalName(2, outlet_marker, "outlet")
gmsh.model.addPhysicalGroup(2, wall_surfaces, wall_marker)
gmsh.model.setPhysicalName(2, wall_marker, "wall")


distance_field = gmsh.model.mesh.field.add("Distance")
threshold_field = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", lc)
gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", lc)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", 0)
gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 0)
min_field = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

gmsh.model.addPhysicalGroup(3, [1], name = "fluid")
g.synchronize()
gmsh.model.mesh.generate(3)
gmsh.write(f'{gmsh_fname}.msh')
gmsh.write(f'{gmsh_fname}.geo_unrolled')

mesh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
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
dofs = locate_dofs_topological((W0, Q), 2, ft.find(wall_marker))
bc_wall = dirichletbc(noslip, dofs, W0)

# Set Outlet Pressure to be 0
W0 = W.sub(1)
Q, _ = W0.collapse()
outlet_pressure = Function(Q)
dofs = locate_dofs_topological((W0, Q), 2, ft.find(outlet_marker))
bc_outlet = dirichletbc(outlet_pressure, dofs, W0)

# Inlet velocity Expression
def inlet_velocity_expression(x):
    zero_val = np.zeros(x.shape[1], dtype=np.float64)
    one_val = np.ones(x.shape[1], dtype=np.float64)
    return (one_val, zero_val, zero_val)
# Inlet 1 Velocity Boundary Condition
W0 = W.sub(0)
Q, _ = W0.collapse()
inlet_velocity = Function(Q)
inlet_velocity.interpolate(inlet_velocity_expression)
dofs = fem.locate_dofs_topological((W0, Q), mesh.topology.dim-1, ft.find(inlet_marker))
bc_inlet = dirichletbc(inlet_velocity, dofs, W0)

bcs = [bc_wall, bc_inlet, bc_outlet]

W0 = W.sub(0)
Q, _ = W0.collapse()

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = Function(Q)
a = form((inner(grad(u), grad(v)) + inner(p, div(v)) - inner(div(u), q)) * dx)
L = form(inner(f, v) * dx)

# Assemble LHS matrix and RHS vector
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
norm_inf_u = la.norm(u.x, type=la.Norm.linf)
norm_inf_p = la.norm(p.x, type=la.Norm.linf)
if MPI.COMM_WORLD.rank == 0:
    print(f"L1 norm of velocity coefficient vector: {norm_u}")
    print(f"L1 norm of pressure coefficient vector: {norm_p}")
    print(f"Linf norm of pressure coefficient vector: {norm_inf_u}")
    print(f"Linf norm of pressure coefficient vector: {norm_inf_p}")

# Save the pressure field
from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement
with XDMFFile(MPI.COMM_WORLD, "StokesDuctPressure.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    P3 = VectorElement("Lagrange", mesh.basix_cell(), 1)
    u1 = Function(functionspace(mesh, P3))
    u1.interpolate(p)
    pfile_xdmf.write_mesh(mesh)
    pfile_xdmf.write_function(u1)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, "StokesDuctVelcoity.xdmf", "w") as pfile_xdmf:
    u.x.scatter_forward()
    P4 = VectorElement("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    u2 = Function(functionspace(mesh, P4))
    u2.interpolate(u)
    pfile_xdmf.write_mesh(mesh)
    pfile_xdmf.write_function(u2)