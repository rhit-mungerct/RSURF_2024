#!/usr/bin/env python

import numpy as np
import math
import sys
import os
import time
import gmsh
from dolfinx import default_scalar_type
from dolfinx.io import gmshio
from mpi4py import MPI
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace, dirichletbc, locate_dofs_topological, locate_dofs_geometrical, Constant, Function
from petsc4py import PETSc

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
# all_surfaces.extend(wall_surfaces[:])
# all_surfaces.extend(outlet_surfaces)

sl = g.addSurfaceLoop(all_surfaces)
v = g.addVolume([sl])

# g.addPlaneSurface([2],2)
g.synchronize()
inlet_marker, outlet_marker, wall_marker = 3, 4, 5
# gmsh.model.addPhysicalGroup(2, inlet_surfaces, name = "inlet")
# gmsh.model.addPhysicalGroup(2, outlet_surfaces, name = "outlet")
# gmsh.model.addPhysicalGroup(2, wall_surfaces, name = "wall")

gmsh.model.addPhysicalGroup(2, inlet_surfaces, inlet_marker)
gmsh.model.setPhysicalName(2, inlet_marker, "inlet")
gmsh.model.addPhysicalGroup(2, outlet_surfaces, outlet_marker)
gmsh.model.setPhysicalName(2, outlet_marker, "outlet")
gmsh.model.addPhysicalGroup(2, wall_surfaces, wall_marker)
gmsh.model.setPhysicalName(2, wall_marker, "wall")

gmsh.model.addPhysicalGroup(3, [1], name = "fluid")
g.synchronize()
gmsh.model.mesh.generate(3)
# gmsh.write(f'{gmsh_fname}.msh')
# gmsh.write(f'{gmsh_fname}.geo_unrolled')

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

# Set Inlet Pressure
W0 = W.sub(0)
Q, _ = W0.collapse()
# inlet_pressure = Constant(Q,1)
dofs = locate_dofs_topological((W0, Q), 2, ft.find(inlet_marker))
u_bc = np.array((1,0,0), dtype=default_scalar_type)
test = Constant(mesh, default_scalar_type((0, 0, 0)))
bc_outlet = dirichletbc(test, dofs, W0)
