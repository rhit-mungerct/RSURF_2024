import sys
import numpy as np
import math
import os
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

mesh = meshgen()

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
dofs = locate_dofs_topological((W0, Q), 2, ft.find(outlet))
bc_outlet = dirichletbc(outlet_pressure, dofs, W0)
