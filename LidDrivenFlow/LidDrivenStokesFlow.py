from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, la
from dolfinx.fem import (Constant, Function, dirichletbc,
                         extract_function_spaces, form, functionspace,
                         locate_dofs_topological)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner

# Create mesh
msh = create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                       [128, 128], CellType.triangle)


# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
                         np.isclose(x[1], 0.0))


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)

# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

P2 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
P1 = element("Lagrange", msh.basix_cell(), 1)
V, Q = functionspace(msh, P2), functionspace(msh, P1)

def mixed_direct():

    # Create the Taylot-Hood function space
    TH = mixed_element([P2, P1])
    W = functionspace(msh, TH)

    # No slip boundary condition
    W0, _ = W.sub(0).collapse()
    noslip = Function(W0)
    facets = locate_entities_boundary(msh, 1, noslip_boundary)
    dofs = locate_dofs_topological((W.sub(0), W0), 1, facets)
    bc0 = dirichletbc(noslip, dofs, W.sub(0))

    # Driving velocity condition u = (1, 0) on top boundary (y = 1)
    lid_velocity = Function(W0)
    lid_velocity.interpolate(lid_velocity_expression)
    facets = locate_entities_boundary(msh, 1, lid)
    dofs = locate_dofs_topological((W.sub(0), W0), 1, facets)
    bc1 = dirichletbc(lid_velocity, dofs, W.sub(0))

    # Collect Dirichlet boundary conditions
    bcs = [bc0, bc1]

    # Define variational problem
    h = ufl.CellDiameter(msh)
    Beta = 0.2
    mu_T = Beta*h*h # Stabilization coefficient
    # mu_T = 0
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = Function(W0)
    a = form((inner(grad(u), grad(v)) - inner(p, div(v)) + inner(div(u), q) + mu_T*inner(grad(p), grad(q))) * dx)
    L = form((inner(f, v) - mu_T * inner(f, grad(q))) * dx)

    # Assemble LHS matrix and RHS vector
    A = fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()
    b = fem.petsc.assemble_vector(L)

    fem.petsc.apply_lifting(b, [a], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Set Dirichlet boundary condition values in the RHS
    fem.petsc.set_bc(b, bcs)

    # Create and configure solver
    ksp = PETSc.KSP().create(msh.comm)
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
        ksp.solve(b, U.vector)
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

    return norm_u, norm_u, u, p

# Solve using a non-blocked matrix and an LU solver
norm_u, norm_p, u, p = mixed_direct()

# ------ Save the solutions to both a .xdmf and .h5 file
# Save the pressure field
from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement
with XDMFFile(MPI.COMM_WORLD, "LidDrivenStokesFlowPressureStabilized.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    P3 = VectorElement("Lagrange", msh.basix_cell(), 1)
    u1 = Function(functionspace(msh, P3))
    u1.interpolate(p)
    u1.name = 'Pressure'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u1)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, "LidDrivenStokesFlowVelocityStabilized.xdmf", "w") as pfile_xdmf:
    u.x.scatter_forward()
    P4 = VectorElement("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u2 = Function(functionspace(msh, P4))
    u2.interpolate(u)
    u2.name = 'Velocity'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u2)
