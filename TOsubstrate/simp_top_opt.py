# simple version of compliance based topology optimization
# attempt to introduce a fixed substrate

import numpy as np
import dolfinx
from dolfinx import fem, cpp, io, mesh
import dolfinx.fem.petsc
from mpi4py import MPI
from petsc4py import PETSc
import ufl

# Parameters
NITER_TOTAL = 50
PENAL=3
LAMBDA=0.6
MU=0.4
L, W, H = 0.303, 0.19, 0.064 # dimensions along axes
EMP=0.06
AXE=0.025
Nx, Ny, Nz = 60, 38, 12 # number of elements along axes

# Mesh
points = [np.array([0, 0, 0]), np.array([L, W, H])]
domain = mesh.create_box(MPI.COMM_WORLD,points,[Nx,Ny,Nz],cell_type=cpp.mesh.CellType.hexahedron,ghost_mode=mesh.GhostMode.shared_facet)
dim = domain.topology.dim
    
def left(x):
    return np.isclose(x[0], 0, atol=1e-6)
    
def right(x):
    return np.isclose(x[0], L, atol=1e-6) & np.greater(x[1], 0.08) & np.less(x[1], 0.12) & np.greater(x[2], 0.016) & np.less(x[2], 0.048)

right_facets = mesh.locate_entities(domain, dim-1, right)
right_indices = np.array(np.hstack([right_facets]), dtype=np.int32)
sorted_right_facets = np.argsort(right_indices)

right_markers = np.full(len(right_facets), 1, dtype=np.int32)        

right_tag = mesh.meshtags(domain, dim-1, right_indices[sorted_right_facets], right_markers[sorted_right_facets])

domain.topology.create_connectivity(dim-1, dim)

# Displacement variable
U = fem.functionspace(domain, ("CG", 1, (dim,)))
u = fem.Function(U, name="Displacement")

# boundary conditions
traction = fem.Constant(domain, np.array([0.0, 0.0, -100.0], dtype=PETSc.ScalarType))
bcdofs = fem.locate_dofs_geometrical(U, left)
u_0 = np.array((0,)*dim, dtype=PETSc.ScalarType)
bc = fem.dirichletbc(u_0, bcdofs, U)

ds = ufl.Measure("ds", domain=domain, subdomain_data=right_tag)
surface_area = W * H

def eps(v): # strain
    return ufl.sym(ufl.grad(v))
def sigma(v): # stress
    return LAMBDA*ufl.tr(eps(v))*ufl.Identity(dim)+2*MU*eps(v)
def dE(u, v): # energy
    return ufl.inner(sigma(u), eps(v))/2

volume_constraint = 0.7

# Density variable
T=fem.functionspace(domain, ("DG", 0))
density = fem.Function(T, name="Density")
def homogenous(x):
    return np.full(len(x[0]), volume_constraint) # initial volume fraction
density.interpolate(homogenous)

VOL = L * W * H

# Static equilibrium variational problem
u_ = ufl.TestFunction(U)
du = ufl.TrialFunction(U)
a = ufl.inner(density**PENAL*sigma(u_), eps(du))*ufl.dx
b = ufl.dot(traction, u_) * ds(1)

solver=PETSc.KSP().create(MPI.COMM_WORLD)
solver.setFromOptions()

B=fem.petsc.assemble_vector(fem.form(b))
fem.petsc.apply_lifting(B, [fem.form(a)], bcs=[[bc]])
B.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(B, [bc])

# Optimisation loop
for i in range(NITER_TOTAL):

    #SOLVE
    A=fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()
    solver.setOperators(A)
    
    iter_solve=solver.solve(B, u.x.petsc_vec)
    u.x.scatter_forward()

    #OPTIMISATION CRITERIA
    sensitivity=-PENAL*density**(PENAL-1)*dE(u,u)

    #LAGRANGE MULTIPLIER FINDER (DICHOTOMY)
    lmin, lmax=0, 1e8
    counter=0
    frac=0
    while np.abs(frac - volume_constraint) > 1e-4:
        lmoy=(lmin+lmax)/2
        densitynew=fem.Function(T)
        densitynew.interpolate(fem.Expression((-sensitivity/lmoy)**0.5, T.element.interpolation_points()))
        denv=density.x.petsc_vec.getArray()
        denvnew=densitynew.x.petsc_vec.getArray()
        criterion=np.maximum(0.001,
                             np.maximum(denv-np.full(len(denv), 0.2),
                                        np.minimum(1,
                                                   np.minimum(
                                                     denv+np.full(len(denv), 0.2),
                                                     denv*denvnew
                                                             )
                                                   )
                                       )         
                            )
        densitynew.x.petsc_vec.setArray(criterion)
        densitynew.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, 
                                      mode=PETSc.ScatterMode.REVERSE)
        frac=MPI.COMM_WORLD.allreduce(
                 fem.assemble_scalar(fem.form(densitynew*ufl.dx))/VOL, 
                 op=MPI.SUM
                                     )

        if  frac-volume_constraint>0:
            lmin=lmoy
        else:
            lmax=lmoy

        counter+=1
    
    density.x.petsc_vec.setArray(densitynew.x.petsc_vec.getArray())
    density.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, 
                               mode=PETSc.ScatterMode.REVERSE)

    compliance=MPI.COMM_WORLD.allreduce(
                   fem.assemble_scalar(fem.form(ufl.action(b,u))), 
                   op=MPI.SUM
                                       )

    print("\n ***** Step {}: compliance=".format(i), compliance, 
          " in : ", counter, "iterations", 
          ", process #", domain.comm.rank, " ***** ")

xdmf = dolfinx.io.XDMFFile(domain.comm, "optimized_design.xdmf", "w")
xdmf.write_mesh(domain)
densitynew.name = "density"
xdmf.write_function(densitynew)
