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
NITER_TOTAL = 20
PENAL = 3
LAMBDA = 0.6
MU = 0.4
L, W, H = 0.606, 0.19, 0.064 # dimensions along axes
Nx, Ny, Nz = 120, 38, 12 # number of elements along axes

# Mesh
points = [np.array([0, 0, 0]), np.array([L, W, H])]
domain = mesh.create_box(MPI.COMM_WORLD,points,[Nx,Ny,Nz],cell_type=cpp.mesh.CellType.hexahedron,ghost_mode=mesh.GhostMode.shared_facet)
dim = domain.topology.dim
    
def left(x):
    return np.isclose(x[0], 0, atol=1e-6)
    
def right(x):
    return np.isclose(x[0], L, atol=1e-6) & np.greater(x[1], 0.08) & np.less(x[1], 0.12)

# Substrate function
def is_substrate(x):
    return x[0] <= L / 2

# Substrate mask
num_cells = Nx*Ny*Nz
cell_midpoints = np.zeros((3, num_cells))
for i in range(num_cells):
    cell = domain.topology.connectivity(dim, 0).links(i)
    vertices = domain.geometry.x[cell]
    midpoint = np.mean(vertices, axis=0)
    cell_midpoints[:, i] = midpoint

substrate_mask = cell_midpoints[0, :] <= L / 2  # element centers in the left half

# Right surface for Neumann boundary conditions
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
traction = fem.Constant(domain, np.array([0.0, 0.0, -1000.0], dtype=PETSc.ScalarType))
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

volume_constraint = 0.4

# Density variable
T = fem.functionspace(domain, ("DG", 0))
density = fem.Function(T, name="Density")
def homogenous(x):
    return np.full(len(x[0]), volume_constraint) # initial volume fraction
def homogenous_with_substrate(x):
    res = np.zeros(shape=len(x[0]))
    for i in range(len(x[0])):
        if x[0][i] <= L / 2:
            res[i] = 1.0
        else:
            res[i] = volume_constraint
    return res

density.interpolate(homogenous_with_substrate)

VOL = L * W * H

# Static equilibrium variational problem
u_ = ufl.TestFunction(U)
du = ufl.TrialFunction(U)
a = ufl.inner(density**PENAL*sigma(u_), eps(du)) * ufl.dx
b = ufl.dot(traction, u_) * ds(1)

solver=PETSc.KSP().create(MPI.COMM_WORLD)

B = fem.petsc.assemble_vector(fem.form(b))
fem.petsc.apply_lifting(B, [fem.form(a)], bcs=[[bc]])
B.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(B, [bc])

# Optimisation loop
for i in range(NITER_TOTAL):

    #SOLVE
    A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
    A.assemble()
    solver.setOperators(A)
    
    iter_solve=solver.solve(B, u.x.petsc_vec)
    u.x.scatter_forward()

    # OPTIMISATION CRITERIA
    sensitivity_form = -PENAL * density**(PENAL-1) * dE(u,u)
    
    # part region indices
    design_idx = np.where(~substrate_mask)[0]

    # LAGRANGE MULTIPLIER FINDER (bisection method)
    lmin, lmax = 0, 1e8 # lower and upper bounds for Lagrange multiplier
    counter = 0
    frac = 0
    while np.abs(frac - volume_constraint) > 1e-4:
        lmoy = (lmin + lmax) / 2 # Lagrange multiplier
        
        sensitivity = fem.Function(T)
        sensitivity.interpolate(fem.Expression(sensitivity_form, T.element.interpolation_points()))
        sensv = sensitivity.x.petsc_vec.getArray() # sensitivity vector
        
        denv = density.x.petsc_vec.getArray()
        densitynew = fem.Function(T)
        densitynew.interpolate(fem.Expression((-sensitivity_form/lmoy)**0.5, T.element.interpolation_points())) # Newton-Raphson step
        denvnew = densitynew.x.petsc_vec.getArray()
        
        denvnew2 = np.copy(denv)
        denvnew2[design_idx] = np.clip(denv[design_idx] * (-sensv[design_idx] / lmoy)**0.5, # Newton-Raphson step
                 np.maximum(0.001, denv[design_idx] - 0.2), # lower move limit
                 np.minimum(1.000, denv[design_idx] + 0.2)) # upper move limit

        frac = np.sum(denvnew2[design_idx]) / len(denvnew2[design_idx])

        densitynew.x.petsc_vec.setArray(denvnew2)
        densitynew.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, 
                                      mode=PETSc.ScatterMode.REVERSE)

        if ((frac - volume_constraint) > 0): # increase Lagrange multiplier
            lmin=lmoy
        else: # decrease Lagrange multiplier
            lmax=lmoy

        counter+=1
    
    density.x.petsc_vec.setArray(densitynew.x.petsc_vec.getArray())
    density.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, 
                               mode=PETSc.ScatterMode.REVERSE)

    # quantities for log
    compliance = MPI.COMM_WORLD.allreduce(
                   fem.assemble_scalar(fem.form(ufl.action(b,u))), 
                   op=MPI.SUM)
                   
    total_frac = MPI.COMM_WORLD.allreduce(
                   fem.assemble_scalar(fem.form(densitynew*ufl.dx))/VOL, 
                   op=MPI.SUM)

    print("\n ***** Step {}: compliance=".format(i), compliance, 
          " in : ", counter, "iterations", 
          ", process #", domain.comm.rank, 
          "\n ***** Total fraction: ", total_frac, " ***** ")

xdmf = dolfinx.io.XDMFFile(domain.comm, "optimized_design.xdmf", "w")
xdmf.write_mesh(domain)

densitynew.name = "density"
xdmf.write_function(densitynew)

u.name = "displacement"
xdmf.write_function(u)

xdmf.close()
