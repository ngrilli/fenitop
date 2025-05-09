# standalone version of compliance based topology optimization

#IMPORTS
import numpy as np
import dolfinx
from dolfinx import fem, cpp, io, mesh
import dolfinx.fem.petsc
from mpi4py import MPI
from petsc4py import PETSc
import ufl

#PARAMETERS
NITER_TOTAL=20
PENAL=3
WEIGHT=10
LAMBDA=0.6
MU=0.4
L, W, H=0.303, 0.19, 0.064
EMP=0.06
AXE=0.025
NX, NY, NZ = 60, 38, 12

#MESH CREATION
points=[np.array([0, 0, 0]), np.array([L, W, H])]
domain=mesh.create_box( 
    MPI.COMM_WORLD, 
    points,
    [NX,NY,NZ],
    cell_type=cpp.mesh.CellType.hexahedron,
    ghost_mode=mesh.GhostMode.shared_facet
    )
dim=domain.topology.dim


#BOUNDARY MARKERS
def axes(x):
    return np.logical_and(
               np.logical_or(np.sqrt( (x[0]-EMP)**2 + (x[2]-H/2)**2 ) <=AXE,
                             np.sqrt( (x[0]-L+EMP)**2 + (x[2]-H/2)**2 ) <=AXE),
               np.logical_or(np.isclose(x[1], 0, atol=1e-6), 
                             np.isclose(x[1], W, atol=1e-6))
           )
def bottom(x):
    return np.isclose(x[2], 0, atol=1e-6)
    
def left(x):
    return np.isclose(x[0], 0, atol=1e-6)
    
def right(x):
    return np.isclose(x[0], L, atol=1e-6)

bottom_facets=mesh.locate_entities(domain, dim-1, bottom)
axes_facets=mesh.locate_entities(domain, dim-1, axes)

left_facets = mesh.locate_entities(domain, dim-1, left)
left_indices = np.array(np.hstack([left_facets]), dtype=np.int32)
sorted_left_facets = np.argsort(left_indices)

right_facets = mesh.locate_entities(domain, dim-1, right)
right_indices = np.array(np.hstack([right_facets]), dtype=np.int32)
sorted_right_facets = np.argsort(right_indices)

print(right_indices)

print(sorted_right_facets)

facet_indices=np.array(np.hstack([bottom_facets, axes_facets]), dtype=np.int32)
facet_markers=[np.full(len(bottom_facets), 1, dtype=np.int32), 
               np.full(len(axes_facets), 2, dtype=np.int32)]
               
left_markers = [np.full(len(left_facets), 3, dtype=np.int32)]
left_markers = np.hstack(left_markers) 

right_markers = [np.full(len(right_facets), 4, dtype=np.int32)]
right_markers = np.hstack(right_markers)          

facet_markers=np.hstack(facet_markers)
sorted_facets=np.argsort(facet_indices)
facet_tag=mesh.meshtags(
    domain, 
    dim-1, 
    facet_indices[sorted_facets],
    facet_markers[sorted_facets]
    )

left_tag = mesh.meshtags(domain, dim-1, left_indices[sorted_left_facets], left_markers[sorted_left_facets])

right_tag = mesh.meshtags(domain, dim-1, right_indices[sorted_right_facets], right_markers[sorted_right_facets])

domain.topology.create_connectivity(dim-1, dim)#not sure about this in parallel

#LOADS AND BC
U=fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,)))
#U=fem.VectorFunctionSpace(domain, ("CG", 1))
u=fem.Function(U, name="Déplacement")

#bcdofs=fem.locate_dofs_geometrical(U, axes)
bcdofs=fem.locate_dofs_geometrical(U, left)
u_0=np.array((0,)*dim, dtype=PETSc.ScalarType)
bc=fem.dirichletbc(u_0, bcdofs, U)

#ds= ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
#ds= ufl.Measure("ds", domain=domain, subdomain_data=left_tag)
ds= ufl.Measure("ds", domain=domain, subdomain_data=right_tag)
surface_area=MPI.COMM_WORLD.allreduce(
                fem.assemble_scalar(fem.form(fem.Constant(domain, 1.)*ds(4))), # ds(4) means surface wit tag 4, which is right surface
                op=MPI.SUM
                                    )

print(surface_area)

def eps(v):
    return ufl.sym(ufl.grad(v))
def sigma(v):
    return (LAMBDA*ufl.nabla_div(v)*ufl.Identity(dim)+2*MU*eps(v))
def dE(u, v):
    return ufl.inner(sigma(u), eps(v))/2

#DENSITY SET UP
T=fem.functionspace(domain, ("DG", 0))

density_old=fem.Function(T)
density=fem.Function(T, name="Density")
def homogenous(x):
    return np.full(len(x[0]), 0.8) # initial volume fraction
density.interpolate(homogenous)

VOL=MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(fem.form(fem.Constant(domain, 1.)*ufl.dx)),
        op=MPI.SUM
                            )
frac_init=MPI.COMM_WORLD.allreduce(
              fem.assemble_scalar(fem.form(density*ufl.dx))/VOL, 
              op=MPI.SUM
                                  )

#VARIATIONAL PROBLEM
u_=ufl.TestFunction(U)
du=ufl.TrialFunction(U)
a=ufl.inner(density**PENAL*sigma(u_), eps(du))*ufl.dx
b=ufl.dot((WEIGHT/surface_area)*ufl.FacetNormal(domain), u_)*ds(4)

solver=PETSc.KSP().create(MPI.COMM_WORLD)
OPTIONS=PETSc.Options()

#DOES NOT WORK IN PARALLEL
#OPTIONS["ksp_type"]="cg"
#OPTIONS["ksp_atol"]=1.0e-10
#OPTIONS["pc_type"]="ilu"

solver.setFromOptions()
compliance_history=[]
#compliance_old=1e30

B=fem.petsc.assemble_vector(fem.form(b))
fem.petsc.apply_lifting(B, [fem.form(a)], bcs=[[bc]])
B.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(B, [bc])

#OPTIMISATION LOOP
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
    while np.abs(frac-frac_init)>1e-4:
        lmoy=(lmin+lmax)/2
        densitynew=fem.Function(T)
        densitynew.interpolate(fem.Expression((-sensitivity/lmoy)**0.5, 
                                              T.element.interpolation_points()))
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

        if  frac-frac_init>0:
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
    compliance_history.append(compliance)

    print("\n ***** Step {}: compliance=".format(i), compliance, 
          " in : ", counter, "iterations", 
          ", process #", domain.comm.rank, " ***** ")


print("ALL CLEAR PROCESS #", MPI.COMM_WORLD.rank)


xdmf = dolfinx.io.XDMFFile(domain.comm, "optimized_design.xdmf", "w")
xdmf.write_mesh(domain)
densitynew.name = "density"
xdmf.write_function(densitynew)
