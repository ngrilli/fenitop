"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

- Nicolò Grilli adapted to FEniCSx version 0.8 and introduced thermal strain

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

import numpy as np
import ufl
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import (FunctionSpace, Function, Constant,
                         dirichletbc, locate_dofs_topological)
from dolfinx.fem import (Expression, form, functionspace) 

from fenitop.utility import create_mechanism_vectors
from fenitop.utility import LinearProblem


def form_fem(fem, opt):
    """Form an FEA problem."""
    # Function spaces and functions
    mesh = fem["mesh"]
    V = functionspace(mesh, ("CG", 1, (mesh.geometry.dim,)))
    S0 = functionspace(mesh, ("DG", 0))
    S = functionspace(mesh, ("CG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    u_field = Function(V)  # Displacement field
    lambda_field = Function(V)  # Adjoint variable field
    rho_field = Function(S0)  # Density field
    rho_phys_field = Function(S)  # Physical density field
    
    x = ufl.SpatialCoordinate(mesh)
    T = Function(S) # temperature field
    T_ex = 293.0 + 0.1*ufl.exp(-((x[0]-30.0)**2)/25)
    expr = Expression(T_ex, V.element.interpolation_points())
    T.interpolate(expr)

    # Material interpolation
    E0, nu = fem["young's modulus"], fem["poisson's ratio"]
    volumetric_thermal_expansion = fem["volumetric thermal expansion"]
    reference_temperature = fem["reference temperature"]
    p, eps = opt["penalty"], opt["epsilon"]
    E = (eps + (1-eps)*rho_phys_field**p) * E0
    _lambda, mu = E*nu/(1+nu)/(1-2*nu), E/(2*(1+nu))  # Lame constants

    # Kinematics
    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):  # 3D or plane strain
        return 2*mu*epsilon(u) + _lambda*ufl.tr(epsilon(u))*ufl.Identity(len(u))   
        
    def sigma_thermal(): # thermal stress independent from u goes into the rhs
        return volumetric_thermal_expansion * (3.0 * _lambda + 2.0 * mu) * (T - reference_temperature) * ufl.Identity(2)
        
    # Boundary conditions
    dim = mesh.topology.dim
    fdim = dim - 1
    disp_facets = locate_entities_boundary(mesh, fdim, fem["disp_bc"])
    bc = dirichletbc(Constant(mesh, np.full(dim, 0.0)),
                     locate_dofs_topological(V, fdim, disp_facets), V)

    tractions, facets, markers = [], [], []
    for marker, (traction, traction_bc) in enumerate(fem["traction_bcs"]):
        tractions.append(Constant(mesh, np.array(traction, dtype=float)))
        current_facets = locate_entities_boundary(mesh, fdim, traction_bc)
        facets.extend(current_facets)
        markers.extend([marker,]*len(current_facets))
    facets = np.array(facets, dtype=np.int32)
    markers = np.array(markers, dtype=np.int32)
    _, unique_indices = np.unique(facets, return_index=True)
    facets, markers = facets[unique_indices], markers[unique_indices]
    sorted_indices = np.argsort(facets)
    facet_tags = meshtags(mesh, fdim, facets[sorted_indices], markers[sorted_indices])

    metadata = {"quadrature_degree": fem["quadrature_degree"]}
    dx = ufl.Measure("dx", metadata=metadata)
    ds = ufl.Measure("ds", domain=mesh, metadata=metadata, subdomain_data=facet_tags)
    b = Constant(mesh, np.array(fem["body_force"], dtype=float))

    # Establish the equilibrium and adjoint equations
    lhs = ufl.inner(sigma(u), epsilon(v))*dx
    rhs = ufl.dot(b, v)*dx + ufl.inner(sigma_thermal(), epsilon(v))*dx # added thermal part here
    for marker, t in enumerate(tractions):
        rhs += ufl.dot(t, v)*ds(marker)
    if opt["opt_compliance"]:
        spring_vec = opt["l_vec"] = None
    else:
        spring_vec, opt["l_vec"] = create_mechanism_vectors(
            V, opt["in_spring"], opt["out_spring"])
    linear_problem = LinearProblem(u_field, lambda_field, lhs, rhs, opt["l_vec"],
                                   spring_vec, [bc], fem["petsc_options"])

    # Define optimization-related variables
    # thermal part added to compliance does not converge
    opt["f_int"] = ufl.inner(sigma(u_field), epsilon(v))*dx # -sigma_thermal()
    opt["compliance"] = ufl.inner(sigma(u_field), epsilon(u_field))*dx
    opt["volume"] = rho_phys_field*dx
    opt["total_volume"] = Constant(mesh, 1.0)*dx

    return linear_problem, u_field, lambda_field, rho_field, rho_phys_field
