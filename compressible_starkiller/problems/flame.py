from __future__ import print_function

import sys
import mesh.patch as patch
import numpy as np
from util import msg
import math


def init_data(my_data, rp, microphysics):
    """ initialize the sedov problem """

    msg.bold("initializing the sedov problem...")

    # make sure that we are passed a valid patch object
    if not isinstance(my_data, patch.CellCenterData2d):
        print("ERROR: patch invalid in sedov.py")
        print(my_data.__class__)
        sys.exit()

    # get the density, momenta, and energy as separate variables
    dens = my_data.get_var("density")
    xmom = my_data.get_var("x-momentum")
    ymom = my_data.get_var("y-momentum")
    ener = my_data.get_var("energy")
    rhox = [my_data.get_var(xname) for xname in microphysics.network.spec_names]
    
    # initialize the components, remember, that ener here is rho*eint
    # + 0.5*rho*v**2, where eint is the specific internal energy
    # (erg/g)
    dens[:, :] = 1.0e9
    xmom[:, :] = 0.0
    ymom[:, :] = 0.0

    # pure He-4 for aprox13
    for spec_name, spec_rhox in zip(microphysics.network.spec_names, rhox):
        if spec_name == "helium-4":
            spec_rhox[:, :] = dens[:, :]
        else:
            spec_rhox[:, :] = 0.0

    E_sedov = 1.3e18

    r_init = rp.get_param("sedov.r_init")

    pi = math.pi

    xmin = rp.get_param("mesh.xmin")
    xmax = rp.get_param("mesh.xmax")

    ymin = rp.get_param("mesh.ymin")
    ymax = rp.get_param("mesh.ymax")

    xctr = 0.5 * (xmin + xmax)
    yctr = 0.5 * (ymin + ymax)

    # initialize the pressure by putting the explosion energy into a
    # volume of constant pressure.  Then compute the energy in a zone
    # from this.
    nsub = 4

    dist = np.sqrt((my_data.grid.x2d - xctr)**2 +
                   (my_data.grid.y2d - yctr)**2)

    p = 1.e-5

    # set ener from p using the EOS
    eos_state = microphysics.eos.eos_type_module.eos_t()
    for i in range(my_data.grid.qx):
        for j in range(my_data.grid.qy):
            eos_state.rho = dens[i, j]
            eos_state.p = p
            eos_state.xn = np.array([rxn[i, j]/eos_state.rho for rxn in rhox])
            microphysics.eos.eos_module.eos(microphysics.eos.eos_input_rp, eos_state)
            ener[i, j] = eos_state.e

    for i, j in np.transpose(np.nonzero(dist < 2.0 * r_init)):

        pzone = 0.0

        for ii in range(nsub):
            for jj in range(nsub):

                xsub = my_data.grid.xl[i] + \
                    (my_data.grid.dx / nsub) * (ii + 0.5)
                ysub = my_data.grid.yl[j] + \
                    (my_data.grid.dy / nsub) * (jj + 0.5)

                dist = np.sqrt((xsub - xctr)**2 +
                               (ysub - yctr)**2)

                if dist <= r_init:
                    # set p from e using the EOS
                    eos_state.rho = dens[i, j]
                    eos_state.e = E_sedov / (pi * r_init * r_init)
                    eos_state.xn = np.array([rxn[i, j]/eos_state.rho for rxn in rhox])
                    microphysics.eos.eos_module.eos(microphysics.eos.eos_input_re, eos_state)
                    p = eos_state.p
                else:
                    p = 1.e-5

                pzone += p

        p = pzone / (nsub * nsub)

        # set e from p using the EOS
        eos_state.rho = dens[i, j]
        eos_state.p = p
        eos_state.xn = np.array([rxn[i, j]/eos_state.rho for rxn in rhox])
        microphysics.eos.eos_module.eos(microphysics.eos.eos_input_rp, eos_state)
        ener[i, j] = eos_state.e


def finalize():
    """ print out any information to the user at the end of the run """

    msg = """
          The script analysis/sedov_compare.py can be used to analyze these
          results.  That will perform an average at constant radius and
          compare the radial profiles to the exact solution.  Sample exact
          data is provided as analysis/cylindrical-sedov.out
          """

    print(msg)
