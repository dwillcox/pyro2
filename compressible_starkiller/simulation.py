from __future__ import print_function

import importlib

import matplotlib.pyplot as plt
import numpy as np

from simulation_null import grid_setup, bc_setup

import compressible
from compressible import Variables
import compressible.BC as BC
import compressible.derives as derives
import mesh.boundary as bnd
import compressible_starkiller.microphysics as microphysics

import util.plot_tools as plot_tools
from util import msg


class Simulation(compressible.Simulation):

    def initialize(self, extra_vars=None, ng=4):
        """
        Initialize the grid and variables for compressible flow and set
        the initial conditions for the chosen problem.

        For the reacting compressible solver, our initialization of
        the data is the same as the compressible solver, but we
        supply additional variables.
        """

        network = self.rp.get_param("network.network_type")

        if network == "StarKiller":
            self.microphysics = microphysics.StarKiller()
        else:
            msg.fail("ERROR: compressible_starkiller requires the StarKiller network")

        self.microphysics.initialize(self.rp)
        extra_vars = self.microphysics.network.spec_names
        
        my_grid = grid_setup(self.rp, ng=ng)
        my_data = self.data_class(my_grid)

        # define solver specific boundary condition routines
        bnd.define_bc("hse", BC.user, is_solid=False)
        # for double mach reflection problem
        bnd.define_bc("ramp", BC.user, is_solid=False)

        bc, bc_xodd, bc_yodd = bc_setup(self.rp)

        # are we dealing with solid boundaries? we'll use these for
        # the Riemann solver
        self.solid = bnd.bc_is_solid(bc)

        # density and energy
        my_data.register_var("density", bc)
        my_data.register_var("energy", bc)
        my_data.register_var("x-momentum", bc_xodd)
        my_data.register_var("y-momentum", bc_yodd)

        # any extras?
        if extra_vars is not None:
            for v in extra_vars:
                my_data.register_var(v, bc)

        # store the EOS gamma as an auxillary quantity so we can have a
        # self-contained object stored in output files to make plots.
        # store grav because we'll need that in some BCs
        my_data.set_aux("gamma", self.rp.get_param("eos.gamma"))
        my_data.set_aux("grav", self.rp.get_param("compressible.grav"))

        my_data.create()

        self.cc_data = my_data

        if self.rp.get_param("particles.do_particles") == 1:
            self.particles = particles.Particles(self.cc_data, bc, self.rp)

        # some auxillary data that we'll need to fill GC in, but isn't
        # really part of the main solution
        aux_data = self.data_class(my_grid)
        aux_data.register_var("ymom_src", bc_yodd)
        aux_data.register_var("E_src", bc)
        aux_data.create()
        self.aux_data = aux_data

        self.ivars = Variables(my_data)

        # derived variables
        self.cc_data.add_derived(derives.derive_primitives)

        # initial conditions for the problem
        problem = importlib.import_module("{}.problems.{}".format(
            self.solver_name, self.problem_name))
        problem.init_data(self.cc_data, self.rp, self.microphysics)

        if self.verbose > 0:
            print(my_data)


    def cons_to_prim(self, U, ivars, myg):
        """ convert an input vector of conserved variables to primitive variables """

        q = myg.scratch_array(nvar=ivars.nq)

        q[:, :, ivars.irho] = U[:, :, ivars.idens]
        q[:, :, ivars.iu] = U[:, :, ivars.ixmom] / U[:, :, ivars.idens]
        q[:, :, ivars.iv] = U[:, :, ivars.iymom] / U[:, :, ivars.idens]

        e = (U[:, :, ivars.iener] -
             0.5 * q[:, :, ivars.irho] * (q[:, :, ivars.iu]**2 +
                                          q[:, :, ivars.iv]**2)) / q[:, :, ivars.irho]

        self.microphysics.eos.evaluate(myg, q[:, :, ivars.ip],
                                       U[:, :, ivars.idens], e,
                                       U[:, :, ivars.irhox:ivars.irhox+ivars.naux],
                                       self.microphysics.eos.eos_input_re)

        if ivars.naux > 0:
            for nq, nu in zip(range(ivars.ix, ivars.ix + ivars.naux),
                              range(ivars.irhox, ivars.irhox + ivars.naux)):
                q[:, :, nq] = U[:, :, nu] / q[:, :, ivars.irho]

        return q


    def prim_to_cons(self, q, ivars, myg):
        """ convert an input vector of primitive variables to conserved variables """

        U = myg.scratch_array(nvar=ivars.nvar)

        U[:, :, ivars.idens] = q[:, :, ivars.irho]
        U[:, :, ivars.ixmom] = q[:, :, ivars.iu] * U[:, :, ivars.idens]
        U[:, :, ivars.iymom] = q[:, :, ivars.iv] * U[:, :, ivars.idens]

        rhoe = myg.scratch_array()
        self.microphysics.eos.evaluate(myg, rhoe,
                                       q[:, :, ivars.irho], q[:, :, ivars.ip],
                                       q[:, :, ivars.ix:ivars.ix+ivars.naux],
                                       self.microphysics.eos.eos_input_rp)

        U[:, :, ivars.iener] = rhoe + \
                               0.5 * q[:, :, ivars.irho] * (q[:, :, ivars.iu]**2 +
                                                            q[:, :, ivars.iv]**2)

        if ivars.naux > 0:
            for nq, nu in zip(range(ivars.ix, ivars.ix + ivars.naux),
                              range(ivars.irhox, ivars.irhox + ivars.naux)):
                U[:, :, nu] = q[:, :, nq] * q[:, :, ivars.irho]

        return U
        

    def burn(self, dt):
        """ react StarKiller network """

        self.microphysics.network.react(self.cc_data, dt)


    def diffuse(self, dt):
        """ diffuse for dt """

        myg = self.cc_data.grid

        dens = self.cc_data.get_var("density")
        e = self.cc_data.get_var("eint")
        ener = self.cc_data.get_var("energy")
        rhox = self.cc_data.data[:, :,
                                 self.ivars.irhox:self.ivars.irhox+self.microphysics.network.nspec]

        # compute T
        temp = myg.scratch_array()

        eos_state = self.microphysics.eos.eos_type_module.eos_t()
        for i in range(myg.qx):
            for j in range(myg.qy):
                eos_state.rho = dens[i, j]
                eos_state.e = e[i, j]
                eos_state.xn = rhox[i, j, :]/eos_state.rho
                self.microphysics.eos.eos_module.eos(self.microphysics.eos.eos_input_re,
                                                     eos_state)
                temp[i, j] = eos_state.t

        # compute div kappa grad T
        k_const = self.rp.get_param("diffusion.k")
        const_opacity = self.rp.get_param("diffusion.constant_kappa")
        k = microphysics.k_th(self.cc_data, temp, k_const, const_opacity)

        div_kappa_grad_T = myg.scratch_array()

        div_kappa_grad_T.v()[:, :] = 0.25 * (
            (k.ip(1) * (temp.ip(2) - temp.v()) -
             k.ip(-1) * (temp.v() - temp.ip(-2))) / myg.dx**2 +
            (k.jp(1) * (temp.jp(2) - temp.v()) -
             k.jp(-1) * (temp.v() - temp.jp(-2))) / myg.dy**2)

        # update energy due to diffusion
        ener[:, :] += div_kappa_grad_T * dt

    def evolve(self):
        """
        Evolve the equations of compressible hydrodynamics through a
        timestep dt.
        """

        # we want to do Strang-splitting here
        self.burn(self.dt / 2)

        self.diffuse(self.dt / 2)

        if self.particles is not None:
            self.particles.update_particles(self.dt / 2)

        # note: this will do the time increment and n increment
        super().evolve()

        if self.particles is not None:
            self.particles.update_particles(self.dt / 2)

        self.diffuse(self.dt / 2)

        self.burn(self.dt / 2)

    def dovis(self):
        """
        Do runtime visualization.
        """

        plt.clf()

        plt.rc("font", size=10)

        # we do this even though ivars is in self, so this works when
        # we are plotting from a file
        ivars = compressible.Variables(self.cc_data)

        # access gamma from the cc_data object so we can use dovis
        # outside of a running simulation.
        gamma = self.cc_data.get_aux("gamma")

        q = self.cons_to_prim(self.cc_data.data, ivars, self.cc_data.grid)

        rho = q[:, :, ivars.irho]
        u = q[:, :, ivars.iu]
        v = q[:, :, ivars.iv]
        p = q[:, :, ivars.ip]
        rhox = q[:, :, ivars.ix:ivars.ix+self.microphysics.network.nspec]
        
        e = self.cc_data.grid.scratch_array()
        eos_state = self.microphysics.eos.eos_type_module.eos_t()
        for i in range(self.cc_data.grid.qx):
            for j in range(self.cc_data.grid.qy):
                eos_state.rho = rho[i, j]
                eos_state.p = p[i, j]
                eos_state.xn = rhox[i, j, :]/eos_state.rho
                self.microphysics.eos.eos_module.eos(self.microphysics.eos.eos_input_rp,
                                                     eos_state)
                e[i, j] = eos_state.e

        Xvals = [q[:, :, idx] for idx in range(ivars.ix, ivars.ix+self.microphysics.network.nspec)]
        Xnames = [r"$\mathrm{X(" + r"{}".format(xn) + r")}$" for xn in self.microphysics.network.spec_names]

        magvel = np.sqrt(u**2 + v**2)

        myg = self.cc_data.grid

        fields = [rho, magvel, u, p, e] +  Xvals
        field_names = [r"$\rho$", r"U", "u", "p", "e"] + Xnames

        f, axes, cbar_title = plot_tools.setup_axes(myg, len(fields))

        for n, ax in enumerate(axes):
            v = fields[n]

            img = ax.imshow(np.transpose(v.v()),
                            interpolation="nearest", origin="lower",
                            extent=[myg.xmin, myg.xmax, myg.ymin, myg.ymax],
                            cmap=self.cm)

            ax.set_xlabel("x")
            ax.set_ylabel("y")

            # needed for PDF rendering
            cb = axes.cbar_axes[n].colorbar(img)
            cb.solids.set_rasterized(True)
            cb.solids.set_edgecolor("face")

            if cbar_title:
                cb.ax.set_title(field_names[n])
            else:
                ax.set_title(field_names[n])

        plt.figtext(0.05, 0.0125, "t = {:10.5g}".format(self.cc_data.t))

        plt.pause(0.001)
        plt.draw()
