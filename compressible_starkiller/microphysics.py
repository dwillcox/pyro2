import numpy as np

from compressible import Variables
from compressible_react.burning import Network

import StarKillerMicrophysics

class StarKiller(object):
    def __init__(self):
        self.eos = StarKillerEOS()
        self.network = StarKillerNetwork()

    def initialize(self, rp):
        probin = ""
        try:
            probin = rp.get_param("network.probin")
        except:
            raise ValueError("ERROR: StarKiller Microphysics requires a probin file via the parameter starkiller.probin.")
        StarKillerMicrophysics.starkiller_initialization_module.starkiller_initialize(probin)
        self.eos.initialize()
        self.network.initialize()
        self.network.eos = self.eos
        self.eos.network = self.network

class StarKillerEOS(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eos_module = None
        self.eos_type_module = None
        self.eos_input_rp = None
        self.eos_input_re = None
        self.network = None

    def initialize(self):
        self.eos_module = StarKillerMicrophysics.Eos_Module()
        self.eos_type_module = StarKillerMicrophysics.Eos_Type_Module()
        self.eos_input_rp = self.eos_type_module.eos_input_rp
        self.eos_input_re = self.eos_type_module.eos_input_re

    def evaluate(self, grid, array_out, array_rho, array_pe, array_xn, input_type):
        # Evaluate the eos with specified input type
        # if input_type == eos_input_rp, return rho*e in array_out
        # if input_type == eos_input_re, return p in array_out

        try:
            assert(input_type == self.eos_input_rp or
                   input_type == self.eos_input_re)
        except:
            raise ValueError("ERROR: EOS input_type must be eos_input_rp or eos_input_re")
        
        eos_state = self.eos_type_module.eos_t()
        for i in range(grid.qx):
            for j in range(grid.qy):
                if (input_type == self.eos_input_rp):
                    eos_state.rho = array_rho[i, j]
                    eos_state.p = array_pe[i, j]
                    eos_state.xn = array_xn[i, j, :]
                elif (input_type == self.eos_input_re):
                    eos_state.rho = array_rho[i, j]
                    eos_state.e = array_pe[i, j]
                    eos_state.xn = array_xn[i, j, :]/eos_state.rho
                self.eos_module.eos(input_type, eos_state)
                if (input_type == self.eos_input_rp):
                    array_out[i, j] = eos_state.e * eos_state.rho
                elif (input_type == self.eos_input_re):
                    array_out[i, j] = eos_state.p

class StarKillerNetwork(Network):
    r"""
    Network from StarKiller Microphysics.
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor

        Parameters
        ----------
        nspec_evolve : int
            Number of species to evolve.
        """

        # Initialize Network parent class
        super().__init__(*args, **kwargs)

        self.network_module = None
        self.actual_network_module = None        
        self.burner_module = None
        self.burn_type_module = None
        self.eos = None


    def initialize(self):
        self.network_module = StarKillerMicrophysics.Network()
        self.actual_network_module = StarKillerMicrophysics.Actual_Network()        
        self.burner_module = StarKillerMicrophysics.Actual_Burner_Module()
        self.burn_type_module = StarKillerMicrophysics.Burn_Type_Module()

        self.nspec = self.actual_network_module.nspec
        self.nspec_evolve = self.actual_network_module.nspec_evolve
        self.naux = self.actual_network_module.naux

        self.A_ion = self.actual_network_module.aion
        self.Z_ion = self.actual_network_module.zion
        self.E_binding = self.actual_network_module.bion

        self.spec_names = []
        for i in range(1, self.nspec+1):
            iname = self.network_module.get_network_species_name(i)
            self.spec_names.append(iname.decode('ascii').strip())

    def react(self, cc_data, dt):
        r"""
        React the network given the data on the grid for timestep dt.
        
        Parameters
        ----------
        cc_data : CellCenterData2d
            The cell-centered data
        """

        myg = cc_data.grid

        ivars = Variables(cc_data)

        dens = cc_data.get_var("density")
        eint = cc_data.get_var("eint")
        ener = cc_data.get_var("energy")
        rhox = cc_data.data[:, :, ivars.irhox:ivars.irhox+self.nspec]

        burn_state_in = self.burn_type_module.burn_t()
        burn_state_out = self.burn_type_module.burn_t()
        eos_state_in = self.eos.eos_type_module.eos_t()

        for i in range(myg.qx):
            for j in range(myg.qy):
                # Form initial state
                eos_state_in.rho = dens[i, j]
                eos_state_in.e = eint[i, j]
                eos_state_in.xn = rhox[i, j, :]/eos_state_in.rho

                # Call EOS to initialize T
                self.eos.eos_module.eos(self.eos.eos_input_re, eos_state_in)
                self.burn_type_module.eos_to_burn(eos_state_in, burn_state_in)
                self.burn_type_module.eos_to_burn(eos_state_in, burn_state_out)

                # Call burner for timestep dt
                self.burner_module.actual_burner(burn_state_in, burn_state_out, dt, 0.0)

                # Store final state
                ener[i, j] += burn_state_out.e * burn_state_out.rho
                for n in range(self.nspec):
                    rhox[i, j, n] = burn_state_out.rho * burn_state_out.xn[n]

def k_th(cc_data, temp, const, constant=1):
    """
    Conductivity

    If constant, just returns the constant defined in the params file.

    Otherwise, it uses the formula for conductivity with constant opacity in the Castro/Microphysics library.

    Parameters
    ----------
    cc_data : CellCenterData2d
        the cell centered data
    temp : ArrayIndexer
        temperature
    const : float
        the diffusion constant or opacity
    constant : int
        Is the conductivity constant (1) or the opacity constant (0)
    """
    myg = cc_data.grid
    k = myg.scratch_array()

    if constant == 1:
        k[:, :] = const
    else:
        sigma_SB = 5.6704e-5  # Stefan-Boltzmann in cgs
        dens = cc_data.get_var("density")
        k[:, :] = (16 * sigma_SB * temp**3) / (3 * const * dens)

    return k
