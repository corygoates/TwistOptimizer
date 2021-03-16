import copy
import machupX as mx
import numpy as np
import scipy.optimize as opt


class TwistOptimizer:
    """Uses MachUpX and scipy.optimize to optimize the twist distribution on a given wing for minimum induced drag.

    Parameters
    ----------
    wing_input : dict
        Wing input dictionary for MachUpX. This specifies the geometry of the wing. May have multiple wing sections. The section on which the twist is to be optimized should be called "design_section".

    V : float
        Freestream velocity.

    rho : float
        Atmospheric density.
    """

    def __init__(self, wing_input, V, rho):

        # Store
        self._wing_input = wing_input
        try:
            self._wing_input["wings"]["design_section"]
        except KeyError:
            raise IOError("The wing must have a section named 'design_section'.")

        # Get velocity and density
        self._V = V
        self._rho = rho


    def optimize(self, N_twist_stations, CL):
        """Optimizes the twist.

        Parameters
        ----------
        N_twist_stations : int
            Number of spanwise stations (spaced evenly) per semispan to use in optimizing the twist.

        CL : float
            Lift coefficient at which to minimize drag.

        Returns
        -------
        s : ndarray
            Array of spanwise points at which the twist has been set by the optimizer.

        twist : ndarray
            Twist values in degrees determined by the optimizer to minimize induced drag.
        """

        # Store
        self._N = N_twist_stations
        self._CL = CL

        # Optimize
        twist0 = np.zeros(self._N)
        result = opt.minimize(self._get_induced_drag, twist0, method='SLSQP', options={'eps' : 0.01, 'disp' : True})
        twist = result.x
        self.C_D = result.fun/1000.0

        # Determine span array
        s = np.linspace(0.0, 0.5, self._N)

        return s, twist


    def _get_induced_drag(self, twist):
        # Calculates the induced drag for the given twist distribution

        # Determine twist array
        s = np.linspace(0.0, 1.0, self._N)
        twist_array = np.concatenate((s[:,np.newaxis], twist[:,np.newaxis]), axis=1)

        # Update wing dict
        wing_dict = copy.deepcopy(self._wing_input)
        wing_dict["wings"]["design_section"]["twist"] = twist_array

        # Set up MachUpX
        self._scene = mx.Scene({"scene" : {"atmosphere" : {"density" : self._rho}}})
        self._scene.add_aircraft("wing", wing_dict, state={"velocity" : self._V})

        # Go to target CL
        self.alpha = self._scene.target_CL(CL=self._CL, set_state=True)

        # Get total drag on design section and winglets
        FM = self._scene.solve_forces(dimensional=False, nondimensional=True, body_frame=False)
        C_D = FM["wing"]["total"]["CD"]
        return C_D*1000.0 # Amplifying this objective function helps the optimizer converge


    def get_distributions(self):
        """Returns the resulting lift and load distributions for the optimized wing. These distributions come from MachUpX, not the optimizer, and so have a numch higher resolution.

        Returns
        -------
        s : ndarray
            Array of span coordinates corresponding to the locations of each twist value.

        twist : ndarray
            Array of twist values in degrees which minimize the induced drag.

        lift_dist : ndarray
            Lift distribution.

        load_dist : ndarray
            Load distribution.
        """

        # Get reference params
        Sw, cw, bw = self._scene.get_aircraft_reference_geometry()

        # Get distributions from MachUpX
        dist = self._scene.distributions()
        s = np.array(dist["wing"]["design_section_right"]["span_frac"])*0.5
        dS = np.array(dist["wing"]["design_section_right"]["area"])
        c = np.array(dist["wing"]["design_section_right"]["chord"])
        twist = np.degrees(np.array(dist["wing"]["design_section_right"]["twist"]))
        Fx = np.array(dist["wing"]["design_section_right"]["Fx"])
        Fy = np.array(dist["wing"]["design_section_right"]["Fy"])
        Fz = np.array(dist["wing"]["design_section_right"]["Fz"])
        CL = np.array(dist["wing"]["design_section_right"]["section_CL"])

        # Calculate lift and drag
        a = np.radians(self.alpha)
        L = -Fz*np.cos(a)-Fx*np.sin(a)
        D = -Fx*np.cos(a)-Fz*np.sin(a)

        # Get CL and CD
        non_dim = 0.5*self._rho*self._V*self._V*dS
        #CL = L/non_dim
        CD = D/non_dim

        # Calculate Cn
        Cn = np.sqrt(Fz**2+Fy**2)/non_dim

        # Calculate load distribution
        load = Cn*c/(self._CL*cw)

        return s, twist, CL/self._CL, load