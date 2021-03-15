import copy
import machupX as mx
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


class TwistOptimizer:
    """Uses MachUpX and scipy.optimize to optimize the twist distribution on a given wing for minimum induced drag.

    Parameters
    ----------
    scene_input : dict
        Scene input dictionary for MachUpX. This specifies the atmospheric conditions, solver parameters, etc.

    wing_input : dict
        Wing input dictionary for MachUpX. This specifies the geometry of the wing. May have multiple wing sections. The section on which the twist is to be optimized should be called "design_section".

    wing_state : dict
        Wing state dictionary.
    """

    def __init__(self, scene_input, wing_input, wing_state):

        # Store
        self._scene_input = scene_input
        self._wing_input = wing_input
        self._wing_state = wing_state
        try:
            self._wing_input["wings"]["design_section"]
        except KeyError:
            raise IOError("The wing must have a section named 'design_section'.")


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
        twist : ndarray
            Array of twist values in degrees which minimize the induced drag.

        s : ndarray
            Array of span coordinates corresponding to the locations of each twist value.
        """

        # Store
        self._N = N_twist_stations
        self._CL = CL

        # Optimize
        twist0 = np.zeros(self._N)
        result = opt.minimize(self._get_induced_drag, twist0, method='SLSQP', options={'eps' : 0.1, 'disp' : True})
        twist = result.x

        # Determine span array
        s = np.linspace(0.0, 1.0, self._N)

        return twist, s


    def _get_induced_drag(self, twist):
        # Calculates the induced drag for the given twist distribution

        # Determine twist array
        s = np.linspace(0.0, 1.0, self._N)
        twist_array = np.concatenate((s[:,np.newaxis], twist[:,np.newaxis]), axis=1)

        # Update wing dict
        wing_dict = copy.deepcopy(self._wing_input)
        wing_dict["wings"]["design_section"]["twist"] = twist_array

        # Set up MachUpX
        scene = mx.Scene(self._scene_input)
        scene.add_aircraft("wing", wing_dict, state=self._wing_state)

        # Go to target CL
        self.alpha = scene.target_CL(CL=self._CL, set_state=True)

        # Get drag
        FM = scene.solve_forces(dimensional=False, nondimensional=True, body_frame=False)
        C_D = FM["wing"]["total"]["CD"]
        return C_D*100.0 # You have to amplify the objective function a bit to get the minimizer to want to do anything...