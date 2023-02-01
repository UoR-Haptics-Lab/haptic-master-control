import numpy as np
import numpy.linalg as la
import numpy.polynomial as npp
import scipy.linalg as sla
from typing import Tuple


class Trajectory:
    """
    A class for representing smooth straight line trajectories

    ...

    Attributes
    ----------
    t0 : float
        Starting time for the trajectory
    t1 : float
        Finishing time for the trajectory
    T0 : numpy.ndarray
        Configuration at the start
    T1 : numpy.ndarray
        Configuration at the finish
    
    Methods
    -------
    evaluate(t):
        Evaluate the trajectory for a given time
    """
    def __init__(self, t0: float=0.0, t1: float=1.0, T0: np.ndarray=np.eye(4), T1: np.ndarray=np.eye(4)) -> None:
        """
        Construct all necessary attributes for the trajectory

        Parameters
        ----------
        t0 : float
            Starting time for the trajectory
        t1 : float
            Finishing time for the trajectory
        T0 : numpy.ndarray
            Configuration at the start
        T1 : numpy.ndarray
            Configuration at the finish
        s : numpy.polynomial.Polynomial
            Time scaling polynomial with the mapping [t0, t1] -> [0.0, 1.0]
        sp : numpy.polynomial.Polynomial
            Time derivative of the time scaling polynomial s
        """
        self.t0 = t0
        self.t1 = t1
        self.T0 = T0
        self.T1 = T1
        
        # Time scaling polynomial
        self.s = npp.Polynomial(self._polynomial_coefficients())

        # The derivative of the time scaling polynomial
        self.sp = self.s.deriv()
    
    def _polynomial_coefficients(self) -> np.ndarray:
        """
        Calculate the coeffcients of the time scaling polynomial. Third order
        polynomial is implemented.

        Returns
        -------
        coefficients (numpy.ndarray): Array of polynomial coefficients in
                                      increasing order.
        """
        A = np.array([[1.0, self.t0, self.t0**2,    self.t0**3],
                      [0.0, 1.0,     2.0 * self.t0, 3.0 * self.t0**2],
                      [1.0, self.t1, self.t1**2,    self.t1**3],
                      [0.0, 1.0,     2.0 * self.t1, 3.0 * self.t1**2]])

        return la.inv(A) @ np.array([0.0, 0.0, 1.0, 0.0])

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the trajectory for a given time.

        Parameters
        ----------
        t : float
            Time for which the trajectory will be evaluated
        
        Returns
        -------
        Position, Velocity (numpy.ndarray, numpy.ndarray): 
            Configuration and screw on the trajectory for a given time.
        """
        return (self.T0 @ sla.expm(sla.logm(la.inv(self.T0) @ self.T1) * self.s(t)),
                self.T0 @ sla.logm(la.inv(self.T0) @ self.T1) @ sla.expm(sla.logm(la.inv(self.T0) @ self.T1) * self.s(t)) * self.sp(t))


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    print(Trajectory.__doc__)    
    t0 = 0.0
    t1 = 10.0
    
    T0 = np.eye(4)
    T1 = np.eye(4)
    T1[0, 3] = 3.0
    
    Tr = Trajectory(t0, t1, T0, T1)

    time = np.linspace(0.0, 10.0, 101)

    T = []
    Tp = []
    for t in time:
        T.append(Tr.evaluate(t)[0])
        Tp.append(Tr.evaluate(t)[1])
    
    fig, ax = plt.subplots()
    ax.plot(time, [Ti[0, 3] for Ti in T])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position x-axis [m]')
    fig.savefig('position_x.pdf')

    fig, ax = plt.subplots()
    ax.plot(time, [Tpi[0, 3] for Tpi in Tp])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity x-axis [m/s]')
    fig.savefig('velocity_x.pdf')