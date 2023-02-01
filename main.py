# from haptic_master.haptic_master import HapticMaster
# from haptic_master.bias_force import BiasForce
from pyHapticMaster.src.haptic_master.haptic_master import HapticMaster
from pyHapticMaster.src.haptic_master.bias_force import BiasForce
import matplotlib.pyplot as plt
import numpy as np
import time
from trajectory import Trajectory


if __name__ == '__main__':
    # IP address and the port number of the robot
    IP = '192.168.0.25'
    PORT = 7654

    # Create HapticMaster instance
    robot = HapticMaster(IP, PORT, 1.0)

    # Trajectory to follow
    # Start and finish times for the trajectory
    t0, t1 = 0.0, 4.0
    # Start and finish configurations on the trajectory
    T0, T1 = np.eye(4), np.eye(4)
    T1[0, 3] = 0.15
    # Create a trajectory instance
    robot_trajectory = Trajectory(t0, t1, T0, T1)

    # Sampling time for the control
    dt = 4000000

    # Number of iterations for the control loop
    NUM_ITER = 100

    # Matrix for storing the robot data with columns
    # position_x, position_y, position_z, velocity_x, velocity_y, velocity_z, force_x, force_y, force_z
    data = np.zeros((NUM_ITER, 9))
    
    # array for storing time
    t = np.zeros(NUM_ITER, dtype=np.int_)

    # Variable to store position, velocity and force measurements
    p, v, f = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    # PD controller parameters
    Kp, Kd = 100, 1

    # Next time step to iterate
    next_timestamp = 0
    
    # Open connection
    robot.connect()

    # Create a bias force to control the robot
    F = BiasForce('myBiasForce')
    robot.create_bias_force(F)

    # Set robot state to position
    robot.set_state('position')

    for i, _ in np.ndenumerate(t):
        # Log time
        t[i] = time.perf_counter_ns()
        
        # Calculate the next timestamp of the control loop
        next_timestamp = t[i] + dt

        # Read the position, velocity and force data from the robot
        p = robot.get_position()
        v = robot.get_velocity()
        f = robot.get_force()

        # Log robot data
        data[i, 0:3] = p
        data[i, 3:6] = v
        data[i, 6:9] = f

        # Send control signal
        F.force = [0.0, 0.0, 0.0]

        robot.set_bias_force(F)

        # Hold the main while loop until the time comes
        while time.perf_counter_ns() < next_timestamp:
            pass

    fig, ax = plt.subplots()
    ax.plot((t - t[0])*1e-9, data[:, 0], label='x-axis')
    ax.plot((t - t[0])*1e-9, data[:, 1], label='y-axis')
    ax.plot((t - t[0])*1e-9, data[:, 2], label='z-axis')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Position [m]')
    ax.legend()
    fig.savefig('position.pdf')

    fig, ax = plt.subplots()
    ax.plot((t - t[0])*1e-9, data[:, 3], label='x-axis')
    ax.plot((t - t[0])*1e-9, data[:, 4], label='y-axis')
    ax.plot((t - t[0])*1e-9, data[:, 5], label='z-axis')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Velocity [m/s]')
    ax.legend()
    fig.savefig('velocity.pdf')

    fig, ax = plt.subplots()
    ax.plot((t - t[0])*1e-9, data[:, 6], label='x-axis')
    ax.plot((t - t[0])*1e-9, data[:, 7], label='y-axis')
    ax.plot((t - t[0])*1e-9, data[:, 8], label='z-axis')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force [N]')
    ax.legend()
    fig.savefig('force.pdf')

    fig, ax = plt.subplots()
    ax.plot((np.diff(t)-dt) * 1e-6)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Jitter [ms]')
    fig.savefig('jitter.pdf')

    # Close connection
    robot.disconnect()