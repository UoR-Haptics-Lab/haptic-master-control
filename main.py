import csv
# from haptic_master.haptic_master import HapticMaster
# from haptic_master.bias_force import BiasForce
from pyHapticMaster.src.haptic_master.haptic_master import HapticMaster
from pyHapticMaster.src.haptic_master.bias_force import BiasForce
import matplotlib.pyplot as plt
import numpy as np
import os
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
    start_time = 0
    
    # Open connection
    robot.connect()

    # Create a bias force to control the robot
    F = BiasForce('myBiasForce')
    robot.create_bias_force(F)

    # Set robot state to position
    robot.set_state('position')

    # start_time = time.perf_counter_ns()

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
        # while time.perf_counter_ns() < next_timestamp:
        while time.perf_counter_ns() - t[i] < dt:
            pass

    # Close connection
    robot.disconnect()

    # Save data to file
    output_folder = os.path.join('.', 'experiments')
    output_file_name = 'experiment-' + time.strftime("%Y_%m_%d-%H_%M_%S") + '.csv'
    with open(os.path.join(output_folder, output_file_name), mode='w') as f:
        f.write('# Units: Time [s], Position [m], Velocity [m/s], Force [N]\n')
        f_writer = csv.writer(f, delimiter=',', quotechar='#', quoting=csv.QUOTE_MINIMAL)

        f_writer.writerow(['time', 'position_x', 'position_y', 'position_z', 'velocity_x', 'velocity_y', 'velocity_z', 'force_x', 'force_y', 'force_z'])
        
        for ti, d in zip(t, data):
            f_writer.writerow(np.hstack((ti, d)))