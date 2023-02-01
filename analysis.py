import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


sns.set_theme(style="darkgrid")


class Experiment:
    def __init__(self, file_name, dt) -> None:
        self.file_name = file_name
        self.dt = dt

        # Output folder for plots
        self.plot_folder = os.path.join(os.path.dirname(self.file_name), 'plots')
        if not os.path.exists(self.plot_folder):
            os.makedirs(self.plot_folder)

    def jitter_analysis(self):
        # Read experimental data from the file
        df = pd.read_csv(self.file_name, skiprows=[0])

        # Jitter analysis
        t = df['time'].to_numpy()
        jitter = (np.diff(t)-self.dt)

        print(f'Sampling time is {self.dt * 1e-3}us and average jitter in the experiment is {np.average(jitter) * 1e-3}us')

        fig, ax = plt.subplots()
        ax.plot(jitter * 1e-3)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Jitter [us]')
        fig.tight_layout()

        output_file = os.path.join(self.plot_folder, 'jitter.pdf')
        fig.savefig(output_file)

    def plot_pvf(self):
        # Read experimental data from the file
        df = pd.read_csv(self.file_name, skiprows=[0])

        # Remove time offset and convert to seconds
        t0 = df['time'][0]
        df['time'] = df['time'].apply(lambda x: (x - t0)*1e-9)

        # Plot position
        fig, ax = plt.subplots()
        ax.plot('time', 'position_x', '', data=df, label='x-axis', lw=2)
        ax.plot('time', 'position_y', '', data=df, label='y-axis', lw=2)
        ax.plot('time', 'position_z', '', data=df, label='z-axis', lw=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [m]')
        ax.legend()
        fig.tight_layout()

        output_file = os.path.join(self.plot_folder, 'position.pdf')
        fig.savefig(output_file)

        # Plot velocity
        fig, ax = plt.subplots()
        ax.plot('time', 'velocity_x', '', data=df, label='x-axis', lw=2)
        ax.plot('time', 'velocity_y', '', data=df, label='y-axis', lw=2)
        ax.plot('time', 'velocity_z', '', data=df, label='z-axis', lw=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Velocity [m/s]')
        ax.legend()

        output_file = os.path.join(self.plot_folder, 'velocity.pdf')
        fig.tight_layout()


        fig.savefig(output_file)

        # Plot force
        fig, ax = plt.subplots()
        ax.plot('time', 'force_x', '', data=df, label='x-axis', lw=2)
        ax.plot('time', 'force_y', '', data=df, label='y-axis', lw=2)
        ax.plot('time', 'force_z', '', data=df, label='z-axis', lw=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Force [N]')
        ax.legend()
        fig.tight_layout()
        
        output_file = os.path.join(self.plot_folder, 'force.pdf')
        fig.savefig(output_file)


if __name__ == '__main__':
    dts = [4000000]
    experiment_files = ['experiment-2023_02_01-15_54_48']
    output_files = [os.path.join('.', 'experiments', e, e + '.csv') for e in experiment_files]

    exp1 = Experiment(output_files[0], dts[0])

    exp1.jitter_analysis()
    exp1.plot_pvf()