from map import Map, Obstacle
import numpy as np
from reference_path import ReferencePath
from spatial_bicycle_models import BicycleModel
import matplotlib.pyplot as plt
from MPC import MPC
from scipy import sparse
import math as math

v_max = 1.0  # m/s
delta_max = 0.66  # rad
ay_max = 4.0  # m/s^2


def GenerateMap():
    # Load map file
    map = Map(file_path='maps/sim_map.png', origin=[-1, -2], resolution=0.005)

    # Specify waypoints
    wp_x = [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25, 0.75, 0.75, 1.25, 1.25, -0.75, -0.75, -0.25]
    wp_y = [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5, -1, -1, -0.5, -0.5, 0, 0, -1.5, -1.5]

    # Specify path resolution
    path_resolution = 0.05  # m / wp

    # Create smoothed reference path
    reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                                   smoothing_distance=5,
                                   max_width=0.23,
                                   circular=True)

    # Add obstacles
    use_obstacles = False
    if use_obstacles:
        obs1 = Obstacle(cx=0.0, cy=0.0, radius=0.05)
        obs2 = Obstacle(cx=-0.8, cy=-0.5, radius=0.08)
        obs3 = Obstacle(cx=-0.7, cy=-1.5, radius=0.05)
        obs4 = Obstacle(cx=-0.3, cy=-1.0, radius=0.08)
        obs5 = Obstacle(cx=0.27, cy=-1.0, radius=0.05)
        obs6 = Obstacle(cx=0.78, cy=-1.47, radius=0.05)
        obs7 = Obstacle(cx=0.73, cy=-0.9, radius=0.07)
        obs8 = Obstacle(cx=1.2, cy=0.0, radius=0.08)
        obs9 = Obstacle(cx=0.67, cy=-0.05, radius=0.06)
        map.add_obstacles([obs1, obs2, obs3, obs4, obs5, obs6, obs7,
                           obs8, obs9])

    return reference_path


# Controller
def GenerateControl(car):
    N = 30

    # Debug
    #N = 3

    Q = sparse.diags([1.0, 0.0, 0.0])
    R = sparse.diags([0.5, 0.0])
    QN = sparse.diags([1.0, 0.0, 0.0])

    # Debug
    # Q = sparse.diags([1.0, 2.0, 3.0])
    # R = sparse.diags([4.0, 0])
    # QN = sparse.diags([6.0, 7.0, 8.0])

    InputConstraints = {'umin': np.array([0.0, -np.tan(delta_max)/car.length]),
                        'umax': np.array([v_max, np.tan(delta_max)/car.length])}

    StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                        'xmax': np.array([np.inf, np.inf, np.inf])}

    mpc_control = MPC(car, N, Q, R, QN, StateConstraints, InputConstraints, ay_max)

    return mpc_control


def show_profiler(ref_path):
    fig, (ax1, ax2) = plt.subplots(num=1, nrows=2)

    vel_profile = ref_path.reference_velocity_profile #[:current_waypoint]
    vel_constrained = ref_path.max_velocity_profile #[:current_waypoint]

    ax1.plot(np.linspace(0, len(vel_profile), num=len(vel_profile)), vel_profile, color='g', label='Solution')
    ax1.plot(np.linspace(0, len(vel_constrained), num=len(vel_constrained)), vel_constrained, color='deepskyblue', label='Reference')

    ax1.set_ylabel('Velocity')
    ax1.set_xlabel('Waypoint id')

    ax1.legend()

    # Acceleration profile
    vel_profile = ref_path.reference_velocity_profile #[:current_waypoint]
    distance_between_wp = ref_path.distance_between_waypoints #[:current_waypoint]

    acc_profile = []
    for i in range(1, len(vel_profile)):
        acc_profile.append((vel_profile[i] - vel_profile[i - 1]) / (2 * distance_between_wp[i - 1]))

    ax2.plot(range(0, len(acc_profile)), acc_profile, color='g', label='Solution')

    # Boundaries
    ax2.plot(range(0, len(acc_profile)), np.ones(len(acc_profile)) * -0.1, color='r')
    ax2.plot(range(0, len(acc_profile)), np.ones(len(acc_profile)) * 0.5, color='r', label='Boundary')

    ax2.set_ylabel('Acceleration')
    ax2.set_xlabel('Waypoint id')
    ax2.legend()


def show_state_profiler(Ey, Ehead):
    fig, (ax1, ax2) = plt.subplots(num=2, nrows=2)

    ax1.plot(np.linspace(0, len(Ey), num=len(Ey)), Ey, color='g')
    ax1.set_ylabel('E_y')
    ax1.set_xlabel('Waypoint id')

    ax2.plot(np.linspace(0, len(Ehead), num=len(Ehead)), Ehead, color='g')
    ax2.set_ylabel('E_head')
    ax2.set_xlabel('Waypoint id')

    ax1.legend()


def Simulation(car, reference_path, mpc):
    ##############
    # Simulation #
    ##############

    # Set simulation time to zero
    t = 0.0

    # Logging containers
    x_log = [car.temporal_state.x]
    y_log = [car.temporal_state.y]
    v_log = [0.0]

    # Velocity profile
    show_profiler(reference_path)

    # Until arrival at end of path
    while car.s < reference_path.length:
        plt.figure(0)

        # Get control signals
        u = mpc.get_control()

        # Simulate car
        car.drive(u)

        # Log car state
        x_log.append(car.temporal_state.x)
        y_log.append(car.temporal_state.y)
        v_log.append(u[0])

        # Increment simulation time
        t += car.deltaTime

        # Plot path and drivable area
        reference_path.show()

        # Plot car
        car.show()

        # Plot MPC prediction
        mpc.show_prediction()

        # Set figure title
        plt.title('MPC Simulation: v(t): {:.2f}, delta(t): {:.2f}, Duration: '
                  '{:.2f} s'.format(u[0], u[1], t))

        plt.axis('off')
        plt.pause(0.001)

    show_state_profiler([item[0] for item in mpc.spartial_st_log], [item[1] for item in mpc.spartial_st_log])
    plt.show()


def Main():
    reference_path = GenerateMap()

    # Instantiate motion model
    car = BicycleModel(length=0.12, width=0.06, reference_path=reference_path, delta_time=0.05)
    control = GenerateControl(car)

    # Compute speed profile
    a_min = -0.1  # m/s^2
    a_max = 0.5  # m/s^2

    SpeedProfileConstraints = {'a_min': a_min, 'a_max': a_max,
                               'v_min': 0.0, 'v_max': v_max, 'ay_max': ay_max}
    car.reference_path.compute_speed_profile(SpeedProfileConstraints)

    Simulation(car, reference_path, control)


if __name__ == '__main__':
    Main()



