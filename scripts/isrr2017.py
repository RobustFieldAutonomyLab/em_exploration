import sys
from pyplanner2d import *
import matplotlib.lines as mlines

import tempfile


def explore_isrr2017_structured(config_file, max_steps, verbose=False, save_history=False, save_fig=True):
    config = load_config(config_file)
    range_noise = math.radians(0.1)
    config.set('Sensor Model', 'range_noise', str(range_noise))

    explorer = EMExplorer(config, verbose, save_history)

    status = 'MAX_STEP'
    actions = []
    for step in range(max_steps):
        if step < 4:
            odom = 0, 0, math.pi / 2.0
            explorer.simulate(odom, core=True)
        else:
            result = explorer.plan()
            if result == planner2d.EMPlanner2D.OptimizationResult.SAMPLING_FAILURE:
                explorer.simulate((0, 0, math.pi / 4), True)
            elif result == planner2d.EMPlanner2D.OptimizationResult.NO_SOLUTION:
                status = 'NO SOLUTION'
                break
            elif result == planner2d.EMPlanner2D.OptimizationResult.TERMINATION:
                status = 'TERMINATION'
                break
            else:
                plot = (explorer.step == 117)
                # plot = False
                if plot:
                    plot_environment(explorer._sim.environment, label=False)
                    plot_pose(explorer._sim.vehicle, explorer._sensor_params)
                    plot_map(explorer._slam.map)
                    plot_virtual_map(explorer._virtual_map, explorer._map_params, None, False)
                    plot_path(explorer._planner, dubins=True, cov=False, rrt=False)

                    l1 = mlines.Line2D([], [], linestyle='none', color='k', marker='+', label='Ground Truth - Landmarks')
                    l2 = mlines.Line2D([], [], linestyle='none', color='orange', marker='+', label='Estimate - Landmarks')
                    l3 = mlines.Line2D([], [], linestyle='-', color='k', label='Ground Truth - Trajectory')
                    l4 = mlines.Line2D([], [], linestyle='-', color='g', label='Estimate - Trajectory')
                    l5 = mlines.Line2D([], [], linestyle='-', color='y', label='RRT')
                    l6 = mlines.Line2D([], [], linestyle='-', color='darkred', label='Best Trajectory')
                    # plt.legend(handles=[l1, l3, l2, l4, l5, l6], fontsize='x-small', loc='lower right')

                    plt.xlabel('X (m)')
                    plt.ylabel('Y (m)')
                    plt.savefig('EM_AOPT_step_{}_before.pdf'.format(explorer.step), dpi=800, bbox_inches='tight')
                    plt.close()

                explorer.savefig(path=True)
                explorer.follow_dubins_path(5)
                explorer.savefig(path=False)

                if plot:
                    plot_environment(explorer._sim.environment, label=False)
                    plot_pose(explorer._sim.vehicle, explorer._sensor_params)
                    plot_map(explorer._slam.map)
                    plot_virtual_map(explorer._virtual_map, explorer._map_params, None, True)
                    l1 = mlines.Line2D([], [], linestyle='none', color='k', marker='+', label='Ground Truth - Landmarks')
                    l2 = mlines.Line2D([], [], linestyle='none', color='orange', marker='+', label='Estimate - Landmarks')
                    l3 = mlines.Line2D([], [], linestyle='-', color='k', label='Ground Truth - Trajectory')
                    l4 = mlines.Line2D([], [], linestyle='-', color='g', label='Estimate - Trajectory')
                    # plt.legend(handles=[l1, l3, l2, l4], fontsize='x-small', loc='lower right')

                    plt.xlabel('X (m)')
                    plt.ylabel('Y (m)')
                    plt.savefig('EM_AOPT_step_{}_after.pdf'.format(explorer.step), dpi=800, bbox_inches='tight')
                    plt.close()


def explore_isrr2017_random(config_file, max_steps, verbose=False, save_history=False, save_fig=True):
    config = load_config(config_file)
    range_noise = math.radians(0.1)
    config.set('Sensor Model', 'range_noise', str(range_noise))

    explorer = EMExplorer(config, verbose, save_history)

    status = 'MAX_STEP'
    actions = []
    for step in range(max_steps):
        if step < 4:
            odom = 0, 0, math.pi / 2.0
            explorer.simulate(odom, core=True)
        else:
            result = explorer.plan()
            if result == planner2d.EMPlanner2D.OptimizationResult.SAMPLING_FAILURE:
                explorer.simulate((0, 0, math.pi / 4), True)
            elif result == planner2d.EMPlanner2D.OptimizationResult.NO_SOLUTION:
                status = 'NO SOLUTION'
                break
            elif result == planner2d.EMPlanner2D.OptimizationResult.TERMINATION:
                status = 'TERMINATION'
                break
            else:
                plot = (explorer.step == 23) or (explorer.step == 45) or (explorer.step == 75) or (explorer.step == 125) or (explorer.step == 157)
                # plot = (explorer.step == 45)
                # plot = False
                if plot:
                    plot_environment(explorer._sim.environment, label=False)
                    plot_pose(explorer._sim.vehicle, explorer._sensor_params)
                    plot_map(explorer._slam.map)
                    plot_virtual_map(explorer._virtual_map, explorer._map_params, None, False)
                    plot_path(explorer._planner, dubins=True, cov=False, rrt=False)

                    l1 = mlines.Line2D([], [], linestyle='none', color='k', marker='+', label='Ground Truth - Landmarks')
                    l2 = mlines.Line2D([], [], linestyle='none', color='orange', marker='+', label='Estimate - Landmarks')
                    l3 = mlines.Line2D([], [], linestyle='-', color='k', label='Ground Truth - Trajectory')
                    l4 = mlines.Line2D([], [], linestyle='-', color='g', label='Estimate - Trajectory')
                    l5 = mlines.Line2D([], [], linestyle='-', color='y', label='RRT')
                    l6 = mlines.Line2D([], [], linestyle='-', color='darkred', label='Best Trajectory')
                    # plt.legend(handles=[l1, l3, l2, l4, l5, l6], fontsize='x-small', loc='lower right')

                    plt.xlabel('X (m)')
                    plt.ylabel('Y (m)')
                    plt.savefig('EM_AOPT_step_{}_before.pdf'.format(explorer.step), dpi=800, bbox_inches='tight')
                    plt.close()

                explorer.savefig(path=True)
                explorer.follow_dubins_path(5)
                explorer.savefig(path=False)

                if plot:
                    plot_environment(explorer._sim.environment, label=False)
                    plot_pose(explorer._sim.vehicle, explorer._sensor_params)
                    plot_map(explorer._slam.map)
                    plot_virtual_map(explorer._virtual_map, explorer._map_params, None, True)
                    l1 = mlines.Line2D([], [], linestyle='none', color='k', marker='+', label='Ground Truth - Landmarks')
                    l2 = mlines.Line2D([], [], linestyle='none', color='orange', marker='+', label='Estimate - Landmarks')
                    l3 = mlines.Line2D([], [], linestyle='-', color='k', label='Ground Truth - Trajectory')
                    l4 = mlines.Line2D([], [], linestyle='-', color='g', label='Estimate - Trajectory')
                    # plt.legend(handles=[l1, l3, l2, l4], fontsize='x-small', loc='lower right')

                    plt.xlabel('X (m)')
                    plt.ylabel('Y (m)')
                    plt.savefig('EM_AOPT_step_{}_after.pdf'.format(explorer.step), dpi=800, bbox_inches='tight')
                    plt.close()


# config_file = sys.path[0] + '/isrr2017_random.ini'
# explore_isrr2017_random(config_file, 100, True, False, True)
config_file = sys.path[0] + '/isrr2017_structured.ini'
explore_isrr2017_structured(config_file, 100, True, False, True)

