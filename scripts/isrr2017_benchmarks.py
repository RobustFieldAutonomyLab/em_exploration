from pyplanner2d import *

import sys
import os
from ConfigParser import SafeConfigParser
from multiprocessing import Pool


max_distance = 400
max_trials = 50
max_nodes_2 = 0.5
fixed_distances = np.arange(6, max_distance, 1)

options = []
# weights = list(np.arange(0.5, 10.0, 1.0))
weights = [1.5]
for seed in range(0, max_trials):
    for weight in weights:
        options.append(('OG_SHANNON', 2.0, weight, max_nodes_2, 0, seed))

# weights = list(np.arange(16.0, 24.0, 1.0))
weights = [23.0]
for seed in range(0, max_trials):
    for weight in weights:
        options.append(('OG_SHANNON', 0.5, weight, max_nodes_2 / 16, 0, seed))

# weights = list(np.arange(4, 10, 1.0))
weights = [9.0]
for seed in range(0, max_trials):
    for weight in weights:
        options.append(('EM_AOPT', 2.0, weight, max_nodes_2, 0, seed))

# weights = list(np.arange(140, 160, 1.0))
weights = [156.0]
for seed in range(0, max_trials):
    for weight in weights:
        options.append(('EM_AOPT', 0.5, weight, max_nodes_2 / 16, 0, seed))

# weights = [0.002, 0.004, 0.006, 0.008, 0.010]
# alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
weights = [0.004]
alphas = [0.05]
for seed in range(0, max_trials):
    for weight in weights:
        for alpha in alphas:
            options.append(('SLAM_OG_SHANNON', 2.0, weight, max_nodes_2, alpha, seed))

# weights = [0.002, 0.004, 0.006, 0.008, 0.010]
# alphas = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
weights = [0.004]
alphas = [0.01]
for seed in range(0, max_trials):
    for weight in weights:
        for alpha in alphas:
            options.append(('SLAM_OG_SHANNON', 0.5, weight, max_nodes_2 / 16, alpha, seed))


def run_exploration(option):
    algorithm, resolution, distance_weight0, max_nodes, alpha, seed = option
    config = SafeConfigParser()
    config.read(sys.path[0] + '/isrr2017_benchmarks.ini')

    range_noise = math.radians(0.1)
    config.set('Sensor Model', 'range_noise', str(range_noise))

    dir = '{}_{}_{}_{}_{}'.format(algorithm, resolution, distance_weight0, alpha, seed)
    print dir
    os.mkdir(dir)
    os.chdir(dir)

    config.set('Planner', 'max_nodes', str(max_nodes))
    config.set('Planner', 'algorithm', algorithm)
    config.set('Planner', 'distance_weight0', str(distance_weight0))
    config.set('Planner', 'distance_weight1', str(0.0))
    config.set('Planner', 'd_weight', str(0.05))
    config.set('Planner', 'alpha', str(alpha))
    config.set('Virtual Map', 'sigma0', str(1.2))
    config.set('Virtual Map', 'resolution', str(resolution))
    config.set('Simulator', 'seed', str(seed))
    config.set('Simulator', 'num', str(20))
    config.write(open(dir + '.ini', 'w'))

    status, planning_time = explore(dir + '.ini', max_distance, False, True, False)

    print dir + ' - ' + status
    os.chdir('..')
    with open('status.txt', 'a') as file:
        file.write(dir + ' - ' + status + '-' + str(planning_time) + '\n')


def run_plotting_landmarks(folder):
    return get_landmarks_uncertainty(folder, True, 20, 6, fixed_distances)


def run_plotting_trajectory(folder):
    return get_trajectory_uncertainty(folder, True, fixed_distances)


def run_plotting_map_entropy(folder):
    return get_map_entropy(folder, fixed_distances)


def exploration():
    pool = Pool()
    pool.map_async(run_exploration, options)
    pool.close()
    pool.join()

    os.chdir('..')


def plotting_separate():
    folders = get_folders()

    pool = Pool()
    results = pool.map(run_plotting_landmarks, folders)
    pool.close()
    pool.join()

    errors = measure_error(results)
    algorithms = ['OG_SHANNON_2.0', 'OG_SHANNON_0.5', 'EM_AOPT_0.5', 'EM_AOPT_2.0',
                  'SLAM_OG_SHANNON_2.0', 'SLAM_OG_SHANNON_0.5']
    best_options = []
    for algorithm in algorithms:
        best_options.append(('', 1e10))
        for option, error in errors.iteritems():
            if not option.startswith(algorithm):
                continue

            m, sigma = error
            if np.sum(m) < best_options[-1][1]:
                best_options[-1] = option, np.sum(m)
    print best_options

    for algorithm, best_option in zip(algorithms, best_options):
        for option, error in errors.iteritems():
            if not option.startswith(algorithm):
                continue
            m, sigma = error
            if option is best_option[0]:
                plt.plot(fixed_distances, m, 'k-', label=option)
            else:
                plt.plot(fixed_distances, m, '-', label=option, alpha=0.3)
            # plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.3)
        plt.legend(fontsize='xx-small')
        plt.xlabel('Distance (m)')
        plt.ylabel('Mapping uncertainty')
        plt.savefig('benchmarks_{}.pdf'.format(algorithm), dpi=800)
        plt.close()


def plotting_comparison():
    folders = get_folders()

    # folders_ = []
    # for f in folders:
    #     if f.startswith('EM_AOPT_2.0_9.0') or f.startswith('OG_SHANNON_0.5_23.0')\
    #           or f.startswith('OG_SHANNON_2.0_1.5') or f.startswith('EM_AOPT_0.5_156.0'):
    #         folders_.append(f)
    # folders = folders_

    pool = Pool()
    results = pool.map(run_plotting_landmarks, folders)
    pool.close()
    pool.join()

    errors = measure_error(results)
    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('EM_AOPT_2.0'):
            plt.plot(fixed_distances, m, '--', label='EM - 2.0')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('EM_AOPT_0.5'):
            plt.plot(fixed_distances, m, '-', label='EM - 0.5')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('OG_SHANNON_2.0'):
            plt.plot(fixed_distances, m, '--', label='OG - 2.0')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('OG_SHANNON_0.5'):
            plt.plot(fixed_distances, m, '-', label='OG - 0.5')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('SLAM_OG_SHANNON_2.0'):
            plt.plot(fixed_distances, m, '--', label='SLAM-OG - 2.0')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('SLAM_OG_SHANNON_0.5'):
            plt.plot(fixed_distances, m, '-', label='SLAM-OG - 0.5')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break
    plt.xlabel('Distance (m)')
    plt.ylabel('Mapping uncertainty')
    plt.legend()
    plt.savefig('benchmarks_landmarks.pdf', dpi=800)
    plt.close()

    pool = Pool()
    results = pool.map(run_plotting_trajectory, folders)
    pool.close()
    pool.join()

    errors = measure_error(results)
    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('EM_AOPT_2.0'):
            plt.plot(fixed_distances, m, '--', label='EM - 2.0')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('EM_AOPT_0.5'):
            plt.plot(fixed_distances, m, '-', label='EM - 0.5')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('OG_SHANNON_2.0'):
            plt.plot(fixed_distances, m, '--', label='OG - 2.0')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('OG_SHANNON_0.5'):
            plt.plot(fixed_distances, m, '-', label='OG - 0.5')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('SLAM_OG_SHANNON_2.0'):
            plt.plot(fixed_distances, m, '--', label='SLAM-OG - 2.0')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('SLAM_OG_SHANNON_0.5'):
            plt.plot(fixed_distances, m, '-', label='SLAM-OG - 0.5')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break
    plt.xlabel('Distance (m)')
    plt.ylabel('Max localization uncertainty')
    plt.legend()
    plt.savefig('benchmarks_trajectory.pdf', dpi=800)
    plt.close()

    pool = Pool()
    results = pool.map(run_plotting_map_entropy, folders)
    pool.close()
    pool.join()

    errors = measure_error(results)
    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('EM_AOPT_2.0'):
            plt.plot(fixed_distances, m, '--', label='EM - 2.0')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('EM_AOPT_0.5'):
            plt.plot(fixed_distances, m, '-', label='EM - 0.5')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('OG_SHANNON_2.0'):
            plt.plot(fixed_distances, m, '--', label='OG - 2.0')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('OG_SHANNON_0.5'):
            plt.plot(fixed_distances, m, '-', label='OG - 0.5')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('SLAM_OG_SHANNON_2.0'):
            plt.plot(fixed_distances, m, '--', label='SLAM-OG - 2.0')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break

    for option, error in errors.iteritems():
        m, sigma = error
        if option.startswith('SLAM_OG_SHANNON_0.5'):
            plt.plot(fixed_distances, m, '-', label='SLAM-OG - 0.5')
            plt.fill_between(fixed_distances, m - sigma, m + sigma, alpha=0.2)
            break
    plt.xlabel('Distance (m)')
    plt.ylabel('Map entropy reduction')
    plt.legend()
    plt.savefig('benchmarks_map_entropy.pdf', dpi=800)
    plt.close()


def planning_time_comparison():
    import pandas as pd
    import seaborn.apionly as sns

    algorithms = ['OG_SHANNON_2.0', 'OG_SHANNON_0.5', 'EM_AOPT_0.5', 'EM_AOPT_2.0',
                  'SLAM_OG_SHANNON_2.0', 'SLAM_OG_SHANNON_0.5']
    planning_time = {a: [0, 0] for a in algorithms}
    d = []
    for line in open('status.txt', 'r'):
        if '-' not in line:
            continue
        for a in algorithms:
            if line.startswith(a):
                if a == 'EM_AOPT_2.0':
                    d.append(['EM', '2.0', float(line.rsplit('-', 1)[1])])
                elif a == 'EM_AOPT_0.5':
                    d.append(['EM', '0.5', float(line.rsplit('-', 1)[1])])
                elif a == 'OG_SHANNON_2.0':
                    d.append(['OG', '2.0', float(line.rsplit('-', 1)[1])])
                elif a == 'OG_SHANNON_0.5':
                    d.append(['OG', '0.5', float(line.rsplit('-', 1)[1])])
                elif a == 'SLAM_OG_SHANNON_2.0':
                    d.append(['SLAM-OG', '2.0', float(line.rsplit('-', 1)[1])])
                elif a == 'SLAM_OG_SHANNON_0.5':
                    d.append(['SLAM-OG', '0.5', float(line.rsplit('-', 1)[1])])
                break
    df = pd.DataFrame(d, columns=['Algorithm', 'Resolution', 'Time (s)'])
    sns.barplot(x='Algorithm', y='Time (s)', hue='Resolution', capsize=0.2, data=df)
    plt.savefig('benchmarks_planning_time.pdf', dpi=800)


if __name__ == '__main__':
    # exploration()
    # plotting_separate()
    # plotting_comparison()
    planning_time_comparison()
