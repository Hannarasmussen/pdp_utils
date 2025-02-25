import numpy as np
import time
from pdp_utils import *


def main():
    test_instances = ['pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt', 
                      'pdp_utils/data/pd_problem/Call_18_Vehicle_5.txt',
                      'pdp_utils/data/pd_problem/Call_35_Vehicle_7.txt',
                      'pdp_utils/data/pd_problem/Call_80_Vehicle_20.txt',
                      'pdp_utils/data/pd_problem/Call_130_Vehicle_40.txt',
                      'pdp_utils/data/pd_problem/Call_300_Vehicle_90.txt'
                      ]

    num_runs = 10
    results = {}

    for instance in test_instances:

        start_time = time.time()
        problem = load_problem(instance)

        initial_solution = generate_random(problem)
        initial_cost = cost_function(initial_solution, problem)
        
        best_solution = initial_solution
        best_cost = initial_cost
        total_cost = initial_cost
        feasiblity, c = feasibility_check(best_solution, problem)

        for run in range(num_runs):
            solution = generate_random(problem)
            cost = cost_function(solution, problem)

            total_cost += cost

            if cost < best_cost:
                best_solution = solution
                best_cost = cost

        avg_cost = total_cost / num_runs
        improvement = 100 * (initial_cost - best_cost) / initial_cost
        end_time = time.time()
        running_time = end_time - start_time

        results[instance] = {
            'Average Objective': avg_cost,
            'Best Objective': best_cost,
            'Improvement (%)': improvement,
            'Running Time (s)': running_time,
            'Best solution': str(best_solution)
        }

        print(f"Initial Solution Cost: {initial_cost}")
        print(f"Best Found Solution Cost: {best_cost}")
        print(f"Average Objective: {avg_cost}")
        print(f"Improvement (%): {improvement:.2f}%")
        print(f"Running Time: {running_time:.2f} seconds")
        print(feasiblity)
        print(c)

    print("\nSummary of results:")
    print(f"{'Instance':<50} {'Avg Objective':<15} {'Best Objective':<15} {'Improvement (%)':<20} {'Running Time (s)':<20} {'Best Solution':<50}")
    for instance, result in results.items():
        print(f"{instance:<50} {result['Average Objective']:<15.2f} {result['Best Objective']:<15.2f} {result['Improvement (%)']:<20.2f} {result['Running Time (s)']:<20.2f} {best_solution}")

if __name__ == "__main__":
    main()