import numpy as np
import time
from pdp_utils import *
from tabulate import tabulate


def main():
    test_instances = [#'pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt', 
                      #'pdp_utils/data/pd_problem/Call_18_Vehicle_5.txt',
                      #'pdp_utils/data/pd_problem/Call_35_Vehicle_7.txt',
                      #'pdp_utils/data/pd_problem/Call_80_Vehicle_20.txt',
                      #'pdp_utils/data/pd_problem/Call_130_Vehicle_40.txt',
                      'pdp_utils/data/pd_problem/Call_300_Vehicle_90.txt'
                      ]

    num_runs = 10
    results = {}

    for instance in test_instances:

        start_time = time.time()
        problem = load_problem(instance)

        start_solution = initial_solution(problem)
        start_cost = cost_function(start_solution, problem)

        #solution = find_best_random(problem)
        #solution = local_search(problem)
        #solution = simulated_annealing_1(problem)
        #solution = simulated_annealing(problem)
        solution = simulated_annealing_weight(problem)


        cost = cost_function(solution, problem)
        
        best_solution = solution

        best_cost = cost
        total_cost = 0

        feasiblity, c = feasibility_check(best_solution, problem)

        for run in range(num_runs):
            solution = initial_solution(problem)
            solution_cost = cost_function(solution, problem)

            total_cost += solution_cost

            if solution_cost < best_cost:
                best_solution = solution
                best_cost = solution_cost

        avg_cost = total_cost / num_runs
        improvement = 100 * (start_cost - best_cost) / start_cost
        end_time = time.time()
        running_time = end_time - start_time

        results[instance] = {
            'Average Objective': avg_cost,
            'Best Objective': best_cost,
            'Improvement (%)': improvement,
            'Running Time (s)': running_time,
            'Best solution': str(best_solution)
     
        }

        print(feasiblity)
        print(c)

        print("\nSummary of results:")
        for instance, result in results.items():
            print(f"{instance:<50} {result['Average Objective']:<15.2f} {result['Best Objective']:<15.2f} {result['Improvement (%)']:<20.2f} {result['Running Time (s)']:<20.2f} {best_solution}")


if __name__ == "__main__":
    main()