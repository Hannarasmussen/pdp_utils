# import numpy as np
# import time
# from pdp_utils import *
# from tabulate import tabulate
# import matplotlib.pyplot as plt

# import os

# import json

# # hvordan skal jeg lagre alle de ulike plotene og løsningene?

# def plot_cost_history(cost_history):
#     plt.figure(figsize=(12, 6))
#     plt.plot(cost_history)
#     plt.xlabel("Iteration")
#     plt.ylabel("Cost")
#     plt.title("Cost Evolution")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("cost_history.png")
#     plt.close()
    
# def plot_operator_scores(score_history):
#     plt.figure(figsize=(12, 6))
#     for name, scores in score_history.items():
#         plt.plot(range(0,10000, 100),scores, label=name)
#     plt.xlabel("Score update steps")
#     plt.ylabel("Normalized Operator Score")
#     plt.title("Operator Score Evolution")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("operator_scores.png")
#     plt.close()

# def plot_operator_deltas(operator_deltas, operator_delta_iters, instance_name):
#     for operator in operator_deltas:
#         plt.figure()
#         plt.scatter(operator_delta_iters[operator], operator_deltas[operator], alpha=0.5)
#         plt.title(f"Delta Values for {operator} ({instance_name})")
#         plt.xlabel("Iteration")
#         plt.ylabel("Delta")
#         plt.grid(True)
#         plt.savefig(f"delta_values_{operator}_{instance_name}.png")
#         plt.close()
    
# def plot_temperature(temperature_history, instance_name):
#     plt.figure()
#     plt.plot(range(len(temperature_history)), temperature_history)
#     plt.title(f"Temperature over Iterations ({instance_name})")
#     plt.xlabel("Iteration")
#     plt.ylabel("Temperature")
#     plt.grid(True)
#     plt.savefig(f"temperature_plot_{instance_name}.png")
#     plt.close()

# def plot_acceptance_probability(acceptance_iter_history, acceptance_prob_history, instance_name):
#     plt.figure()
#     plt.scatter(acceptance_iter_history, acceptance_prob_history, alpha=0.5)
#     plt.title(f"Acceptance Probability (Positive Delta) ({instance_name})")
#     plt.xlabel("Iteration")
#     plt.ylabel("Probability")
#     plt.grid(True)
#     plt.savefig(f"acceptance_probability_{instance_name}.png")
#     plt.close()

# def report_best_solution_iterations(best_iterations_per_run, instance_name):
#     print(f"Best solution iterations for {instance_name}:")
#     for i, iter in enumerate(best_iterations_per_run):
#         print(f"Run {i + 1}: Iteration {iter}")

# def report_final_objectives(objective_values_per_instance):
#     for instance, values in objective_values_per_instance.items():
#         print(f"Final objectives for {instance}: {values}")

# def main():
#     test_instances = [#'pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt', 
#                       'pdp_utils/data/pd_problem/Call_18_Vehicle_5.txt',
#                       #'pdp_utils/data/pd_problem/Call_35_Vehicle_7.txt',
#                       #'pdp_utils/data/pd_problem/Call_80_Vehicle_20.txt',
#                       #'pdp_utils/data/pd_problem/Call_130_Vehicle_40.txt',
#                       #'pdp_utils/data/pd_problem/Call_300_Vehicle_90.txt'
#                       ]

#     num_runs = 1
#     results = {}
# #    results[instance] = []  # Liste med resultater for hver run

#     os.makedirs("plots", exist_ok=True)


#     for instance in test_instances:
#         start_time = time.time()
#         problem = load_problem(instance)
#         start_solution = initial_solution(problem)
#         start_cost = cost_function(start_solution, problem)

#         #solution = find_best_random(problem)
#         #solution = local_search(problem)
#         #solution = simulated_annealing_1(problem)
#         #solution = simulated_annealing(problem)
#         #solution = simulated_annealing_weight(problem)
#         # solution = General_Adaptive_Metahuristics_Framework(problem, start_solution)

#         #Jeg vil lagre best solution for hver run, og så ta snittet av de beste løsningene
#         #Jeg må nulstille best_solution og best_cost for hver run ? 
       
#         best_solution = start_solution
#         best_iterations_per_run = []
#         objective_values_per_instance = {}
#         #best_solution = start_solution
#         best_cost = start_cost
#         total_cost = 0

#         #må jeg flytte feasibility check til inni for løkken?
#         feasiblity, c = feasibility_check(best_solution, problem)

#         for run in range(num_runs):
#             solution, operator_scores_history, cost_history, temperature_history, acceptance_iter_history, acceptance_prob_history, operator_deltas, operator_delta_iters = General_Adaptive_Metahuristics_Framework(problem, start_solution)
#             #solution, operator_scores_history, cost_history = General_Adaptive_Metahuristics_Framework(problem, start_solution)
#             solution_cost = cost_function(solution, problem)
#             total_cost += solution_cost

#             if solution_cost < best_cost:
#                 best_solution = solution
#                 best_cost = solution_cost
#                 best_iterations_per_run.append(run)
#                 objective_values_per_instance[instance] = solution_cost

#         avg_cost = total_cost / num_runs
#         improvement = 100 * (start_cost - best_cost) / start_cost
#         end_time = time.time()
#         running_time = end_time - start_time

#         #Printe en liste av alle best solutions fra hver run og den beste løsningen av alternativene
#         best_solution_list = [cost_function(solution, problem) for solution in best_iterations_per_run]
#         best_solution = min(best_solution_list)
#         best_solution = best_solution_list[best_iterations_per_run.index(best_solution)]
#         best_solution = str(best_solution)
     

#         results[instance] = {
#             'Average Objective': avg_cost,
#             'Best Objective': best_cost,
#             'Improvement (%)': improvement,
#             'Running Time (s)': running_time,
#             'Best solution': str(best_solution)
     
#         }

#         results[instance].append({
#             'Run': run + 1,
#             'Objective': solution_cost,
#             'Best solution': str(solution),
#             'Cost history': cost_history,
#             'Operator scores': operator_scores_history,
#             'Temperature history': temperature_history,
#             'Acceptance prob history': (acceptance_iter_history, acceptance_prob_history),
#             'Operator deltas': (operator_deltas, operator_delta_iters)
#         })


#         print(feasiblity)
#         print(c)
       


#         #plot_operator_scores(operator_scores_history)
#         #plot_cost_history(cost_history)


#         plot_operator_scores(operator_scores_history)
#         plot_cost_history(cost_history)
#         plot_temperature(temperature_history, instance.split('/')[-1])
#         plot_acceptance_probability(acceptance_iter_history, acceptance_prob_history, instance.split('/')[-1])
#         plot_operator_deltas(operator_deltas, operator_delta_iters, instance.split('/')[-1])

#         report_best_solution_iterations(best_iterations_per_run, instance.split('/')[-1])
#         report_final_objectives(objective_values_per_instance)

#         print(tabulate(results[instance], headers="keys", tablefmt="fancy_grid"))

#         # with open(f"results_{instance_name}.json", "w") as f:
#         #     json.dump(results[instance], f, indent=4)

      
#         #print("\nSummary of results:")
#         #for instance, result in results.items():
#         #    print(f"{instance:<50} {result['Average Objective']:<15.2f} {result['Best Objective']:<15.2f} {result['Improvement (%)']:<20.2f} {result['Running Time (s)']:<20.2f} {best_solution}")


    

#     print("\nResult summary:")
#     for name, r in results.items():
#         print(f"{name}")
#         for k, v in r.items():
#             print(f"{k}: {v}")
#         print()

#     print(f"\nSummary for {instance}:")
#     for r in results[instance]:
#         print(f"Run {r['Run']}: Objective = {r['Objective']}")

# if __name__ == "__main__":
#     main()


import numpy as np
import time
from pdp_utils import *
from tabulate import tabulate
import matplotlib.pyplot as plt
import os
import json

# hvordan skal jeg lagre alle de ulike plotene og løsningene?

def plot_cost_history(cost_history, instance_name, run_id):
    plt.figure(figsize=(12, 6))
    plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(f"Cost Evolution - Run {run_id} ({instance_name})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{instance_name}_run{run_id}_cost_history.png")
    plt.close()

def plot_operator_scores(score_history, instance_name, run_id):
    plt.figure(figsize=(12, 6))
    for name, scores in score_history.items():
        plt.plot(range(0, 10000, 100), scores, label=name)
    plt.xlabel("Score update steps")
    plt.ylabel("Normalized Operator Score")
    plt.title(f"Operator Score Evolution - Run {run_id} ({instance_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{instance_name}_run{run_id}_operator_scores.png")
    plt.close()

def plot_operator_deltas(operator_deltas, operator_delta_iters, instance_name, run_id):
    for operator in operator_deltas:
        plt.figure()
        plt.scatter(operator_delta_iters[operator], operator_deltas[operator], alpha=0.5)
        plt.title(f"Delta Values for {operator} - Run {run_id} ({instance_name})")
        plt.xlabel("Iteration")
        plt.ylabel("Delta")
        plt.grid(True)
        plt.savefig(f"plots/{instance_name}_run{run_id}_delta_{operator}.png")
        plt.close()

def plot_temperature(temperature_history, instance_name, run_id):
    plt.figure()
    plt.plot(range(len(temperature_history)), temperature_history)
    plt.title(f"Temperature over Iterations - Run {run_id} ({instance_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Temperature")
    plt.grid(True)
    plt.savefig(f"plots/{instance_name}_run{run_id}_temperature.png")
    plt.close()

def plot_acceptance_probability(acceptance_iter_history, acceptance_prob_history, instance_name, run_id):
    plt.figure()
    plt.scatter(acceptance_iter_history, acceptance_prob_history, alpha=0.5)
    plt.title(f"Acceptance Probability (Positive Delta) - Run {run_id} ({instance_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.savefig(f"plots/{instance_name}_run{run_id}_acceptance_probability.png")
    plt.close()

def report_best_solution_iterations(best_iterations_per_run, instance_name):
    print(f"Best solution iterations for {instance_name}:")
    for i, iter in enumerate(best_iterations_per_run):
        print(f"Run {i + 1}: Iteration {iter}")

def report_final_objectives(objective_values_per_instance):
    for instance, values in objective_values_per_instance.items():
        print(f"Final objectives for {instance}: {values}")

def main():
    test_instances = [  'pdp_utils/data/pd_problem/Call_7_Vehicle_3.txt',
                        #'pdp_utils/data/pd_problem/Call_18_Vehicle_5.txt',
                        # 'pdp_utils/data/pd_problem/Call_35_Vehicle_7.txt',
                        # 'pdp_utils/data/pd_problem/Call_80_Vehicle_20.txt',
                        # 'pdp_utils/data/pd_problem/Call_130_Vehicle_40.txt',
                        #'pdp_utils/data/pd_problem/Call_300_Vehicle_90.txt'
                        ]
    num_runs = 10
    results = {}

    os.makedirs("plots", exist_ok=True)

    for instance in test_instances:
        instance_name = instance.split("/")[-1].replace(".txt", "")
        problem = load_problem(instance)
        start_solution = initial_solution(problem)
        start_cost = cost_function(start_solution, problem)

        best_solution = start_solution
        best_cost = start_cost
        total_cost = 0
        best_iterations_per_run = []
        objective_values_per_instance = {}
        results[instance] = []

        for run in range(num_runs):
            start_time = time.time()
            solution, operator_scores_history, cost_history, temperature_history, acceptance_iter_history, acceptance_prob_history, operator_deltas, operator_delta_iters = General_Adaptive_Metahuristics_Framework(problem, start_solution)
            solution_cost = cost_function(solution, problem)
            total_cost += solution_cost

            if solution_cost < best_cost:
                best_solution = solution
                best_cost = solution_cost
                best_iterations_per_run.append(run)
                objective_values_per_instance[instance] = solution_cost

            results[instance].append({
                'Run': run + 1,
                'Objective': solution_cost,
                'Best solution': str(solution),
                'Cost history': cost_history,
                'Operator scores': operator_scores_history,
                'Temperature history': temperature_history,
                'Acceptance prob history': (acceptance_iter_history, acceptance_prob_history),
                'Operator deltas': (operator_deltas, operator_delta_iters)
            })

            
        
        avg_cost = total_cost / num_runs
        improvement = 100 * (start_cost - best_cost) / start_cost
        running_time = time.time() - start_time

        # summary = {
        #     'Average Objective': avg_cost,
        #     'Best Objective': best_cost,
        #     'Improvement (%)': improvement,
        #     'Running Time (s)': running_time,
        #     'Best solution': str(best_solution)
        # }

        summary = {
            'Average Objective': f"{avg_cost:.0f}",
            'Best Objective': f"{best_cost:.0f}",
            'Improvement (%)': f"{improvement:.2f}",
            'Running Time (s)': f"{running_time:.3f}",
            'Best solution': str(best_solution)
        }

        plot_operator_scores(operator_scores_history, instance_name, run + 1)
        plot_cost_history(cost_history, instance_name, run + 1)
        plot_temperature(temperature_history, instance_name, run + 1)
        plot_acceptance_probability(acceptance_iter_history, acceptance_prob_history, instance_name, run + 1)
        plot_operator_deltas(operator_deltas, operator_delta_iters, instance_name, run + 1)

        print(tabulate([summary], headers="keys", tablefmt="fancy_grid"))
        # with open(f"results/results_{instance_name}.json", "w") as f:
        #     json.dump(results[instance], f, indent=4)

        report_best_solution_iterations(best_iterations_per_run, instance_name) # denne gir ikke mening
        report_final_objectives(objective_values_per_instance)

        print(f"\nSummary for {instance}:")
        for r in results[instance]:
            print(f"Run {r['Run']}: Objective = {r['Objective']}")

if __name__ == "__main__":
    main()
