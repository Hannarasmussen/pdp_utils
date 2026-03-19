import numpy as np
import time
from pdp_utils import *
from tabulate import tabulate
import matplotlib.pyplot as plt
import os
import json


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

def plot_all_runs_cost_histories(cost_histories, instance_name):
    plt.figure(figsize=(12, 6))
    for i, cost_history in enumerate(cost_histories):
        plt.plot(cost_history, label=f"Run {i + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(f"Cost History Across Runs ({instance_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{instance_name}_all_runs_cost_history.png")
    plt.close()

def plot_operator_scores(score_history, instance_name, run_id):
    plt.figure(figsize=(12, 6))
    for name, scores in score_history.items():
        plt.plot(scores, label=name)
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

def plot_operator_deltas_normalized(operator_deltas, operator_delta_iters, instance_name, run_id):
    plt.figure(figsize=(12, 6))
    
    for operator in operator_deltas:
        deltas = np.array(operator_deltas[operator])
        if len(deltas) == 0:
            continue
        # Normaliser til [0, 1]
        min_delta = np.min(deltas)
        max_delta = np.max(deltas)
        if max_delta > min_delta:
            normalized_deltas = (deltas - min_delta) / (max_delta - min_delta)
        else:
            normalized_deltas = np.zeros_like(deltas)  # hvis alle verdier er like

        plt.scatter(operator_delta_iters[operator], normalized_deltas, alpha=0.5, label=operator)

    plt.title(f"Normalized Delta Values per Operator - Run {run_id} ({instance_name})")
    plt.xlabel("Iteration")
    plt.ylabel("Normalized Delta")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{instance_name}_run{run_id}_normalized_deltas_all.png")
    plt.close()


# def plot_temperature(temperature_history, instance_name, run_id):
#     plt.figure()
#     plt.plot(range(len(temperature_history)), temperature_history)
#     plt.title(f"Temperature over Iterations - Run {run_id} ({instance_name})")
#     plt.xlabel("Iteration")
#     plt.ylabel("Temperature")
#     plt.grid(True)
#     plt.savefig(f"plots/{instance_name}_run{run_id}_temperature.png")
#     plt.close()

# def plot_acceptance_probability(acceptance_iter_history, acceptance_prob_history, instance_name, run_id):
#     plt.figure()
#     plt.scatter(acceptance_iter_history, acceptance_prob_history, alpha=0.5)
#     plt.title(f"Acceptance Probability (Positive Delta) - Run {run_id} ({instance_name})")
#     plt.xlabel("Iteration")
#     plt.ylabel("Probability")
#     plt.grid(True)
#     plt.savefig(f"plots/{instance_name}_run{run_id}_acceptance_probability.png")
#     plt.close()

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
                        #'pdp_utils/data/pd_problem/Call_35_Vehicle_7.txt',
                        #'pdp_utils/data/pd_problem/Call_80_Vehicle_20.txt',
                        #'pdp_utils/data/pd_problem/Call_130_Vehicle_40.txt'
                        ]
    num_runs = 1
    results = {}

    os.makedirs("plots", exist_ok=True)

    
    for instance in test_instances:
        start_time = time.time()
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

        cost_histories = []

        for run in range(num_runs):
            print(f"Running instance {instance_name} - Run {run + 1}/{num_runs}")
            solution, operator_scores_history, cost_history, acceptance_iter_history, acceptance_prob_history, operator_deltas, operator_delta_iters, best_iteration = General_Adaptive_Metahuristics_Framework(problem, start_solution)
            cost_histories.append(cost_history)
            
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
                #'Temperature history': temperature_history,
                'Acceptance prob history': (acceptance_iter_history, acceptance_prob_history),
                'Operator deltas': (operator_deltas, operator_delta_iters),
                'Best iteration': best_iteration
            })

            #plot_operator_deltas_normalized(operator_deltas, operator_delta_iters, instance_name, run + 1)

        
        avg_cost = total_cost / num_runs
        improvement = 100 * (start_cost - best_cost) / start_cost
        running_time = time.time() - start_time

        summary = {
            'Average Objective': f"{avg_cost:.0f}",
            'Best Objective': f"{best_cost:.0f}",
            'Improvement (%)': f"{improvement:.2f}",
            'Running Time (s)': f"{running_time:.3f}",
            'Best solution': str(best_solution)
        }

        plot_all_runs_cost_histories(cost_histories, instance_name)
        
        plot_operator_scores(operator_scores_history, instance_name, run + 1)
        plot_cost_history(cost_history, instance_name, run + 1)
        #plot_temperature(temperature_history, instance_name, run + 1)
        #plot_acceptance_probability(acceptance_iter_history, acceptance_prob_history, instance_name, run + 1)
        plot_operator_deltas(operator_deltas, operator_delta_iters, instance_name, run + 1)
        

        print(tabulate([summary], headers="keys", tablefmt="fancy_grid"))
       
        report_final_objectives(objective_values_per_instance)
        print(f"Best solution found at iteration {best_iteration} for instance {instance_name}")

        print(f"\nSummary for {instance}:")
        for r in results[instance]:
            print(f"Run {r['Run']}: Objective = {r['Objective']}")

if __name__ == "__main__":
    main()
