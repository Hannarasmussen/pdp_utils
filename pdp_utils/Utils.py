import numpy as np
import random
import math
import copy
import matplotlib as plt
import heapq


def load_problem(filename):
    """

    :rtype: object
    :param filename: The address to the problem input file
    :return: named tuple object of the problem attributes
    """
    A = []
    B = []
    C = []
    D = []
    E = []
    with open(filename) as f:
        lines = f.readlines()
        num_nodes = int(lines[1])
        num_vehicles = int(lines[3])
        num_calls = int(lines[num_vehicles + 5 + 1])

        for i in range(num_vehicles):
            A.append(lines[1 + 4 + i].split(','))

        for i in range(num_vehicles):
            B.append(lines[1 + 7 + num_vehicles + i].split(','))

        for i in range(num_calls):
            C.append(lines[1 + 8 + num_vehicles * 2 + i].split(','))

        for j in range(num_nodes * num_nodes * num_vehicles):
            D.append(lines[1 + 2 * num_vehicles + num_calls + 9 + j].split(','))

        for i in range(num_vehicles * num_calls):
            E.append(lines[1 + 1 + 2 * num_vehicles + num_calls + 10 + j + i].split(','))
        f.close()

    Cargo = np.array(C, dtype=np.double)[:, 1:]
    D = np.array(D, dtype=int)

    TravelTime = np.zeros((num_vehicles + 1, num_nodes + 1, num_nodes + 1))
    TravelCost = np.zeros((num_vehicles + 1, num_nodes + 1, num_nodes + 1))
    for j in range(len(D)):
        TravelTime[D[j, 0]][D[j, 1], D[j, 2]] = D[j, 3]
        TravelCost[D[j, 0]][D[j, 1], D[j, 2]] = D[j, 4]

    VesselCapacity = np.zeros(num_vehicles)
    StartingTime = np.zeros(num_vehicles)
    FirstTravelTime = np.zeros((num_vehicles, num_nodes))
    FirstTravelCost = np.zeros((num_vehicles, num_nodes))
    A = np.array(A, dtype=int)
    for i in range(num_vehicles):
        VesselCapacity[i] = A[i, 3]
        StartingTime[i] = A[i, 2]
        for j in range(num_nodes):
            FirstTravelTime[i, j] = TravelTime[i + 1, A[i, 1], j + 1] + A[i, 2]
            FirstTravelCost[i, j] = TravelCost[i + 1, A[i, 1], j + 1]
    TravelTime = TravelTime[1:, 1:, 1:]
    TravelCost = TravelCost[1:, 1:, 1:]
    VesselCargo = np.zeros((num_vehicles, num_calls + 1))
    B = np.array(B, dtype=object)
    for i in range(num_vehicles):
        VesselCargo[i, np.array(B[i][1:], dtype=int)] = 1
    VesselCargo = VesselCargo[:, 1:]

    LoadingTime = np.zeros((num_vehicles + 1, num_calls + 1))
    UnloadingTime = np.zeros((num_vehicles + 1, num_calls + 1))
    PortCost = np.zeros((num_vehicles + 1, num_calls + 1))
    E = np.array(E, dtype=int)
    for i in range(num_vehicles * num_calls):
        LoadingTime[E[i, 0], E[i, 1]] = E[i, 2]
        UnloadingTime[E[i, 0], E[i, 1]] = E[i, 4]
        PortCost[E[i, 0], E[i, 1]] = E[i, 5] + E[i, 3]

    LoadingTime = LoadingTime[1:, 1:]
    UnloadingTime = UnloadingTime[1:, 1:]
    PortCost = PortCost[1:, 1:]
    output = {
        'n_nodes': num_nodes,
        'n_vehicles': num_vehicles,
        'n_calls': num_calls,
        'Cargo': Cargo,
        'TravelTime': TravelTime,
        'FirstTravelTime': FirstTravelTime,
        'VesselCapacity': VesselCapacity,
        'LoadingTime': LoadingTime,
        'UnloadingTime': UnloadingTime,
        'VesselCargo': VesselCargo,
        'TravelCost': TravelCost,
        'FirstTravelCost': FirstTravelCost,
        'PortCost': PortCost
    }
    return output

def feasibility_check(solution, problem):
    """
    :rtype: tuple
    :param solution: The input solution of order of calls for each vehicle to the problem
    :param problem: The pickup and delivery problem object
    :return: whether the problem is feasible and the reason for probable infeasibility
    """
    num_vehicles = problem['n_vehicles']
    Cargo = problem['Cargo']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']
    VesselCapacity = problem['VesselCapacity']
    LoadingTime = problem['LoadingTime']
    UnloadingTime = problem['UnloadingTime']
    VesselCargo = problem['VesselCargo']
    solution = np.append(solution, [0])
    ZeroIndex = np.array(np.where(solution == 0)[0], dtype=int)
    feasibility = True
    tempidx = 0
    c = 'Feasible'
    for i in range(num_vehicles):
        currentVPlan = solution[tempidx:ZeroIndex[i]]
        currentVPlan = currentVPlan - 1
        NoDoubleCallOnVehicle = len(currentVPlan)
        tempidx = ZeroIndex[i] + 1
        if NoDoubleCallOnVehicle > 0:

            if not np.all(VesselCargo[i, currentVPlan]):
                feasibility = False
                c = 'incompatible vessel and cargo'
                break
            else:
                LoadSize = 0
                currentTime = 0
                sortRout = np.sort(currentVPlan, kind='mergesort')
                I = np.argsort(currentVPlan, kind='mergesort')
                Indx = np.argsort(I, kind='mergesort')
                LoadSize -= Cargo[sortRout, 2]
                LoadSize[::2] = Cargo[sortRout[::2], 2]
                LoadSize = LoadSize[Indx]
                if np.any(VesselCapacity[i] - np.cumsum(LoadSize) < 0):
                    feasibility = False
                    c = 'Capacity exceeded'
                    break
                Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
                Timewindows[0] = Cargo[sortRout, 6]
                Timewindows[0, ::2] = Cargo[sortRout[::2], 4]
                Timewindows[1] = Cargo[sortRout, 7]
                Timewindows[1, ::2] = Cargo[sortRout[::2], 5]

                Timewindows = Timewindows[:, Indx]

                PortIndex = Cargo[sortRout, 1].astype(int)
                PortIndex[::2] = Cargo[sortRout[::2], 0]
                PortIndex = PortIndex[Indx] - 1

                LU_Time = UnloadingTime[i, sortRout]
                LU_Time[::2] = LoadingTime[i, sortRout[::2]]
                LU_Time = LU_Time[Indx]
                Diag = TravelTime[i, PortIndex[:-1], PortIndex[1:]]
                FirstVisitTime = FirstTravelTime[i, int(Cargo[currentVPlan[0], 0] - 1)]

                RouteTravelTime = np.hstack((FirstVisitTime, Diag.flatten()))

                ArriveTime = np.zeros(NoDoubleCallOnVehicle)
                for j in range(NoDoubleCallOnVehicle):
                    ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows[0, j]))
                    if ArriveTime[j] > Timewindows[1, j]:
                        feasibility = False
                        c = 'Time window exceeded at call {}'.format(j)
                        break
                    currentTime = ArriveTime[j] + LU_Time[j]

    return feasibility, c

def cost_function(Solution, problem):
    """

    :param Solution: the proposed solution for the order of calls in each vehicle
    :param problem:
    :return:
    """
    num_vehicles = problem['n_vehicles']
    Cargo = problem['Cargo']
    TravelCost = problem['TravelCost']
    FirstTravelCost = problem['FirstTravelCost']
    PortCost = problem['PortCost']

    NotTransportCost = 0
    RouteTravelCost = np.zeros(num_vehicles)
    CostInPorts = np.zeros(num_vehicles)

    Solution = np.append(Solution, [0])
    ZeroIndex = np.array(np.where(Solution == 0)[0], dtype=int)
    tempidx = 0

    for i in range(num_vehicles + 1):
        currentVPlan = Solution[tempidx:ZeroIndex[i]]
        currentVPlan = currentVPlan - 1
        NoDoubleCallOnVehicle = len(currentVPlan)
        tempidx = ZeroIndex[i] + 1

        if i == num_vehicles:
            NotTransportCost = np.sum(Cargo[currentVPlan, 3]) / 2
        else:
            if NoDoubleCallOnVehicle > 0:
                sortRout = np.sort(currentVPlan, kind='mergesort')
                I = np.argsort(currentVPlan, kind='mergesort')
                Indx = np.argsort(I, kind='mergesort')

                PortIndex = Cargo[sortRout, 1].astype(int)
                PortIndex[::2] = Cargo[sortRout[::2], 0]
                PortIndex = PortIndex[Indx] - 1

                Diag = TravelCost[i, PortIndex[:-1], PortIndex[1:]]

                FirstVisitCost = FirstTravelCost[i, int(Cargo[currentVPlan[0], 0] - 1)]
                RouteTravelCost[i] = np.sum(np.hstack((FirstVisitCost, Diag.flatten())))
                CostInPorts[i] = np.sum(PortCost[i, currentVPlan]) / 2

    TotalCost = NotTransportCost + sum(RouteTravelCost) + sum(CostInPorts)
    return TotalCost

def initial_solution(problem):
    num_vehicles = problem['n_vehicles']
    solution = [0] * num_vehicles
    for i in range(problem['n_calls']):
        solution.append(i+1)
        solution.append(i+1)
    return solution

def split_into_vehicles(solution):
    vehicles = []
    current_vehicle = []
    for call in solution:
        if call == 0:
            vehicles.append(current_vehicle)
            current_vehicle = []
        else:
            current_vehicle.append(call)     
    vehicles.append(current_vehicle)
    return vehicles

def combine_vehicles(vehicles):
    solution = []
    for vehicle in vehicles:
        solution.extend(vehicle)
        solution.append(0)
    if solution[-1] == 0:
        solution.pop()
    return solution

def check_vehicle_feasibility(vehicle_plan, vehicle_idx, problem):
    """
    Check if a vehicle route is feasible without checking the entire solution.
    
    Args:
        vehicle_plan: List of calls for this vehicle (without the trailing 0)
        vehicle_idx: Index of the vehicle in the problem
        problem: Problem data
        
    Returns:
        bool: True if the vehicle route is feasible, False otherwise
    """
    if not vehicle_plan:
        return True
    
    Cargo = problem['Cargo']
    TravelTime = problem['TravelTime']
    FirstTravelTime = problem['FirstTravelTime']
    VesselCapacity = problem['VesselCapacity']
    LoadingTime = problem['LoadingTime']
    UnloadingTime = problem['UnloadingTime']
    VesselCargo = problem['VesselCargo']
    
    # Convert to 0-indexed call numbers
    currentVPlan = [c - 1 for c in vehicle_plan]
    
    NoDoubleCallOnVehicle = len(currentVPlan)
    
    if NoDoubleCallOnVehicle > 0:
        # Check if the vehicle can transport all calls
        if not np.all(VesselCargo[vehicle_idx, currentVPlan]):
            return False
        
        # Check capacity constraints
        LoadSize = 0
        currentTime = 0
        sortRout = np.sort(currentVPlan, kind='mergesort')
        I = np.argsort(currentVPlan, kind='mergesort')
        Indx = np.argsort(I, kind='mergesort')
        
        LoadSize -= Cargo[sortRout, 2]  # Negative for delivery (unloading)
        LoadSize[::2] = Cargo[sortRout[::2], 2]  # Positive for pickup (loading)
        LoadSize = LoadSize[Indx]
        
        # Check capacity constraints
        if np.any(VesselCapacity[vehicle_idx] - np.cumsum(LoadSize) < 0):
            return False
        
        # Check time window constraints
        Timewindows = np.zeros((2, NoDoubleCallOnVehicle))
        Timewindows[0] = Cargo[sortRout, 6]  # Delivery start time
        Timewindows[0, ::2] = Cargo[sortRout[::2], 4]  # Pickup start time
        Timewindows[1] = Cargo[sortRout, 7]  # Delivery end time
        Timewindows[1, ::2] = Cargo[sortRout[::2], 5]  # Pickup end time
        
        Timewindows = Timewindows[:, Indx]
        
        PortIndex = Cargo[sortRout, 1].astype(int)  # Delivery port
        PortIndex[::2] = Cargo[sortRout[::2], 0]  # Pickup port
        PortIndex = PortIndex[Indx] - 1  # 0-indexed
        
        LU_Time = UnloadingTime[vehicle_idx, sortRout]  # Unloading time
        LU_Time[::2] = LoadingTime[vehicle_idx, sortRout[::2]]  # Loading time
        LU_Time = LU_Time[Indx]
        
        if len(PortIndex) > 1:
            Diag = TravelTime[vehicle_idx, PortIndex[:-1], PortIndex[1:]]
            RouteTravelTime = Diag.flatten()
        else:
            RouteTravelTime = []
        
        FirstVisitTime = FirstTravelTime[vehicle_idx, int(Cargo[currentVPlan[0], 0] - 1)]
        RouteTravelTime = np.hstack((FirstVisitTime, RouteTravelTime))
        
        ArriveTime = np.zeros(NoDoubleCallOnVehicle)
        for j in range(NoDoubleCallOnVehicle):
            ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows[0, j]))
            if ArriveTime[j] > Timewindows[1, j]:
                return False
            currentTime = ArriveTime[j] + LU_Time[j]
    
    return True

#Removal
def remove_calls(solution, problem, min_percent, max_percent):
    """
    Removes a random set of calls from the current solution.
    """
    new_solution = solution.copy()
    vehicles = split_into_vehicles(new_solution)
    num_calls = problem['n_calls']
    call_ids = list(range(1, num_calls + 1))

    percentage = random.uniform(min_percent, max_percent)
    num_calls_to_remove = int(num_calls * percentage / 100)
    num_calls_to_remove = min(num_calls_to_remove, len(call_ids))  # unngå over-sampling -- hva betyr det

    chosen_calls = random.sample(call_ids, num_calls_to_remove)

    for call in chosen_calls:
        for vehicle in vehicles:
            while call in vehicle:
                vehicle.remove(call)

    return combine_vehicles(vehicles), chosen_calls

def remove_costly1(solution, problem, num_calls_to_remove):
    """
    Fjerner de mest kostbare samtalene (2 forekomster) fra løsningen.

    Args:
        solution (list): Nåværende løsning (flat liste med alle kjøretøy)
        problem (dict): Problemdata
        num_calls_to_remove (int): Antall samtaler som skal fjernes

    Returns:
        Tuple[new_solution, List[int]]: Ny løsning og liste med fjernede samtaler
    """
    new_solution = solution.copy()
    vehicles = split_into_vehicles(new_solution)
    removed_calls = []

    for _ in range(num_calls_to_remove):
        all_calls = set(call for vehicle in vehicles for call in vehicle)
        call_scores = []

        for call in all_calls:
            # Finn kjøretøyet som har begge forekomster
            for idx, vehicle in enumerate(vehicles):
                if vehicle.count(call) == 2:
                    temp_vehicle = vehicle.copy()
                    temp_vehicle = [c for c in temp_vehicle if c != call]
                    temp_vehicles = vehicles.copy()
                    temp_vehicles[idx] = temp_vehicle
                    temp_solution = combine_vehicles(temp_vehicles)

                    original_cost = cost_function(combine_vehicles(vehicles), problem)
                    new_cost = cost_function(temp_solution, problem)
                    improvement = original_cost - new_cost

                    call_scores.append((improvement, call, idx))
                    break  # vi trenger bare én vehicle med begge forekomster

        if not call_scores:
            break  # Ingen igjen å fjerne

        # Velg den callen som gir størst forbedring
        call_scores.sort(reverse=True)
        _, best_call, best_vehicle_idx = call_scores[0]

        # Fjern best_call fra det riktige kjøretøyet
        vehicles[best_vehicle_idx] = [c for c in vehicles[best_vehicle_idx] if c != best_call]
        removed_calls.append(best_call)

    final_solution = combine_vehicles(vehicles)
    return final_solution, removed_calls

def remove_costly(solution, problem, min_percent, max_percent):
    """
    Fjerner de mest kostbare samtalene (2 forekomster) fra løsningen, 
    og reintegrerer dem med valgt metode (greedy eller k-regret).

    Args:
        solution (list): Nåværende løsning (flat liste med alle kjøretøy)
        problem (dict): Problemdata
        min_percent (float): Minimum prosentandel samtaler å fjerne
        max_percent (float): Maksimum prosentandel samtaler å fjerne
        reinsertion_method (str): 'greedy' eller 'k_regret'
        k (int): k-verdi brukt ved k-regret

    Returns:
        list: Ny løsning etter fjerning og reintegrering
    """
    new_solution = solution.copy()
    vehicles = split_into_vehicles(new_solution)
    
    all_calls = list(set(call for vehicle in vehicles for call in vehicle if call != 0))
    num_calls_to_remove = max(1, int(len(all_calls) * random.uniform(min_percent, max_percent) / 100))
    removed_calls = []

    current_cost = cost_function(combine_vehicles(vehicles), problem)
    call_scores = []

    # Beregn kostnaden av å fjerne hver call
    for call in all_calls:
        for idx, vehicle in enumerate(vehicles):
            if vehicle.count(call) == 2:
                temp_vehicle = [c for c in vehicle if c != call]
                temp_vehicles = [v.copy() for v in vehicles]
                temp_vehicles[idx] = temp_vehicle
                temp_solution = combine_vehicles(temp_vehicles)
                new_cost = cost_function(temp_solution, problem)
                improvement = current_cost - new_cost
                heapq.heappush(call_scores, (-improvement, call, idx))  

    if not call_scores:
        return solution  # Returner original løsning hvis ingen calls kan fjernes

    # Fjern de mest kostbare callsene
    for _ in range(num_calls_to_remove):
        if not call_scores:
            break
        _, best_call, best_vehicle_idx = heapq.heappop(call_scores)  # Pop den mest kostbare
        vehicles[best_vehicle_idx] = [c for c in vehicles[best_vehicle_idx] if c != best_call]
        removed_calls.append(best_call)

    reduced_solution = combine_vehicles(vehicles)

    return reduced_solution, removed_calls

#Insertion
def greedy(solution, problem, chosen_calls):
    """
    Greedy reinsert each of the calls removed earlier into the best possible position
    in a compatible vehicle. If not feasible, place it in the dummy vehicle.
    """
    new_solution = solution.copy()
    vehicles = split_into_vehicles(new_solution)
    dummy_vehicle_index = len(vehicles) - 1
    dummy_vehicle = vehicles[dummy_vehicle_index]

    for chosen_call in chosen_calls:
        best_cost = float('inf')
        best_vehicles = None

        compatible_vehicles = [
            (v_idx, v) for v_idx, v in enumerate(vehicles[:dummy_vehicle_index])
            if problem['VesselCargo'][v_idx][chosen_call - 1]
        ]

        for v_idx, to_vehicle in compatible_vehicles:
            for i in range(len(to_vehicle) + 1):
                for j in range(i, len(to_vehicle) + 1):
                    temp_vehicle = to_vehicle[:i] + [chosen_call] + to_vehicle[i:j] + [chosen_call] + to_vehicle[j:]
                    if check_vehicle_feasibility(temp_vehicle, v_idx, problem):
                        temp_vehicles = [v.copy() for v in vehicles]
                        temp_vehicles[v_idx] = temp_vehicle
                        candidate_solution = combine_vehicles(temp_vehicles)
                        candidate_cost = cost_function(candidate_solution, problem)

                        if candidate_cost < best_cost:
                            best_cost = candidate_cost
                            best_vehicles = temp_vehicles

        if best_vehicles:
            vehicles = best_vehicles
        else:
            dummy_vehicle += [chosen_call, chosen_call]
            vehicles[dummy_vehicle_index] = dummy_vehicle

    return combine_vehicles(vehicles)

def k_regret(solution, problem, chosen_calls, k):
    """
    K-regret reinsertion heuristic for the pickup and delivery problem.

    Args:
        solution: Current solution (list of call indices with 0 as vehicle separator)
        problem: Problem instance data
        chosen_calls: List of calls that were removed and need to be reinserted
        k: Number of best insertions to consider for regret calculation

    Returns:
        list: New solution after applying the K-regret heuristic
    """
    new_solution = solution.copy()
    vehicles = split_into_vehicles(new_solution)
    dummy_vehicle_index = len(vehicles) - 1
    dummy_vehicle = vehicles[dummy_vehicle_index]

    unplaced_calls = chosen_calls.copy()

    while unplaced_calls:
        insertions_by_call = {}

        # Finn de k-beste plasseringene for hver call
        for call in unplaced_calls:
            insertions = []

            compatible_vehicles = [
                (v_idx, v) for v_idx, v in enumerate(vehicles[:dummy_vehicle_index])
                if problem['VesselCargo'][v_idx][call - 1]
            ]

            for v_idx, vehicle in compatible_vehicles:
                for i in range(len(vehicle) + 1):
                    for j in range(i, len(vehicle) + 1):
                        temp_vehicle = vehicle[:i] + [call] + vehicle[i:j] + [call] + vehicle[j:]
                        if check_vehicle_feasibility(temp_vehicle, v_idx, problem):
                            temp_vehicles = [v.copy() for v in vehicles]
                            temp_vehicles[v_idx] = temp_vehicle
                            candidate_solution = combine_vehicles(temp_vehicles)
                            candidate_cost = cost_function(candidate_solution, problem)
                            insertions.append((candidate_cost, v_idx, i, j))

            insertions.sort()
            if insertions:
                insertions_by_call[call] = insertions[:k]  # lagre kun k beste

        if not insertions_by_call:
            break  # ingen mulig plassering

        # Finn konfliktende beste plasseringer
        position_map = {}
        for call, insertions in insertions_by_call.items():
            pos = tuple(insertions[0][1:])  # (v_idx, i, j)
            position_map.setdefault(pos, []).append((call, insertions))

        # Behandle alle plasseringer
        placed_calls = []
        for pos, calls_with_insertions in position_map.items():
            if len(calls_with_insertions) == 1:
                # Bare én call vil ha denne posisjonen – sett inn direkte
                call, insertions = calls_with_insertions[0]
                _, v_idx, i, j = insertions[0]
                vehicles[v_idx] = vehicles[v_idx][:i] + [call] + vehicles[v_idx][i:j] + [call] + vehicles[v_idx][j:]
                placed_calls.append(call)
            else:
                # Flere calls ønsker samme posisjon – velg den med høyest regret
                regret_list = []
                for call, insertions in calls_with_insertions:
                    if len(insertions) >= 2:
                        regret = insertions[1][0] - insertions[0][0]
                    else:
                        regret = float('inf')  # kun én mulig plassering
                    regret_list.append((regret, call, insertions))

                regret_list.sort(reverse=True)
                _, chosen_call, chosen_insertions = regret_list[0]
                _, v_idx, i, j = chosen_insertions[0]
                vehicles[v_idx] = vehicles[v_idx][:i] + [chosen_call] + vehicles[v_idx][i:j] + [chosen_call] + vehicles[v_idx][j:]
                placed_calls.append(chosen_call)

        # Fjern plasserte calls
        for call in placed_calls:
            if call in unplaced_calls:
                unplaced_calls.remove(call)

    # Hvis det fortsatt er calls som ikke ble plassert, sett dem i dummy vehicle
    for call in unplaced_calls:
        dummy_vehicle += [call, call]
    vehicles[dummy_vehicle_index] = dummy_vehicle

    final_solution = combine_vehicles(vehicles)

    # Valider at alle chosen_calls faktisk er i løsningen
    all_calls = set()
    for val in final_solution:
        if val != 0:
            all_calls.add(val)
    for call in chosen_calls:
        if final_solution.count(call) != 2:
            raise ValueError(f"Call {call} ble ikke riktig reintegrert i løsningen.")

    return final_solution

#Greedy
def remove_small(solution, problem):
  
    """
    Greedy reinsertion of a randomly selected call into the best position 
    of one compatible vehicle (or dummy if none are valid).
    """
    #print(f"Solution before remove small: {solution}")
    temp_sol, chosen_calls = remove_calls(solution, problem, 5, 10)
    repaired_solution = greedy(temp_sol, problem, chosen_calls)
    #print(f"Solution after remove small: {repaired_solution}")
    return repaired_solution

def remove_medium(solution, problem):
    """
    Remove a medium number of calls and then reinsert them using a greedy approach.
    """
    new_solution, chosen_calls = remove_calls(solution, problem, 10, 25)
    return greedy(new_solution, problem, chosen_calls)

def remove_large(solution, problem):
    """
    Remove a large number of calls and then reinsert them using a greedy approach.
    """
    new_solution, chosen_calls = remove_calls(solution, problem, 25, 50)
    return greedy(new_solution, problem, chosen_calls)

def remove_all(solution, problem):
    """
    Remove all calls and then reinsert them using a greedy approach.
    """
    new_solution, chosen_calls = remove_calls(solution, problem, 50, 100)
    return greedy(new_solution, problem, chosen_calls)

def remove_random(solution, problem):
    """
    Remove a random number of calls and then reinsert them using a greedy approach.
    """
    new_solution, chosen_calls = remove_calls(solution, problem, 5, 50)
    return greedy(new_solution, problem, chosen_calls)

#regret
def remove_small_regret(solution, problem):
    #print(f"Solution before remove small regret: {solution}")
    temp_sol, chosen_calls = remove_calls(solution, problem, 5, 10)
    return k_regret(temp_sol, problem, chosen_calls, 2)

def remove_medium_regret(solution, problem):
    temp_sol, chosen_calls = remove_calls(solution, problem, 10, 25)
    return k_regret(temp_sol, problem, chosen_calls, 2)

def remove_large_regret(solution, problem):
    temp_sol, chosen_calls = remove_calls(solution, problem, 25, 50)
    return k_regret(temp_sol, problem, chosen_calls, 2)

def remove_random_regret(solution, problem):
    temp_sol, chosen_calls = remove_calls(solution, problem, 5, 50)
    return k_regret(temp_sol, problem, chosen_calls, 2)

#costly
def costly_small(solution, problem):
    """
    Remove a small number of calls and then reinsert them using the costly method.
    """
    new_solution, chosen_calls = remove_costly(solution, problem, 5, 10)
    return greedy(new_solution, problem, chosen_calls)

def costly_medium(solution, problem):   
    """
    Remove a medium number of calls and then reinsert them using the costly method.
    """
    new_solution, chosen_calls = remove_costly(solution, problem, 10, 15)
    return greedy(new_solution, problem, chosen_calls)

def costly_large(solution, problem):
    """
    Remove a large number of calls and then reinsert them using the costly method.
    """
    new_solution, chosen_calls = remove_costly(solution, problem, 25, 50)
    return greedy(new_solution, problem, chosen_calls)

def costly_random(solution, problem):
    """
    Remove a large number of calls and then reinsert them using the costly method.
    """
    new_solution, chosen_calls = remove_costly(solution, problem, 5, 50)
    return greedy(new_solution, problem, chosen_calls)


def General_Adaptive_Metahuristics_Framework(problem, initial_solution):
    """ General Adaptive Metaheuristics Framework for Pickup and Delivery Problem with Adaptive Operator Selection """

    # Parameters
    max_iterations = 10000
    escape_condition = 500
    score_update_interval = 100

    iteration = 0
    iterations_since_best = 0
    escape_intensity = 0
    best_iteration = 0

    current_solution = initial_solution.copy()
    current_cost = cost_function(current_solution, problem)
    best_solution = initial_solution.copy()
    best_cost = current_cost

    #Plotting
    cost_history = []
    acceptance_iter_history = []
    acceptance_prob_history = []

    operators = [
        remove_small,
        remove_medium,
        remove_large,
        remove_all,
        # remove_random,
        remove_small_regret,
        remove_medium_regret,
        remove_large_regret,
        # remove_random_regret,
        costly_small,
        costly_medium,
        costly_large,
        # costly_random
    ]
    
    operator_names = [op.__name__ for op in operators]
    num_operators = len(operators)
    operator_scores_raw = [0 for _ in range(num_operators)]
    operator_scores_normalized = normalize_scores(operator_scores_raw)

    #Plotting
    operator_scores_history = {name: [] for name in operator_names}
    operator_improvements = [0 for _ in range(num_operators)]   
    operator_deltas = {name: [] for name in operator_names}
    operator_delta_iters = {name: [] for name in operator_names}
    
    seen_solutions = set()
    seen_solutions.add(str(initial_solution)) 

    #Iterations
    while iteration < max_iterations:
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: Current cost: {current_cost}, Best cost: {best_cost}")
        #Escape
        if iterations_since_best > escape_condition :
            escape_intensity += 1
            print(f"Iteration {iteration}: Escape triggered")
            while True:
                #escape_solution = escape(current_solution, problem, iterations_since_best)
                escape_solution = escape1(current_solution, problem, iterations_since_best, escape_intensity)
                feasible, _ = feasibility_check(escape_solution, problem)
                if feasible:
                    break
            current_solution = escape_solution # er dette permanent skadelig
            current_cost = cost_function(current_solution, problem)
            iterations_since_best = 0

        selected_operator = select_heuristic(operator_scores_normalized, iteration, max_iterations)
        operator_name = operator_names[selected_operator]

        assert 0 <= selected_operator < num_operators, f"Invalid operator index: {selected_operator}"


        # Velg operatør og anvend den på løsningen
        new_solution = operators[selected_operator](current_solution.copy(), problem)
        new_cost = cost_function(new_solution, problem)
        feasible, _ = feasibility_check(new_solution, problem)

        if feasible:
            #operator_name = operator_names[selected_operator]
            delta = 0.2 * ((max_iterations - iteration) / max_iterations) * best_cost
            accepted = False
            delta_cost = new_cost - current_cost

            solution_id = str(new_solution)
            is_unique = solution_id not in seen_solutions

            operator_scores_raw[selected_operator] # New best solution
            #New best cost
            if new_cost < best_cost:
                #delta_cost = new_cost - current_cost
                best_cost = new_cost
                best_solution = new_solution.copy()
                best_iteration = iteration
                current_solution = new_solution.copy()
                current_cost = new_cost
                iterations_since_best = -1 #kan den være -1 siden jeg legger til +1 senere
                accepted = True
                operator_scores_raw[selected_operator] += 4 # New best solution
                print(f"New best solution found with cost {best_cost}")

            #New cost
            elif new_cost < best_cost + delta:
                current_solution = new_solution
                current_cost = new_cost
                accepted = True
                operator_scores_raw[selected_operator] += 2 #New accepted solution
                acceptance_iter_history.append(iteration)

            if accepted: #Trenger jeg kun å gi score til de unike som er akseptert? eller også unike som ikke blir akseptert? de tas jo ikke med videre
                if is_unique:
                    operator_scores_raw[selected_operator] +=1  # Unique solution
                    seen_solutions.add(solution_id)

                operator_deltas[operator_name].append(delta_cost) # er dette riktig måte å plotte score for hver operatør?
                operator_delta_iters[operator_name].append(iteration)

        iterations_since_best += 1
        iteration += 1
        cost_history.append(best_cost)

    

        # if iteration % score_update_interval == 0:
        #     for i in range(num_operators):
        #         operator_scores[i] += operator_improvements[i]
        #     normalize_scores(operator_scores)
        #     for i, name in enumerate(operator_names):
        #         operator_scores_history[name].append(operator_scores[i])
        #     operator_improvements = [0 for _ in range(num_operators)]

        #Operator score update
        if iteration % score_update_interval == 0:
            operator_scores_normalized = normalize_scores(operator_scores_raw)
            operator_scores_raw = [0 for _ in range(num_operators)]
            # #Plotting
            for i, name in enumerate(operator_names):
                operator_scores_history[name].append(operator_scores_normalized[i])
            #Reset operator improvements 

    return (
        best_solution,
        operator_scores_history,
        cost_history,
        acceptance_iter_history,
        acceptance_prob_history,
        operator_deltas,
        operator_delta_iters,
        best_iteration
    )               

def escape1(current_solution, problem, iterations_since_best, escape_intensity):
    """  
    Args:
        current_solution: Nåværende løsning
        problem: Probleminstans
        iterations_since_best: Antall iterasjoner siden siste forbedring
    
    Returns:
        Ny løsning etter unnslupping
    """

    escape_solution = current_solution.copy()
    
    intensity = min(escape_intensity / 10, 1.0)

    
    mild_escapes = [remove_small]
    medium_escapes = [remove_medium]
    strong_escapes = [remove_medium_regret] #?????

    if escape_intensity < 0.3:
        escape_methods = mild_escapes
    elif escape_intensity < 0.7:
        escape_methods = mild_escapes + medium_escapes
    else:
        escape_methods = mild_escapes + medium_escapes + strong_escapes

    weights = []
    for method in escape_methods:
        if method in mild_escapes:
            weights.append(1.0)
        elif method in medium_escapes:
            weights.append(1.0 + 2 * intensity)
        else: 
            weights.append(1.0 + 4 * intensity)

    num_escapes = int(3 + intensity * 35)
    successful_escapes = 0
    
    for i in range(num_escapes):
        method = remove_small
        proposed_solution = method(escape_solution, problem)
        feasible, _ = feasibility_check(proposed_solution, problem)

        if feasible:
                escape_solution = proposed_solution
                successful_escapes += 1

    if successful_escapes == 0:
        fallback_solution = remove_small(escape_solution, problem)
        if feasibility_check(fallback_solution, problem)[0]:
            escape_solution = fallback_solution
    
    return escape_solution


def select_heuristic(operator_scores, iteration, max_iterations):
    """
    Mer sofistikert heuristikkvalg med adaptive parametere
    
    Args:
        operator_scores: Nåværende operator scores
        iteration: Nåværende iterasjon
        max_iterations: Maksimalt antall iterasjoner
    
    Returns:
        Valgt operator indeks
    """
    # Implementer eksplorasjon vs utnyttelse
    exploration_rate = 1.0 - (iteration / max_iterations)
    
    if random.random() < exploration_rate:
        return random.randint(0, len(operator_scores) - 1)
    else:
        # Bruk roulette wheel, men med mer dynamisk vekting
        indices = list(range(len(operator_scores)))
        
        # Legg til en liten tilfeldighetsfaktor
        weighted_scores = [
            score * (1 + random.uniform(-0.1, 0.1)) 
            for score in operator_scores
        ]
        
        return random.choices(indices, weights=weighted_scores, k=1)[0]

def normalize_scores(operator_scores):
    scores = np.array(operator_scores)
    exp = np.exp(scores - np.max(scores))
    normalized = exp / np.sum(exp)
    return normalized