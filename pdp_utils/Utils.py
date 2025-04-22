import numpy as np
import random
import math
import copy
import matplotlib as plt


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


#operators

def dummy_reinsert(solution, problem):
    """
    Takes a random call from the dummy vehicle and tries to insert it
    into the first better position in the actual vehicles.
    
    Args:
        solution: Current solution (list of call indices with 0 as vehicle separator)
        problem: Problem instance data
        
    Returns:
        list: New solution after reinsertion
    """

    #print(f"Solution before dummy reinsert: {solution}")
    new_solution = solution.copy()

    vehicles = split_into_vehicles(new_solution)
    
    dummy_vehicle = vehicles[-1]
    actual_vehicles = vehicles[:-1]
    
    if not dummy_vehicle:
        return solution
    
    call_pairs = {}
    for call in dummy_vehicle:
        call_id = abs(call)
        if call_id in call_pairs:
            call_pairs[call_id].append(call)
        else:
            call_pairs[call_id] = [call]
    
    if not call_pairs:
        return solution
    
    random_call_id = random.choice(list(call_pairs.keys()))
    call_pair = call_pairs[random_call_id]
    
    for call in call_pair:
        dummy_vehicle.remove(call)
    
    best_cost = float('inf')
    best_vehicles = None
    
    for v_idx, vehicle in enumerate(actual_vehicles):
        if not np.all(problem['VesselCargo'][v_idx, [c-1 for c in call_pair]]):
            continue
        
        for pickup_pos in range(len(vehicle) + 1):
            pickup_vehicle = vehicle.copy()
            pickup_vehicle.insert(pickup_pos, call_pair[0])
            
            if check_vehicle_feasibility(pickup_vehicle, v_idx, problem):
                for delivery_pos in range(pickup_pos + 1, len(pickup_vehicle) + 1):
                    test_vehicle = pickup_vehicle.copy()
                    test_vehicle.insert(delivery_pos, call_pair[0])
                    
                    if check_vehicle_feasibility(test_vehicle, v_idx, problem):
                        test_vehicles = actual_vehicles.copy()
                        test_vehicles[v_idx] = test_vehicle
                        test_vehicles.append(dummy_vehicle)
                        test_solution = combine_vehicles(test_vehicles)
                        
                        test_cost = cost_function(test_solution, problem)
                        
                        if test_cost < best_cost:
                            best_cost = test_cost
                            best_vehicles = test_vehicles.copy()
                            break
            
            # If we found a better solution, stop searching
            if best_vehicles is not None:
                break
                
        # If we found a better solution, stop searching
        if best_vehicles is not None:
            break
    
    # If we found a better solution, return it
    if best_vehicles is not None:
        return combine_vehicles(best_vehicles)
    
    # If no better solution was found, put the call back in dummy vehicle
    dummy_vehicle.extend(call_pair)

    #print(f"Solution after dummy reinsert: {new_solution}")
    return solution

def swap_calls(solution, problem):
    """
    Swaps the position of two random calls either within the same vehicle 
    or between different vehicles.
    
    Args:
        solution: Current solution (list of call indices with 0 as vehicle separator)
        problem: Problem instance data
        
    Returns:
        list: New solution after swap
    """

    #print(f"Solution before swap: {solution}")
    # Create a copy of the solution
    new_solution = solution.copy()

    num_calls = problem['n_calls']
    
    #Kan jeg finne en bedre måte å velge hvilke som skal swapes?
    call_one = random.choice(range(1, num_calls + 1))
    #print(f"Call one: {call_one}")
    call_two = random.choice(range(1, num_calls + 1))
    #print(f"Call two: {call_two}")
    
    if call_one == call_two:
        return new_solution
    
    if call_one == 0 or call_two == 0:
        return new_solution
    
    # Find the vehicles that contain the calls
    vehicles = split_into_vehicles(new_solution)


    #returnerer en liste med indeksen til vehicle som inneholder call_one
    vehicle_one = [i for i, v in enumerate(vehicles) if call_one in v]
    vehicle_two = [i for i, v in enumerate(vehicles) if call_two in v]


    if vehicle_one == vehicle_two:
        vehicle_idx = vehicle_one[0]
        v = vehicles[vehicle_idx] 
        swapped_list = [call_one if x == call_two else call_two if x == call_one else x for x in v]

        vehicles[vehicle_idx] = swapped_list
    else:   
        for v in vehicle_one:
            if v == call_one:
                v = call_two
        for v in vehicle_two:
            if v == call_two:
                v = call_one

    new_solution = combine_vehicles(vehicles)

    #print(f"Solution after swap: {new_solution}")
    return new_solution

def shuffle_vehicle(solution, problem):

    #print(f"Solution before shuffle: {solution}")
    new_solution = solution.copy()
    vehicles = split_into_vehicles(new_solution)

    vehicle_idx = random.choice(range(len(vehicles)))
    vehicle = vehicles[vehicle_idx]


    random.shuffle(vehicle)
    vehicles[vehicle_idx] = vehicle

    new_solution = combine_vehicles(vehicles)

    #print(f"Solution after shuffle: {new_solution}")
    return new_solution

def one_reinsert(solution, problem):
    """
    Gjør en liten endring i løsningen ved å flytte en tilfeldig valgt forespørsel fra ett kjøretøy til et annet.
    """
    new_solution = copy.deepcopy(solution)

    vehicles = split_into_vehicles(new_solution)
    
    non_empty_vehicles = [v for v in vehicles if v] #tomme lister blir en boolsk verdi i python
    if not non_empty_vehicles:
        return new_solution 
        
    #velg tilfeldig bil
    from_vehicle = random.choice(non_empty_vehicles)

    chosen_call = random.choice([c for c in from_vehicle if c != 0])

    if from_vehicle.count(chosen_call) != 2:

        raise ValueError(f"Feil: chosen_call {chosen_call} finnes ikke to ganger i from_vehicle {from_vehicle}")
    
    from_vehicle[:] = [c for c in from_vehicle if c != chosen_call]

    to_vehicle = random.choice(vehicles)

    if not to_vehicle:  # Hvis kjøretøyet er tomt, legg inn på start
        to_vehicle.append(chosen_call)
        to_vehicle.append(chosen_call)
    else:
        insert_index1 = random.randint(0, len(to_vehicle))
        insert_index2 = random.randint(0, len(to_vehicle))
        while insert_index1 == insert_index2:  # Forsikre oss om at indeksene er forskjellige
            insert_index2 = random.randint(0, len(to_vehicle))

        to_vehicle.insert(min(insert_index1, insert_index2), chosen_call)
        to_vehicle.insert(max(insert_index1, insert_index2), chosen_call)

    new_solution = combine_vehicles(vehicles)

    assert new_solution.count(chosen_call) == 2, f"Feil: {chosen_call} finnes {new_solution.count(chosen_call)} ganger!"
    return new_solution

def greedy_reinsert(solution, problem):
    """
    Greedy reinsertion of a random call into the best position of one random compatible vehicle.
    If no feasible position is found, inserts it into the dummy vehicle.
    """
    #print(f"Solution before greedy reinsert: {solution}")
    new_solution = solution.copy()
    vehicles = split_into_vehicles(new_solution)
    dummy_vehicle_index = len(vehicles) - 1
    dummy_vehicle = vehicles[dummy_vehicle_index]

    # Velg tilfeldig kjøretøy som har minst én call med to forekomster
    # from_vehicle = random.choice([v for v in vehicles if sum(v.count(c) == 2 for c in set(v)) > 0])
    from_vehicle = random.choice([v for v in vehicles if len(v) != 0])
    chosen_call = random.choice([c for c in set(from_vehicle)])

    # Fjern call fra originalt kjøretøy
    from_vehicle[:] = [c for c in from_vehicle if c != chosen_call]

    # Finn kompatible kjøretøy (utenom dummy)
    compatible_vehicle_indices = [
        idx for idx in range(dummy_vehicle_index)
        if problem['VesselCargo'][idx][chosen_call - 1]
    ]

    if not compatible_vehicle_indices:
        # Ingen kompatible biler – legg rett i dummy
        dummy_vehicle += [chosen_call, chosen_call]
        vehicles[dummy_vehicle_index] = dummy_vehicle
        return combine_vehicles(vehicles)

    # Velg én tilfeldig kompatibel bil
    to_idx = random.choice(compatible_vehicle_indices)
    to_vehicle = vehicles[to_idx]

    best_insertion = None
    best_cost = float('inf')

    for i in range(len(to_vehicle)+1):
        for j in range(i, len(to_vehicle)+1):
            temp_vehicle = to_vehicle[:i] + [chosen_call] + to_vehicle[i:j] + [chosen_call] + to_vehicle[j:]
            if check_vehicle_feasibility(temp_vehicle, to_idx, problem):
                temp_vehicles = vehicles.copy()
                temp_vehicles[to_idx] = temp_vehicle
                candidate_solution = combine_vehicles(temp_vehicles)
                candidate_cost = cost_function(candidate_solution, problem)

                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_insertion = (i, j)

    if best_insertion:
        i, j = best_insertion
        to_vehicle.insert(i, chosen_call)
        to_vehicle.insert(j, chosen_call)
        new_solution = combine_vehicles(vehicles)

        if cost_function(new_solution, problem) < cost_function(solution, problem):...
           # print(f"New better solution found: {cost_function(new_solution, problem)}")
        else:...
            #print(f"Worse solution accepted with cost {cost_function(new_solution, problem)}")
        return new_solution
    
    

    #print(f"No feasible insertion for call {chosen_call}, placed in dummy.")
    dummy_vehicle += [chosen_call, chosen_call]
    vehicles[dummy_vehicle_index] = dummy_vehicle
    return combine_vehicles(vehicles)
    # return solution


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
    num_calls_to_remove = min(num_calls_to_remove, len(call_ids))  # unngå over-sampling

    chosen_calls = random.sample(call_ids, num_calls_to_remove)


    for call in chosen_calls:
        for vehicle in vehicles:
            while call in vehicle:
                vehicle.remove(call)

    return combine_vehicles(vehicles), chosen_calls

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
#burde det være mulig å plasserre i dummy her også? 
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


def General_Adaptive_Metahuristics_Framework(problem, initial_solution):
    """ General Adaptive Metaheuristics Framework for Pickup and Delivery Problem with Adaptive Operator Selection """
    
    # Parametere
    max_iterations = 10000
    escape_condition = 1000
    score_update_interval = 100

    # Initielle løsninger og kostnader
    current_solution = initial_solution.copy()
    current_cost = cost_function(current_solution, problem)

    best_solution = initial_solution.copy()
    best_cost = current_cost

    # Historikk for plott og analyse
    cost_history = []
    acceptance_iter_history = []
    acceptance_prob_history = []

    iteration = 0
    iterations_since_best = 0
    best_iteration = 0

    # Heuristiske operatorer
    operators = [
        shuffle_vehicle,
        #swap_calls,
        #dummy_reinsert,
        one_reinsert,
        greedy_reinsert,
        remove_small,
        remove_medium,
        remove_large,
        #remove_all
    ]
    
    operator_names = [op.__name__ for op in operators]
    num_operators = len(operators)
    operator_scores = [1.0 for _ in range(num_operators)]
    normalize_scores(operator_scores)
    
    operator_scores_history = {name: [] for name in operator_names}
    operator_improvements = [0 for _ in range(num_operators)]
    
    operator_deltas = {name: [] for name in operator_names}
    operator_delta_iters = {name: [] for name in operator_names}

    while iteration < max_iterations:
        # Escape-mekanisme
        if iterations_since_best > escape_condition :
            print(f"Iteration {iteration}: Escape triggered")
            while True:
                escape_solution = escape(current_solution, problem, iterations_since_best)
                feasible, _ = feasibility_check(escape_solution, problem)
                if feasible:
                    break
            current_solution = escape_solution
            current_cost = cost_function(current_solution, problem)
            iterations_since_best = 0

        # Velg operator og generer ny løsning
        selected_operator = select_heuristic(operator_scores, iteration, max_iterations)
        assert 0 <= selected_operator < num_operators, f"Invalid operator index: {selected_operator}"
        # for i in range(max(min(iterations_since_best // 5, problem['n_calls']), 1)):
        new_solution = operators[selected_operator](current_solution.copy(), problem)
        new_cost = cost_function(new_solution, problem)
        feasible, _ = feasibility_check(new_solution, problem)

        if feasible:
            operator_name = operator_names[selected_operator]
            delta = 0.2 * ((max_iterations - iteration) / max_iterations) * best_cost
            accepted = False

            if new_cost < best_cost:
                delta_cost = new_cost - current_cost
                best_cost = new_cost
                best_solution = new_solution.copy()
                best_iteration = iteration
                current_solution = new_solution.copy()
                current_cost = new_cost
                iterations_since_best = 0
                operator_improvements[selected_operator] += 1 
                accepted = True
                print(f"New best solution found with cost {best_cost}")

            elif new_cost < best_cost + delta:
                delta_cost = new_cost - current_cost
                current_solution = new_solution
                current_cost = new_cost
                accepted = True
                acceptance_iter_history.append(iteration)

            if accepted:
                operator_scores[selected_operator] += 1
                operator_deltas[operator_name].append(delta_cost)
                #acceptance_prob_history.append(delta_cost / best_cost)
                operator_delta_iters[operator_name].append(iteration)

        iterations_since_best += 1
        iteration += 1
        cost_history.append(best_cost)

        # Oppdater operatørscore periodisk
        if iteration % score_update_interval == 0:
            for i in range(num_operators):
                operator_scores[i] += operator_improvements[i]
            normalize_scores(operator_scores)
            for i, name in enumerate(operator_names):
                operator_scores_history[name].append(operator_scores[i])
            operator_improvements = [0 for _ in range(num_operators)]

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

def escape1(current_solution, problem, iterations_since_best):
    """  
    Args:
        current_solution: Nåværende løsning
        problem: Probleminstans
        iterations_since_best: Antall iterasjoner siden siste forbedring
    
    Returns:
        Ny løsning etter unnslupping
    """
    escape_solution = current_solution.copy()
    
    # Øk dramatikken i unnslupping basert på hvor lenge siden siste forbedring
    escape_intensity = min(iterations_since_best / 50, 1.0)
    
    # Velg operatorer med økende intensitet
    escape_methods = [
        #shuffle_vehicle,
        #swap_calls,
        #dummy_reinsert,
        #one_reinsert,
        #greedy_reinsert,
        remove_all,
        #remove_small,
        remove_medium,
        remove_large,
    ]
    
    # Antall ganger vi kjører unnsluppe-operatorer øker med intensitet
    num_escapes = int(2 + escape_intensity * 5)
    
    for _ in range(num_escapes):
        # Vekt operatorene basert på intensitet
        weights = [
            1.0, 
            1.0 + escape_intensity, 
            1.0 + 2 * escape_intensity,
            1.0 + 3 * escape_intensity,
            1.0 + 4 * escape_intensity,
            1.0 + 5 * escape_intensity,
        ]
        
        #escape_method = random.choices(escape_methods, weights)[0]
        escape_method = escape_methods[0]  # For testing purposes, use only the first method
        proposed_solution = escape_method(escape_solution, problem)
        if feasibility_check(proposed_solution, problem)[0]:
            escape_solution = proposed_solution
    
    return escape_solution

import random

def escape(current_solution, problem, iterations_since_best):
    """  
    Escapes from local optimum by applying strong diversification moves.

    Args:
        current_solution: Current solution
        problem: Problem instance
        iterations_since_best: Number of iterations since last improvement
    
    Returns:
        New solution after escape
    """
    escape_solution = current_solution.copy()
    
    # Scale intensity from 0 to 1
    escape_intensity = min(iterations_since_best / 50, 1.0)
    
    # List of available escape operators
    escape_methods = [
        remove_small,
        remove_medium,
        remove_large,
        remove_all,
        swap_calls,
        shuffle_vehicle,
        one_reinsert,
        greedy_reinsert,
    ]
    
    # Weight operators: higher weight means more likely to be chosen
    weights = {
        remove_small: 1.0 + escape_intensity * 1,
        remove_medium: 1.0 + escape_intensity * 2,
        remove_large: 1.0 + escape_intensity * 3,
        remove_all: 0.5 + escape_intensity * 5,  # Risky, use carefully
        swap_calls: 1.0 + escape_intensity * 1.5,
        shuffle_vehicle: 1.0 + escape_intensity * 2.5,
        one_reinsert: 1.0 + escape_intensity * 1,
        greedy_reinsert: 1.0 + escape_intensity * 2,
    }

    # Number of escape attempts scales with intensity
    num_escapes = int(2 + escape_intensity * 5)

    successful_escapes = 0

    for _ in range(num_escapes):
        method = random.choices(escape_methods, weights=[weights[m] for m in escape_methods])[0]
        proposed_solution = method(escape_solution, problem)
        feasible, _ = feasibility_check(proposed_solution, problem)

        if feasible:
            escape_solution = proposed_solution
            successful_escapes += 1

    # If no successful escape found, apply a guaranteed aggressive shake
    if successful_escapes == 0:
        fallback_solution = remove_large(escape_solution, problem)
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
    exp_scores = np.exp(scores - np.max(scores))
    softmax = exp_scores / np.sum(exp_scores)
    for i in range(len(operator_scores)):
        operator_scores[i] = softmax[i]

def accept_solution(new_solution, new_cost, incumbent_cost, problem, T):
    """ Accept function for Simulated Annealing """

    delta_E = new_cost - incumbent_cost

    feasible, _ = feasibility_check(new_solution, problem)

    if feasible:
        if delta_E < 0:
            return True 
        elif random.random() < math.exp(-delta_E / T):
            return True  
    return False

def accept(new_cost, max_iterations, iteration, best_cost):

    delta = 0.2 * ((max_iterations - iteration) / max_iterations) * best_cost

    if new_cost < best_cost:
        return True 
    elif new_cost < best_cost + delta:
        return True


def k_regret(solution, problem):
   
   #pick a random car
   #for each call in the car or pick a random number of calls from the car
    #store best solution and next best solution
    #if best placement between two calls is equal, choose the next best position for the call with the best next best solution
    #if the best solution is better than the current solution, apply the change

    #maybe i can use this as a dictinoary to store the best and next best solution, in other methods like greedy reinsert.
    #best_solution = {}
    return



    total_improvements = sum(operator_improvements)
    if total_improvements > 0:
        for i in range(num_operators):
            operator_probabilities[i] = operator_improvements[i] / total_improvements
    else:
        operator_probabilities[:] = [1/num_operators] * num_operators

    # def normalize_scores(operator_scores):
    # """Normalize operator scores to prevent extreme values"""
    # min_score = 0.1  # Minimum allowed score
    # max_score = 10.0  # Maximum allowed score

    # for idx in range(len(operator_scores)):
    #     operator_scores[idx] = max(min_score, min(max_score, operator_scores[idx]))