import numpy as np
import random
import math
import copy


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
    # Create a copy of the solution to modify
    new_solution = solution.copy()
    
    # Split the solution into vehicle routes
    vehicles = split_into_vehicles(new_solution)
    
    # Last vehicle is the dummy vehicle
    dummy_vehicle = vehicles[-1]
    actual_vehicles = vehicles[:-1]
    
    # If dummy vehicle has no calls, return the original solution
    if not dummy_vehicle:
        return solution
    
    # Get all call pairs in dummy vehicle
    call_pairs = {}
    for call in dummy_vehicle:
        call_id = abs(call)
        if call_id in call_pairs:
            call_pairs[call_id].append(call)
        else:
            call_pairs[call_id] = [call]
    
    # Select a random call pair
    if not call_pairs:
        return solution
    
    random_call_id = random.choice(list(call_pairs.keys()))
    call_pair = call_pairs[random_call_id]
    
    # Remove the call pair from dummy vehicle
    for call in call_pair:
        dummy_vehicle.remove(call)
    
    # Try to insert the calls into each vehicle and position
    best_cost = float('inf')
    best_vehicles = None
    
    for v_idx, vehicle in enumerate(actual_vehicles):
        # Check if vehicle can handle this call
        if not np.all(problem['VesselCargo'][v_idx, [c-1 for c in call_pair]]):
            continue
        
        # Try all possible insertion positions for the pickup
        for pickup_pos in range(len(vehicle) + 1):
            pickup_vehicle = vehicle.copy()
            pickup_vehicle.insert(pickup_pos, call_pair[0])
            
            # If pickup insertion is feasible, try all delivery positions
            if check_vehicle_feasibility(pickup_vehicle, v_idx, problem):
                for delivery_pos in range(pickup_pos + 1, len(pickup_vehicle) + 1):
                    test_vehicle = pickup_vehicle.copy()
                    test_vehicle.insert(delivery_pos, call_pair[0])
                    
                    # Check if delivery insertion is feasible
                    if check_vehicle_feasibility(test_vehicle, v_idx, problem):
                        # Calculate new cost
                        test_vehicles = actual_vehicles.copy()
                        test_vehicles[v_idx] = test_vehicle
                        test_vehicles.append(dummy_vehicle)
                        test_solution = combine_vehicles(test_vehicles)
                        
                        test_cost = cost_function(test_solution, problem)
                        
                        # If this is the best solution so far, save it
                        if test_cost < best_cost:
                            best_cost = test_cost
                            best_vehicles = test_vehicles.copy()
                            # Stop at first better solution
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

def shuffle_vehicle(solution):

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

def simulated_annealing(problem):
    """ Simulated Annealing for Pickup and Delivery Problem """

    current_solution = initial_solution(problem)    

    incumbent = current_solution.copy()
    incumbent_cost = cost_function(incumbent, problem)

    best_solution = current_solution.copy()
    best_cost = cost_function(best_solution, problem)


    operator_1 = shuffle_vehicle
    operator_2 = swap_calls
    operator_3 = dummy_reinsert

    Tf = 0.1  # Final temperature
    probability_01 = 1/3
    probability_02 = 1/3
    probability_03 = 1/3
    delta_Es = []

    for w in range(1, 100):

        #selected_operator = random.choices([operator_1, operator_2], [probability_01, probability_02], k=1)[0]
        selected_operator = random.choices([operator_1, operator_2, operator_3], [probability_01, probability_02, probability_03], k=1)[0]

        # Apply the selected operator
        if selected_operator == operator_1:
            new_solution = operator_1(incumbent)
        elif selected_operator == operator_2:
            new_solution = operator_2(incumbent, problem)
        else:
            new_solution = operator_3(incumbent, problem)

        
        delta_E = cost_function(new_solution, problem) - incumbent_cost 
        feasible, _ = feasibility_check(new_solution, problem)
        
        # Feasibility check before accepting the solution
        if feasible and delta_E < 0:
            incumbent = new_solution
            incumbent_cost = cost_function(incumbent, problem)
            if incumbent_cost < best_cost: 
                best_solution = incumbent
                best_cost = incumbent_cost
        elif feasible:
            if random.random() < 0.8:  # Temperature-based acceptance
                incumbent = new_solution
                incumbent_cost = cost_function(incumbent, problem)

            delta_Es.append(delta_E)

    # Calculate DeltaAvg and initial temperature (T0)
    DeltaAvg = np.mean(delta_Es)
    T0 = -DeltaAvg / math.log(0.8)
    alpha = (Tf / T0) ** (1/9900)
    
    T = T0  # Initial temperature

    # Simulated Annealing Loop
    for iteration in range(1, 9900):

        selected_operator = random.choices([operator_1, operator_2, operator_3], [probability_01, probability_02, probability_03], k=1)[0]
        #selected_operator = random.choices([operator_1, operator_2], [probability_01, probability_02], k=1)[0]

        # Apply the selected operator
        if selected_operator == operator_1:
            new_solution = operator_1(incumbent)
        elif selected_operator == operator_2:
            new_solution = operator_2(incumbent, problem)
        else:
            new_solution = operator_3(incumbent, problem)
        
        delta_E = cost_function(new_solution, problem) - incumbent_cost
        feasible, _ = feasibility_check(new_solution, problem)

        # Feasibility check before accepting the solution
        if feasible and delta_E < 0:
            incumbent = new_solution
            incumbent_cost = cost_function(incumbent, problem)
            if incumbent_cost < best_cost:
                best_solution = incumbent
                best_cost = incumbent_cost
        elif feasible and random.random() < math.exp(-delta_E / T):  # Temperature-based acceptance
            incumbent = new_solution
            incumbent_cost = cost_function(incumbent, problem)

        # Update the temperature
        T = alpha * T

    return best_solution

def simulated_annealing_weight(problem):
    """ Simulated Annealing for Pickup and Delivery Problem with Adaptive Operator Selection """

    current_solution = initial_solution(problem)    

    incumbent = current_solution.copy()
    incumbent_cost = cost_function(incumbent, problem)

    best_solution = current_solution.copy()
    best_cost = cost_function(best_solution, problem)

    operators = [shuffle_vehicle, swap_calls, dummy_reinsert]
    operator_success = [1, 1, 1]
    operator_usage = [1, 1, 1] 

    Tf = 0.1
    delta_Es = []

    for w in range(1, 100):

        probabilities = [s / sum(operator_success) for s in operator_success] 
        selected_index = random.choices(range(len(operators)), probabilities, k=1)[0]
        selected_operator = operators[selected_index]

        new_solution = selected_operator(incumbent) if selected_operator == shuffle_vehicle else selected_operator(incumbent, problem)

        delta_E = cost_function(new_solution, problem) - incumbent_cost 
        feasible, _ = feasibility_check(new_solution, problem)

        if feasible and delta_E < 0:
            incumbent = new_solution
            incumbent_cost = cost_function(incumbent, problem)
            if incumbent_cost < best_cost:
                best_solution = incumbent
                best_cost = incumbent_cost
            operator_success[selected_index] += 1 
        elif feasible:
            if random.random() < 0.8:
                incumbent = new_solution
                incumbent_cost = cost_function(incumbent, problem)
                operator_success[selected_index] += 1  

        operator_usage[selected_index] += 1  
        delta_Es.append(delta_E)

    DeltaAvg = np.mean(delta_Es)
    T0 = -DeltaAvg / math.log(0.8)
    alpha = (Tf / T0) ** (1/9900)
    
    T = T0  

    for iteration in range(1, 9900):

        probabilities = [s / sum(operator_success) for s in operator_success] 
        selected_index = random.choices(range(len(operators)), probabilities, k=1)[0]
        selected_operator = operators[selected_index]

        new_solution = selected_operator(incumbent) if selected_operator == shuffle_vehicle else selected_operator(incumbent, problem)

        delta_E = cost_function(new_solution, problem) - incumbent_cost
        feasible, _ = feasibility_check(new_solution, problem)

        #print(f"probability: {probabilities}")
        #print(f"selected operator: {selected_operator}")

        if feasible and delta_E < 0:
            incumbent = new_solution
            incumbent_cost = cost_function(incumbent, problem)
            if incumbent_cost < best_cost:
                best_solution = incumbent
                best_cost = incumbent_cost
            operator_success[selected_index] += 1  
        elif feasible and random.random() < math.exp(-delta_E / T):
            incumbent = new_solution
            incumbent_cost = cost_function(incumbent, problem)
            operator_success[selected_index] += 1 

        operator_usage[selected_index] += 1
        T = alpha * T  

    return best_solution

def generate_random(problem):
    """
    Genererer en tilfeldig løsning for pickup and delivery-problemet.
    2. Generate Random Solution (generate_random)
    The function generates a random solution but does not ensure that the solution 
    respects the vehicle capacities and constraints. Specifically, the vehicle's remaining 
    capacity is updated after assigning a call, but no checks seem to be in place to ensure
    that the vehicle doesn’t exceed its capacity.

    Improvement suggestion:

    You could include additional checks within this function to ensure that after assigning 
    each call, the total cargo weight of the vehicle doesn’t exceed its capacity. This could
    prevent infeasible solutions from being generated in the first place.
    You might also want to consider removing or adjusting the logic for randomly shuffling 
    calls and vehicles. If the random assignment of calls leads to infeasibility, you might
    need to generate a new solution instead of continuing.   

    """

    solution = initial_solution(problem)
    new_solution = []
    dummy = [] 
    assigned_calls = set()  

    num_calls = problem['n_calls']
    Cargo = problem['Cargo'][:, 2]
    VesselCargo = problem['VesselCargo']
    VesselCapacity = problem['VesselCapacity']

    remaining_capacity = VesselCapacity.copy()

    vehicles = list(range(len(VesselCapacity)))

    calls = list(range(num_calls))
    np.random.shuffle(calls) 

    for vehicle in vehicles:
        vehicle_calls = []

        for call in calls:
            if call in assigned_calls:
                continue
      
            call_size = Cargo[call]

            if VesselCargo[vehicle, call] == 1 and remaining_capacity[vehicle] >= call_size:
                    vehicle_calls.append(call + 1) 
                    vehicle_calls.append(call + 1)  
                    assigned_calls.add(call)
                    remaining_capacity[vehicle] -= call_size
        
        if vehicle_calls:  
            np.random.shuffle(vehicle_calls) 
            new_solution.extend(vehicle_calls)
            new_solution.append(0)

    for call in calls:
        if call not in assigned_calls:
            dummy.append(call + 1)
            dummy.append(call + 1)
            assigned_calls.add(call)

    if dummy:
        new_solution.extend(dummy)
        
    return new_solution

def find_best_random(problem):

    best_solution = generate_random(problem)
    best_cost = cost_function(best_solution, problem)

    for _ in range(1000):
        current = generate_random(problem)

        feasible, _ = feasibility_check(current, problem)
        if feasible:
            current_cost = cost_function(current, problem)
            if current_cost < best_cost:
                best_solution = current
                best_cost = current_cost


    return best_solution

def local_search(problem):
    best_solution = initial_solution(problem) #starter med en løsning
    best_cost = cost_function(best_solution, problem) #lagrer best cost 
    num_iterations = 10000

    rejected_solutions = 0

    for i in range(num_iterations):
        new_solution = one_reinsert(best_solution, problem)
        feasible, _ = feasibility_check(new_solution, problem)
        
        if not feasible: #ev putte in if feasible
            rejected_solutions += 1
            continue 
        
        new_cost = cost_function(new_solution, problem)
        
        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost

    print(f"Total rejected solutions: {rejected_solutions}")
    return best_solution

def one_reinsert(solution, problem):
    """
    Gjør en liten endring i løsningen ved å flytte en tilfeldig valgt forespørsel fra ett kjøretøy til et annet.
    """
    new_solution = copy.deepcopy(solution)

    vehicles = []
    current_vehicle = []
    for call in new_solution:
        if call == 0:
            vehicles.append(current_vehicle)
            current_vehicle = []
        else:
            current_vehicle.append(call)     
    vehicles.append(current_vehicle)

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

    new_solution = []
    for vehicle in vehicles:
        new_solution.extend(vehicle)
        new_solution.append(0)
        
    # Fjern siste 0 hvis det er flere enn nødvendig
    if new_solution[-1] == 0:
        new_solution.pop()

    assert new_solution.count(chosen_call) == 2, f"Feil: {chosen_call} finnes {new_solution.count(chosen_call)} ganger!"
    return new_solution

def simulated_annealing_1(problem):
    """ Simulated Annealing for Pickup and Delivery Problem """

    current_solution = initial_solution(problem)

    incumbent = current_solution.copy()
    incumbent_cost = cost_function(incumbent, problem)

    best_solution = current_solution.copy()
    best_cost = cost_function(best_solution, problem)

    Tf = 0.1  # Final temperature
    delta_Es = []

    for w in range(1, 100):
        #new_solution = shuffle_vehicle(incumbent)
        #new_solution = swap_calls(incumbent,problem)
        new_solution = dummy_reinsert(incumbent, problem)
        #new_solution = greedy_reinsert(incumbent, problem)
        #new_solution = optimal_vehicle_route(incumbent, problem)
        #new_solution = one_reinsert(incumbent, problem)
        delta_E = cost_function(new_solution, problem) - incumbent_cost 
        feasible, _ = feasibility_check(new_solution, problem)
        
        # Feasibility check before accepting the solution
        if feasible: # Husk at denne returnerer to ting, du må ta feasibility check før if-setningen og hente ut feasibility
            if delta_E < 0:
                incumbent = new_solution
                incumbent_cost = cost_function(incumbent, problem)
                if incumbent_cost < best_cost: 
                    best_solution = incumbent
                    best_cost = incumbent_cost
        elif feasible:
            if random.random() < 0.8:  # Temperature-based acceptance
                incumbent = new_solution
                incumbent_cost = cost_function(incumbent, problem)

            delta_Es.append(delta_E)

    # Calculate DeltaAvg and initial temperature (T0)
    DeltaAvg = np.mean(delta_Es)
    T0 = -DeltaAvg / math.log(0.8)
    alpha = (Tf / T0) ** (1/9900)
    
    T = T0  # Initial temperature

    # Simulated Annealing Loop
    for iteration in range(1, 9900):
        #new_solution = shuffle_vehicle(incumbent)
        #new_solution = swap_calls(incumbent, problem)
        new_solution = dummy_reinsert(incumbent, problem)
        #new_solution = greedy_reinsert(incumbent, problem)
        #new_solution = optimal_vehicle_route(incumbent, problem)
        #new_solution = one_reinsert(incumbent, problem)
        delta_E = cost_function(new_solution, problem) - incumbent_cost
        feasible, _ = feasibility_check(new_solution, problem)

        # Feasibility check before accepting the solution
        if feasible:
            if delta_E < 0:
                incumbent = new_solution
                incumbent_cost = cost_function(incumbent, problem)
                if incumbent_cost < best_cost:
                    best_solution = incumbent
                    best_cost = incumbent_cost
        elif feasible and random.random() < math.exp(-delta_E / T):  # Temperature-based acceptance
            incumbent = new_solution
            incumbent_cost = cost_function(incumbent, problem)

        # Update the temperature
        T = alpha * T

    return best_solution
