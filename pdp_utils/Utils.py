import numpy as np
import random

from collections import namedtuple


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
    sol = [0] * num_vehicles
    for i in range(problem['n_calls']):
        sol.append(i+1)
        sol.append(i+1)
    return sol

def generate_random(problem):
    """
    Genererer en tilfeldig løsning for pickup and delivery-problemet.
    """

    solution = initial_solution(problem)
    dummy = [0] 
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
                    assigned_calls.add(call + 1)
                    remaining_capacity[vehicle] -= call_size
        
        if vehicle_calls:  
            np.random.shuffle(vehicle_calls) 
            solution.extend(vehicle_calls)

    for call in calls:
        if call not in assigned_calls:
            dummy.append(call + 1 )
            dummy.append(call + 1)

    if dummy:
        solution.extend(dummy)
        
    return solution

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

#def relocate(solution, problem):
    """ Move a random call from one vehicle to another and return the new solution """
    new_solution = solution.copy()
    
    num_calls = len(solution)
    num_vehicles = problem['n_vehicles']
    
    call_to_move = random.randint(0, num_calls - 1)
    
    valid_vehicles = np.where(problem['VesselCargo'][:, call_to_move] == 1)[0]
    
    if len(valid_vehicles) > 0:
        new_vehicle = random.choice(valid_vehicles) + 1  
        new_solution[call_to_move] = new_vehicle  

    return new_solution


#def local_search(problem, initial_solution, max_iterations=100):
    
    best_solution = initial_solution
    best_cost = cost_function(best_solution, problem)

    for _ in range(max_iterations):
        new_solution = relocate(best_solution, problem)
        
        feasible, _ = feasibility_check(new_solution, problem)
        if not feasible:
            continue 
        
        new_cost = cost_function(new_solution, problem)
        
        if new_cost < best_cost:
            best_solution = new_solution
            best_cost = new_cost
    
    return best_solution, best_cost



