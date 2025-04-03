import numpy as np
import random
import copy
from pdp_utils import feasibility_check, split_into_vehicles, combine_vehicles, cost_function

def calculate_delta(vehicle, call, problem):
    """
    Calculate cost delta when removing a call from a vehicle.
    
    Args:
        vehicle: Vehicle route 
        call: Call to remove
        problem: Problem instance data
        
    Returns:
        float: Cost difference after removal (negative means cost reduction)
    """
    original_cost = calculate_vehicle_cost(vehicle, problem)
    modified_vehicle = [c for c in vehicle if c != call]
    new_cost = calculate_vehicle_cost(modified_vehicle, problem)
    return new_cost - original_cost

def greedy_reinsert(solution, problem):
    """
    Makes a small change to the solution by moving a randomly chosen call within a vehicle,
    but inserts the call in a greedy way (minimizing the additional cost).
    """
    new_solution = solution[:]
    vehicles = split_into_vehicles(new_solution)

# fjerning
    non_empty_vehicles = [i for i, v in enumerate(vehicles) if len(set(v)) > 1]
    vehicle_idx = random.choice(non_empty_vehicles) # hvordan kan jeg fjerne smartere?
    from_vehicle = vehicles[vehicle_idx]
    
    chosen_call = random.choice([c for c in from_vehicle if c != 0])
    if from_vehicle.count(chosen_call) != 2:
        raise ValueError(f"Feil: chosen_call {chosen_call} finnes ikke to ganger i from_vehicle {from_vehicle}")
  
    # Remove the chosen call from the vehicle
    from_vehicle[:] = [c for c in from_vehicle if c != chosen_call]

# tilbakelegging innad i vehicle

    original_cost = calculate_vehicle_cost(from_vehicle, vehicle_idx, problem)
    best_position = None
    best_cost = float('inf')
    
    # Greedy insertion: Try inserting at best spot
    for i in range(len(from_vehicle) + 1):
        test_vehicle = from_vehicle[:i] + [chosen_call] + from_vehicle[i:] + [chosen_call]
        vehicle_cost = calculate_vehicle_cost(test_vehicle, vehicle_idx, problem)
        
        if vehicle_cost < best_cost:
            best_cost = vehicle_cost
            best_position = i
    
    # Apply best found insertion
    if best_position is not None and best_cost < original_cost:
        vehicles[vehicle_idx] = from_vehicle[:best_position] + [chosen_call] + from_vehicle[best_position:] + [chosen_call]
        new_solution = combine_vehicles(vehicles)

    return new_solution

def calculate_vehicle_cost(vehicle, vehicle_idx, problem):
    """
    Calculate the cost for a single vehicle route without recalculating the entire solution.
    
    Args:
        vehicle: List of calls for this vehicle (without the trailing 0)
        vehicle_idx: Index of the vehicle in the problem
        problem: Problem data
        
    Returns:
        float: The cost associated with this vehicle route
    """
    if not vehicle:
        return 0.0
    
    Cargo = problem['Cargo']
    TravelCost = problem['TravelCost']
    FirstTravelCost = problem['FirstTravelCost']
    PortCost = problem['PortCost']
    
    sorted_plan = np.sort(vehicle, kind='mergesort')
    I = np.argsort(vehicle, kind='mergesort')
    Indx = np.argsort(I, kind='mergesort')

    PortIndex = Cargo[sorted_plan, 1].astype(int)
    PortIndex[::2] = Cargo[sorted_plan[::2], 0]
    PortIndex = PortIndex[Indx] - 1
    
    # Calculate route travel cost
    RouteTravelCost = 0
    # Calculate travel costs between consecutive ports
    if len(PortIndex) > 1:
        Diag = TravelCost[(vehicle_idx - 1), PortIndex[:-1], PortIndex[1:]]
        RouteTravelCost += np.sum(Diag.flatten())
        
    # Add first visit cost
    FirstVisitCost = FirstTravelCost[vehicle_idx - 1, int(Cargo[vehicle[0], 0] - 1)]
    RouteTravelCost += FirstVisitCost
        
        # Add port costs      
    CostInPorts = np.sum(PortCost[vehicle_idx - 1, vehicle]) / 2

    TotalVehicleCost = RouteTravelCost + CostInPorts
    return TotalVehicleCost


def find_best_insertion(to_vehicle, chosen_call, vehicles, to_vehicle_idx, problem):
    """
    Finn en god plassering for `chosen_call` i `to_vehicle` basert på inkrementell kostnadsberegning.
    Reduserer antall kall til cost_function() ved å sjekke færre muligheter.
    """
    best_position = None
    best_cost = float('inf')
    
    # Hent eksisterende kostnad for kjøretøyet
    base_cost = cost_function(vehicles, problem)

    for i in range(len(to_vehicle) + 1):
        # Sett inn kallet midlertidig og beregn kun differansen i kostnad
        temp_vehicle = to_vehicle[:]
        temp_vehicle.insert(i, chosen_call)
        temp_vehicle.insert(i + 1, chosen_call)

        # Oppdater hele løsningen
        temp_solution = vehicles[:]
        temp_solution[to_vehicle_idx] = temp_vehicle

        # Beregn differanse i kostnad i stedet for hele løsningen på nytt
        new_cost = cost_function(temp_solution, problem)
        delta_cost = new_cost - base_cost

        if delta_cost < best_cost:
            best_cost = delta_cost
            best_position = i

    return best_position

def route_optimizer(solution, problem):
    """
    Reorganizes all calls within a randomly selected vehicle to find a more optimal sequence.
    This operator works on one vehicle at a time and completely resequences its calls.
    
    Strategy:
    1. Select a random vehicle with multiple calls
    2. Extract all unique calls from that vehicle
    3. Generate all possible permutations (limited to reasonable size) or use greedy insertion
    4. Select the best permutation based on cost
    """
    new_solution = copy.deepcopy(solution)
    
    # Split solution into vehicles
    vehicles = split_into_vehicles(new_solution)
    
    # Find vehicles with at least 2 unique calls
    candidate_vehicles = []
    for i, vehicle in enumerate(vehicles):
        # Count unique calls in vehicle
        unique_calls = set([c for c in vehicle if c != 0])
        if len(unique_calls) >= 2:  # Need at least 2 unique calls to reorder
            candidate_vehicles.append(i)
    
    if not candidate_vehicles:
        return new_solution
    
    # Select a random vehicle to optimize
    vehicle_idx = random.choice(candidate_vehicles)
    vehicle = vehicles[vehicle_idx]
    
    # Extract unique calls from the vehicle
    unique_calls = list(set([c for c in vehicle if c != 0]))
    
    # Cache for storing cost calculations to avoid redundancy
    cost_cache = {}
    
    # Cache for storing feasibility results
    feasibility_cache = {}
    
    # If too many unique calls, use a greedy approach instead of trying all permutations
    if len(unique_calls) > 5:  # Permutations become too many with more than 5 unique calls
        # Use greedy insertion to build a new route
        best_vehicle = []
        remaining_calls = unique_calls.copy()
        
        # Start with a random call
        initial_call = random.choice(remaining_calls)
        best_vehicle.append(initial_call)
        best_vehicle.append(initial_call)
        remaining_calls.remove(initial_call)
        
        # Greedy insertion of remaining calls
        while remaining_calls:
            best_call = None
            best_positions = (0, 0)
            best_insertion_cost = float('inf')
            
            # Try each remaining call in all possible positions
            for call in remaining_calls:
                for i in range(len(best_vehicle) + 1):
                    for j in range(i, len(best_vehicle) + 1):
                        # Create test route with this call inserted
                        test_vehicle = best_vehicle.copy()
                        test_vehicle.insert(i, call)
                        test_vehicle.insert(j + 1, call)  # +1 because we've already inserted at i
                        
                        # Use cached values if available
                        test_key = tuple(test_vehicle)
                        if test_key in cost_cache:
                            current_cost = cost_cache[test_key]
                            is_feasible = feasibility_cache[test_key]
                        else:
                            # Create full solution for evaluation
                            temp_solution = []
                            for idx, v in enumerate(vehicles):
                                if idx == vehicle_idx:
                                    temp_solution.extend(test_vehicle)
                                else:
                                    temp_solution.extend(v)
                                temp_solution.append(0)
                            
                            # Remove the last 0 if necessary
                            if temp_solution[-1] == 0:
                                temp_solution.pop()
                            
                            # Check feasibility and cost
                            is_feasible, _ = feasibility_check(temp_solution, problem)
                            feasibility_cache[test_key] = is_feasible
                            
                            if is_feasible:
                                current_cost = cost_function(temp_solution, problem)
                                cost_cache[test_key] = current_cost
                            else:
                                current_cost = float('inf')
                        
                        # Update best insertion if better
                        if is_feasible and current_cost < best_insertion_cost:
                            best_insertion_cost = current_cost
                            best_call = call
                            best_positions = (i, j + 1)
            
            # If found a feasible insertion, add it to the route
            if best_call:
                best_vehicle.insert(best_positions[0], best_call)
                best_vehicle.insert(best_positions[1], best_call)
                remaining_calls.remove(best_call)
            else:
                # If no feasible insertion found, use random insertion as fallback
                call = random.choice(remaining_calls)
                i = random.randint(0, len(best_vehicle))
                j = random.randint(i, len(best_vehicle))
                best_vehicle.insert(i, call)
                best_vehicle.insert(j + 1, call)
                remaining_calls.remove(call)
        
        # Update the vehicle with the optimized route
        vehicles[vehicle_idx] = best_vehicle
    else:
        # For small number of unique calls, try all permutations of call ordering
        best_vehicle = vehicle  # Default to current route
        best_cost = float('inf')
        
        # Try different permutations of unique calls
        import itertools
        for perm in itertools.permutations(unique_calls):
            # Create a test route with this permutation
            test_vehicle = []
            for call in perm:
                test_vehicle.append(call)
                test_vehicle.append(call)
            
            # Use cached values if available
            test_key = tuple(test_vehicle)
            if test_key in cost_cache:
                current_cost = cost_cache[test_key]
                is_feasible = feasibility_cache[test_key]
            else:
                # Create full solution for evaluation
                temp_solution = []
                for idx, v in enumerate(vehicles):
                    if idx == vehicle_idx:
                        temp_solution.extend(test_vehicle)
                    else:
                        temp_solution.extend(v)
                    temp_solution.append(0)
                
                # Remove the last 0 if necessary
                if temp_solution[-1] == 0:
                    temp_solution.pop()
                
                # Check feasibility and cost
                is_feasible, _ = feasibility_check(temp_solution, problem)
                feasibility_cache[test_key] = is_feasible
                
                if is_feasible:
                    current_cost = cost_function(temp_solution, problem)
                    cost_cache[test_key] = current_cost
                else:
                    current_cost = float('inf')
            
            # Update best route if better
            if is_feasible and current_cost < best_cost:
                best_cost = current_cost
                best_vehicle = test_vehicle
        
        # Update the vehicle with the optimized route
        vehicles[vehicle_idx] = best_vehicle
    
    # Reconstruct the solution
    new_solution = []
    for vehicle in vehicles:
        new_solution.extend(vehicle)
        new_solution.append(0)
    
    # Remove the last 0 if necessary
    if new_solution[-1] == 0:
        new_solution.pop()
    
    return new_solution


#Jeg vil ha en operator som omorganiserer alle forespørslene inni et kjøretøy til optimal rekkefølge. Denne operatoren skal kun brukes på et kjøretøy av gangen.
# Mulig det er raskest å fjerne alle og deretter plassere dem på nytt i optimal rekkefølge. Hvilke datapunkter kan jeg ta inn for å få til dette.
#Hvilke operatorer kan jeg bruke for å gi meg ny informasjon om hvilke forespørsler som skal flyttes? hvordan kan denne informasjonen lagres. 
# Tar denne operatoren veldig lang tid om data inputet blir veldig stort? ønsker ikke å gjøre dette mange ganger så midlertidig info må lagres. 
#Hvordan kan jeg lagre (ny) informasjonen på en effektiv måte? havner de i den mest optimale rruten eller bare billigst. Hva er det som måles, kan jeg ta begge eller bare en? 

#jeg vil ha en operator som, fjerner en tilfeldig call og plasserer den på en optimal plass og sjekker om den er bedre enn før.
#Hvordan kan jeg lagre informasjonen om hvor den var før og hvor den er nå? 
#husk å regne ut avstand mellom de du fjerner og finn ut diferansen til den originale reisen. : eksempel [1,2,3,4] fjern 2 regn ut [1-2] og [2-3] delta [1-3]

#Jeg vil ha en operator som fjerner den costliest callen og plasserer den på en optimal plass og sjekker om den er bedre enn før.
#Kan ikke alltid ta den dyreste siden det ofte blir den samme

#Jeg vil ha en operator som sjekker hvilke oppdrag som bare er helt løk og burde legges til i dummy??? 

# er det smart å gå igjennom bilene sjekke innad eller se på hele bilde, når blir koden for stor og kjører alt for sakte?

#fjerne den dyreste
#kan jeg finne ut av travel distance og korte ned mellom kjøretøy?
#finne cost of not transporting
#kartlegge max og passe på at de blir feasible?
#pickup time og delivery time window for å få tid til flest?
#kan jeg lagre hvilke noder ulike kjøretøy og calls starter på?
#hvilke calls som kun kan kjøres av et kjøretøy
#diversification for removing
#intensification for intensification

#ha en med randomremoval


# hvordan skal jeg legge inn ? 
# shuffle calls you have removed then insert in best position, try to find compatible vehicle and place in best position
# 1. remove call
# 2. shuffle calls
# 3. insert in best position

#Do not calculate everything everytime
#some of the cost that is not touched
#some feasibility checks that are not touched
#only add delta cost and

#check if something is on the same node as the pickup or delivery

#b + c - a delta cost 

#do not check where the destination is very far away


#hvis du fjerner to like calls er det lettere å legge dem tilbake


#fern alle calls og plasser dem på best plass??, deretter gå gjennom dummy og gjør det sammme??
#vurder best position og nest best position, hvis det er en stor forskjell på disse to, så er det en god ide å flytte den
#slide 16 - kvalitet greedy + regret-k. 

#hva er det som gjør at en løsning er bedre enn en annen?:
    # intensification
    # diversification
    # Kjøretid
    # Kostnad
    # Feasibility
    # Travel time

#ha alltid vekten normalisert 0 < vekt < 1



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
Use code with caution.
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
Use code with caution.
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
Use code with caution.
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
Returns both a feasibility flag and a message with the reason for infeasibility.

Args:
    vehicle_plan: List of calls for this vehicle (without the trailing 0)
    vehicle_idx: Index of the vehicle in the problem
    problem: Problem data
    
Returns:
    tuple: (feasibility (bool), message (str))
"""
if not vehicle_plan:
    return True, "Route is empty, assumed feasible."

Cargo = problem['Cargo']
TravelTime = problem['TravelTime']
FirstTravelTime = problem['FirstTravelTime']
VesselCapacity = problem['VesselCapacity']
LoadingTime = problem['LoadingTime']
UnloadingTime = problem['UnloadingTime']
VesselCargo = problem['VesselCargo']

print("vehicle plan" + str(vehicle_plan))

# Convert to 0-indexed call numbers
currentVPlan = [c - 1 for c in vehicle_plan]

NoDoubleCallOnVehicle = len(currentVPlan)

if NoDoubleCallOnVehicle > 0:
    print("current VPlan" + str(currentVPlan))
    # Ensure all call indices in currentVPlan are valid
    if not all(0 <= c < VesselCargo.shape[1] for c in currentVPlan):
        return False, "Invalid call index in vehicle route."
    
    # Check if the vehicle can transport all calls
    print("VesselCargo" + str(VesselCargo))
    print("currentidx " + str(vehicle_idx - 1))
    if not np.all(VesselCargo[vehicle_idx - 1, currentVPlan]):
        return False, "Incompatible vessel and cargo for the route."
    
    # Check capacity constraints
    LoadSize = np.zeros(NoDoubleCallOnVehicle)
    currentTime = 0
    sortRout = np.sort(currentVPlan, kind='mergesort')
    I = np.argsort(currentVPlan, kind='mergesort')
    Indx = np.argsort(I, kind='mergesort')
    
    LoadSize -= Cargo[sortRout, 2]  # Negative for delivery (unloading)
    LoadSize[::2] = Cargo[sortRout[::2], 2]  # Positive for pickup (loading)
    LoadSize = LoadSize[Indx]
    
    # Check capacity constraints
    if np.any(VesselCapacity[vehicle_idx - 1] - np.cumsum(LoadSize) < 0):
        return False, "Capacity exceeded during the route."
    
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
    
    LU_Time = UnloadingTime[vehicle_idx - 1, sortRout]  # Unloading time
    LU_Time[::2] = LoadingTime[vehicle_idx - 1, sortRout[::2]]  # Loading time
    LU_Time = LU_Time[Indx]
    
    if len(PortIndex) > 1:
        Diag = TravelTime[vehicle_idx - 1, PortIndex[:-1], PortIndex[1:]]
        RouteTravelTime = Diag.flatten()
    else:
        RouteTravelTime = []
    
    FirstVisitTime = FirstTravelTime[vehicle_idx - 1, int(Cargo[currentVPlan[0], 0] - 1)]
    RouteTravelTime = np.hstack((FirstVisitTime, RouteTravelTime))
    
    ArriveTime = np.zeros(NoDoubleCallOnVehicle)
    for j in range(NoDoubleCallOnVehicle):
        ArriveTime[j] = np.max((currentTime + RouteTravelTime[j], Timewindows[0, j]))
        if ArriveTime[j] > Timewindows[1, j]:
            return False, f"Time window exceeded at call {j + 1}."
        currentTime = ArriveTime[j] + LU_Time[j]

return True, "Route is feasible."
Use code with caution.
def check_vehicle_feasibility1(vehicle_plan, vehicle_idx, problem):
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

Cargo = problem['Cargo'][:2]
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

    # Ensure all call indices in currentVPlan are valid
    if not all(0 <= c < VesselCargo.shape[1] for c in currentVPlan):
        return False  # Handle invalid index scenario

    # Proceed with the check if all call indices are valid
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
Use code with caution.
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
Use code with caution.
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
Use code with caution.
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
Use code with caution.
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
Use code with caution.
def evaluate_best_placement_for_all_calls(solution, problem):
"""
Evaluates the best placement (pickup and delivery) for each call in the solution.
Stores the best placements in a dictionary with call_id as key and (pickup_pos, delivery_pos, cost) as values.
"""
best_placements = {}
seen_calls = {}

pickup_position = None
delivery_position = None

# Iterate through the entire solution, treating it as a single list of calls
for position, call_id in enumerate(solution):
        if call_id == 0:
            continue

        #hvis det er første instans av call_id, sett pickup_position til posisjonen
        #lagre call_id i seen_calls
        
        if call_id not in seen_calls:
            pickup_position = position
            print(seen_calls)
            seen_calls[call_id] = position
        #hvis det er andre instans av call_id, sett delivery_position til posisjonen
        else:
            delivery_position = position


        # Calculate cost for this particular solution configuration
        cost = cost_function(solution, problem)

        # Store the best placement if it's the first time we encounter this call
        if call_id not in best_placements:
            best_placements[call_id] = (pickup_position, delivery_position, cost)

        # If we find a better (lower cost) placement, update the dictionary
        elif cost < best_placements[call_id][2]:
            best_placements[call_id] = (pickup_position, delivery_position, cost)

return best_placements
Use code with caution.
def apply_smart_operator(solution, best_placements):
"""
Uses the stored best placements to improve the solution by moving calls to their optimal positions.
"""
new_solution = copy.deepcopy(solution)

# Iterate through each call in best placements and move it to the optimal position
#print(best_placements)
#print(new_solution)
#print(best_pickup)
#print(best_delivery)
for call_id, (best_pickup, best_delivery, _) in best_placements.items():
    #if call id does not have a best placement, continue
    if call_id not in new_solution:
        continue
    # Move the call to its best positions (pickup first, delivery second)
    new_solution = place_call_in_best_position(new_solution, call_id, best_pickup, best_delivery)

return new_solution
Use code with caution.
def best_placement_operator(solution, problem):
"""
Operator for improving the solution by placing calls in their best positions.
Uses evaluate_best_placement_for_all_calls and apply_smart_operator.
"""
# Step 1: Evaluate the best placement for all calls
best_placements = evaluate_best_placement_for_all_calls(solution, problem)

# Step 2: Apply the best placements to the current solution
new_solution = apply_smart_operator(solution, best_placements)

return new_solution
Use code with caution.
def place_call_in_best_position(solution, call_id, pickup_position, delivery_position):
"""
Places a call's pickup and delivery in the best positions in the same vehicle.
"""
if solution is None:
raise ValueError("Solution cannot be None")

new_solution = copy.deepcopy(solution)

print(new_solution)

# Find and remove the existing pickup and delivery positions for the call
while call_id in new_solution:
    print("kom meg hit")
    new_solution.remove(call_id)
    print("yes!")
    print(new_solution)

# Insert the call back into the optimal positions (pickup first, delivery second)
new_solution.insert(pickup_position, call_id)
print(pickup_position)
print("1")
print(new_solution)
print(delivery_position)
new_solution.insert(delivery_position, call_id)
print("2")

return new_solution
Use code with caution.
def greedy_reinsert(solution, problem):
"""
Removes a random call from the solution and reinserts it in the
most cost-effective (greedy) position.

Args:
    solution: Current solution (list of call indices with 0 as vehicle separator)
    problem: Problem instance data
    
Returns:
    list: New solution after greedy reinsertion
"""
# Make a copy to avoid modifying the original solution
new_solution = solution.copy()

# Split the solution into vehicles
vehicles = split_into_vehicles(new_solution)

print("vehicles" + str(vehicles))

# Get all non-empty vehicles
non_empty_vehicles = [v_idx for v_idx, v in enumerate(vehicles) if v and any(c != 0 for c in v)]
if not non_empty_vehicles:
    return new_solution

# Choose a random vehicle that contains at least one call
from_vehicle_idx = random.choice(non_empty_vehicles)
from_vehicle = vehicles[from_vehicle_idx - 1]

print("vehicle number " + str(from_vehicle_idx - 1) + str(from_vehicle))

# Choose a random call from the vehicle
call_options = [c for c in from_vehicle if c != 0]
if not call_options:
    return new_solution

chosen_call = random.choice(call_options)

# Remove the chosen call from its current position
call_indices = [i for i, c in enumerate(from_vehicle) if c == chosen_call]
if len(call_indices) != 2:
    # Ensure we have both pickup and delivery
    return new_solution

# Remove in reverse order to maintain correct indices
from_vehicle.pop(max(call_indices))
from_vehicle.pop(min(call_indices))

# Store the updated vehicles list
vehicles[from_vehicle_idx - 1] = from_vehicle

# Now find the best position to reinsert
best_cost = float('inf')
best_insert_vehicle = None
best_pickup_pos = None
best_delivery_pos = None

# Try inserting in each vehicle (including the original one)
for v_idx, vehicle in enumerate(vehicles):
    # Check if this vehicle can handle this call
    #if not np.all(problem['VesselCargo'][v_idx, [chosen_call-1]]):
    #    continue
    if chosen_call - 1 >= len(problem['VesselCargo']):
        continue
    
    # Try each possible insertion position for pickup
    for pickup_pos in range(len(vehicle) + 1):
        vehicle_with_pickup = vehicle.copy()
        vehicle_with_pickup.insert(pickup_pos, chosen_call)
        
        # Verify the intermediate solution is feasible
        print("vehicle_with_pickup" + str(vehicle_with_pickup))
        print("v_idx" + str(v_idx))

        if not check_vehicle_feasibility(vehicle_with_pickup, v_idx, problem):
            continue
        
        # Try each possible delivery position after pickup
        for delivery_pos in range(pickup_pos + 1, len(vehicle_with_pickup) + 1):
            test_vehicle = vehicle_with_pickup.copy()
            test_vehicle.insert(delivery_pos, chosen_call)
            
            # Check feasibility of the complete solution
            if not check_vehicle_feasibility(test_vehicle, v_idx - 1, problem):
                continue
            
            # Create the test solution to evaluate
            test_vehicles = vehicles.copy()
            test_vehicles[v_idx] = test_vehicle
            test_solution = combine_vehicles(test_vehicles)
            
            # Calculate cost
            test_cost = cost_function(test_solution, problem)
            
            # Update best solution if this is better
            if test_cost < best_cost:
                best_cost = test_cost
                best_insert_vehicle = v_idx
                best_pickup_pos = pickup_pos
                best_delivery_pos = delivery_pos

# If we found a valid insertion position, use it
if best_insert_vehicle is not None:
    best_vehicle = vehicles[best_insert_vehicle]
    
    # Insert at the best positions (delivery first to maintain pickup position)
    best_vehicle.insert(best_pickup_pos, chosen_call)
    best_vehicle.insert(best_delivery_pos, chosen_call)
    
    vehicles[best_insert_vehicle] = best_vehicle
    
    return combine_vehicles(vehicles)

# If no better position found, restore the call to its original position
# This shouldn't happen often if the problem is well-defined
from_vehicle = vehicles[from_vehicle_idx - 1]
from_vehicle.insert(call_indices[0], chosen_call)
from_vehicle.insert(call_indices[1], chosen_call)
vehicles[from_vehicle_idx - 1] = from_vehicle

return combine_vehicles(vehicles)
Use code with caution.
#hvordan skal jeg lagre beste løsning og i tillegg til å noen ganger akseptere en dårligere løsning?

def update_operator_probabilities(operator_improvements, operator_probabilities, num_operators):
total_improvements = sum(operator_improvements)
if total_improvements > 0:
for i in range(num_operators):
operator_probabilities[i] = operator_improvements[i] / total_improvements
else:
operator_probabilities[:] = [1/num_operators] * num_operators

def normalize_scores(operator_scores):
"""Normalize operator scores to prevent extreme values"""
min_score = 0.1  # Minimum allowed score
max_score = 10.0  # Maximum allowed score

for idx in range(len(operator_scores)):
    operator_scores[idx] = max(min_score, min(max_score, operator_scores[idx]))
Use code with caution.
def General_Adaptive_Metahuristics_Framework(problem, initial_solution):
""" General Adaptive Metahuristics Framework for Pickup and Delivery Problem with Adaptive Operator Selection """

best_placements = {}

max_iterations = 10000
escape_condition = 100
update_frequency = 100

#s <- initial_solution
current_solution = initial_solution.copy()
current_cost = cost_function(current_solution, problem)

#solution s_best <- s
best_solution = initial_solution.copy()
best_cost = cost_function(best_solution, problem)

iterations_since_best = 0
iterations_since_escape = 0
# skal jeg ha med noe som skiller på bedring og bedring med escape?
iterations_since_escape_best = 0
iteration = 0

operators = [shuffle_vehicle,
            swap_calls, 
            dummy_reinsert,
            one_reinsert,
            #best_placement_operator
            greedy_reinsert
            ]
num_operators = len(operators)

operator_scores = [1.0] * num_operators
operator_improvements = [0] * num_operators
operator_probabilities = [1/num_operators] * num_operators

normalize_scores(operator_scores)

while iteration < max_iterations:
    if iterations_since_escape > escape_condition:
        print(f"Iteration {iteration}: Escape triggered")
        #apply an escape algorithm
        current_solution = escape(
            current_solution,
            problem,
            iterations_since_best
            )
        
        current_cost = cost_function(
            current_solution,
            problem
            )
        
       #må sjekke om det er en forbedring og kun oppdatere iterations since best da,
       #men trenger kanskje en annen måling siden jeg også skal vite hvor lenge siden jeg fikk en bra løsning
        
        iterations_since_escape = 0 # er vel ikke sikkert at det blir en forbedring

        #Kan jeg ha med denne eller blir det sabotasje av rammeverket og escape?
        # kommer det senere så jeg bare skal gjøre iterations since best helt tilslutt?
        if current_cost < best_cost:
            best_solution = current_solution
            best_cost = current_cost
            iterations_since_best = 0
            print(f"New best solution found with cost {best_cost}")

     #s_marked <- s
     #Trenger jeg incumbent eller bør jeg bruke current_solution?
    incumbent = current_solution.copy()
    incumbent_cost = current_cost

   
    #select a heuristic, from the set of heuristics based on selection parameters
    #apply the heuristic to the s_marked
    selected_operator = select_heuristic(
            operator_scores,
            iteration,
            max_iterations
            )
    
    assert 0 <= selected_operator < len(operators), f"Invalid operator index: {selected_operator}"

    print(f"Iteration {iteration}: Selected operator {selected_operator}")

    new_solution = operators[selected_operator](incumbent, problem)
    new_cost = cost_function(new_solution, problem)

    feasible, _ = feasibility_check(new_solution, problem)

    if feasible:
    #if the cost of the new solution is better than the cost of the best solution
        if new_cost < best_cost:
            #s_best <- s_marked
            best_solution = new_solution
            best_cost = new_cost
            #jeg bør lage et mer detaljert poengsystem for å se hvilke operatorer som er best
            #Hvis de gir meg en bedre løsning så bør de få mer poeng 4
            #hvis det er drastisk bedre løsning 5
            #hvis de gir meg en original løsning bør de få litt poeng 2
            #samme beste løsning som før 1
            operator_scores[selected_operator] += 1
            normalize_scores(operator_scores)
            iterations_since_best = 0
            print(f"New best solution found with cost {best_cost}")
        
    # kan jeg bruke min simulated annealing her i steden for å lage en ny funksjon?
    #if accept(s_marked, s) then
    accepted = accept_solution(
        new_solution,
        new_cost,
        incumbent_cost, 
        problem,
        1.0
        )
    
    print(f"Old cost: {incumbent_cost}, New cost: {new_cost}, Accepted: {accepted}")

    if accepted:
        #s <- s_marked
        current_solution = new_solution
        current_cost = new_cost

    #update selecion parameters and iterate iterations_since_best
    iterations_since_best += 1  
    iteration += 1

    if iteration % update_frequency == 0:
        update_operator_probabilities(operator_improvements=operator_improvements,
                                    operator_probabilities=operator_probabilities,
                                    num_operators=num_operators)
        
        operator_improvements = [0] * num_operators  # Nullstill forbedringstellere
        print(f"Updated operator probabilities: {operator_probabilities}")

print(f"Best solution:{best_solution} found with cost {best_cost}")
return best_solution
Use code with caution.
def escape1(current_solution, problem):

escape_solution = current_solution.copy()

# Denne kan jeg tweeke litt mer på??
num = random.randint(2, 5)
for _ in range(num):

    escape_method = random.choice([
        shuffle_vehicle,
        swap_calls,
        dummy_reinsert
    ])
    
    escape_solution = escape_method(escape_solution, problem)

return escape_solution
Use code with caution.
def escape(current_solution, problem, iterations_since_best):
"""
Mer intelligent unnsluppe-metode

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
    shuffle_vehicle,
    swap_calls,
    dummy_reinsert,
    one_reinsert,
    greedy_reinsert
]

# Antall ganger vi kjører unnsluppe-operatorer øker med intensitet
num_escapes = int(2 + escape_intensity * 5)

for _ in range(num_escapes):
    # Vekt operatorene basert på intensitet
    weights = [
        1.0, 
        1.0 + escape_intensity, 
        1.0 + 2 * escape_intensity
    ]
    
    escape_method = random.choices(escape_methods, weights=weights)[0]
    escape_solution = escape_method(escape_solution, problem)

return escape_solution
Use code with caution.
def select_heuristic1(operator_scores):
"""Select a heuristic based on their scores using roulette wheel selection"""
indices = list(range(len(operator_scores)))
return random.choices(indices, weights=operator_scores, k=1)[0]

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
Use code with caution.
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