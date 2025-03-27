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

