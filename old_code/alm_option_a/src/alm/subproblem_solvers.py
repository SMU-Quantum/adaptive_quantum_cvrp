# cvrp_tripartite_solver/src/alm/subproblem_solvers.py

import random
import heapq 
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field

from common.cvrp_instance import CVRPInstance # Will be loaded in __main__

@dataclass(order=True) 
class Label:
    cost: float = field(compare=True)
    load: int = field(compare=False) 
    current_node: int = field(compare=False)
    path: Tuple[int, ...] = field(compare=False, hash=True) 

    def __hash__(self):
        return hash((self.cost, self.load, self.current_node, self.path))

    def __eq__(self, other):
        if not isinstance(other, Label):
            return NotImplemented
        return (self.cost == other.cost and
                self.load == other.load and
                self.current_node == other.current_node and
                self.path == other.path)

    def dominates(self, other_label: "Label") -> bool:
        if self.current_node != other_label.current_node:
            return False 

        cost_dominance = self.cost <= other_label.cost
        load_dominance = self.load <= other_label.load
        
        if cost_dominance and load_dominance:
            if self.cost < other_label.cost or self.load < other_label.load:
                return True
        return False

def solve_espprc_with_dominance(
    instance: CVRPInstance,
    modified_node_rewards: Dict[int, float],
    tabu_customers: Optional[Set[int]] = None,
    debug_espprc: bool = False # Add a debug flag
) -> Optional[List[int]]:
    depot = instance.depot
    capacity = instance.capacity
    if tabu_customers is None:
        tabu_customers = set()

    if instance.dimension == 1 and depot == 0:
        if debug_espprc: print("  DEBUG_ESPPRC: Instance has only depot, returning [0,0]")
        return [0, 0] 

    non_dominated_labels_at_node: Dict[int, List[Label]] = {i: [] for i in range(instance.dimension)}
    
    pq_counter = 0 
    initial_label = Label(cost=0.0, load=0, current_node=depot, path=(depot,))
    unprocessed_labels_pq: List[Tuple[float, int, Label]] = []
    
    heapq.heappush(unprocessed_labels_pq, (initial_label.cost, pq_counter, initial_label))
    pq_counter += 1
    non_dominated_labels_at_node[depot].append(initial_label) 

    completed_routes: List[Tuple[float, Tuple[int, ...]]] = []
    visited_states_costs: Dict[Tuple[int, int, frozenset], float] = {}
    extensions_count = 0
    # Increased safeguard for potentially deeper searches in E-n13-k4
    max_extensions_safeguard = (instance.dimension ** 3) * instance.dimension + 2000 


    if debug_espprc: print(f"  DEBUG_ESPPRC: Starting ESPPRC. Depot={depot}, Cap={capacity}, Tabu={tabu_customers}")
    if debug_espprc: print(f"  DEBUG_ESPPRC: Initial Rewards (sample): {dict(list(modified_node_rewards.items())[:5])}")


    while unprocessed_labels_pq and extensions_count < max_extensions_safeguard:
        extensions_count += 1
        
        _current_cost_in_pq, _tie_breaker, current_label = heapq.heappop(unprocessed_labels_pq)
        if debug_espprc and extensions_count % 500 == 0 : # Print periodically
            print(f"    DEBUG_ESPPRC: Ext {extensions_count}, PQ size {len(unprocessed_labels_pq)}, Popped Label: {current_label}")


        customers_visited_in_current_path = frozenset(n for n in current_label.path if n != depot)
        state_key = (current_label.current_node, current_label.load, customers_visited_in_current_path)

        if state_key in visited_states_costs and visited_states_costs[state_key] <= current_label.cost:
            if debug_espprc: print(f"    DEBUG_ESPPRC: Pruning (visited_states_costs) Label to {current_label.current_node}. Cost {current_label.cost:.2f} vs stored {visited_states_costs[state_key]:.2f}. Path: {current_label.path}")
            continue 
        visited_states_costs[state_key] = current_label.cost
        
        is_dominated_by_processed = False
        # This check is tricky. A label popped from PQ should be non-dominated by labels *already fully processed and removed from PQ*.
        # The non_dominated_labels_at_node list contains labels that were non-dominated *when they were added*.
        # A simple check: if current_label.cost is worse than an already established non-dominated label at this node for similar resources.
        # This is effectively part of the dominance check when adding, but can be re-checked here.
        # For now, relying on dominance check before adding to PQ.

        for next_node_idx in range(instance.dimension):
            if next_node_idx == current_label.current_node:
                continue

            if next_node_idx != depot and next_node_idx in tabu_customers:
                if debug_espprc: print(f"    DEBUG_ESPPRC: Skipping extension to {next_node_idx} (tabu)")
                continue

            original_arc_cost = instance.distance_matrix[current_label.current_node][next_node_idx]
            reward_for_next_node = 0.0
            if next_node_idx != depot and next_node_idx in modified_node_rewards:
                reward_for_next_node = modified_node_rewards[next_node_idx]
            
            modified_arc_cost = original_arc_cost - reward_for_next_node
            new_path_cost = current_label.cost + modified_arc_cost

            new_load = current_label.load
            if next_node_idx != depot:
                new_load += instance.demands[next_node_idx]

            if new_load > capacity:
                if debug_espprc: print(f"    DEBUG_ESPPRC: Skipping extension {current_label.current_node}->{next_node_idx} (capacity: {new_load} > {capacity})")
                continue 

            if next_node_idx != depot and next_node_idx in current_label.path:
                if debug_espprc: print(f"    DEBUG_ESPPRC: Skipping extension {current_label.current_node}->{next_node_idx} (elementarity: {next_node_idx} in {current_label.path})")
                continue 

            new_path_tuple = current_label.path + (next_node_idx,)
            new_generated_label = Label(cost=new_path_cost,
                                        load=new_load,
                                        current_node=next_node_idx,
                                        path=new_path_tuple)
            if debug_espprc: print(f"    DEBUG_ESPPRC: Try extend {current_label.current_node}->{next_node_idx}. New Label: C={new_generated_label.cost:.2f}, L={new_generated_label.load}, P={new_generated_label.path}")


            if next_node_idx == depot: 
                if len(new_generated_label.path) > 2: 
                    completed_routes.append((new_generated_label.cost, new_generated_label.path))
                    if debug_espprc: print(f"    DEBUG_ESPPRC: Completed route: {new_generated_label.path}, ModCost: {new_generated_label.cost:.2f}")
            else: 
                is_dominated_by_existing = False
                for existing_label in non_dominated_labels_at_node[next_node_idx]:
                    if existing_label.dominates(new_generated_label):
                        is_dominated_by_existing = True
                        if debug_espprc: print(f"      DEBUG_ESPPRC: New label {new_generated_label} to {next_node_idx} DOMINATED by existing {existing_label}")
                        break
                
                if not is_dominated_by_existing:
                    current_non_dominated_list = non_dominated_labels_at_node[next_node_idx]
                    new_nd_list_for_node = []
                    added_new_label_to_nd_list = False
                    for existing_label in current_non_dominated_list:
                        if not new_generated_label.dominates(existing_label):
                            new_nd_list_for_node.append(existing_label)
                        elif debug_espprc:
                            print(f"      DEBUG_ESPPRC: New label {new_generated_label} DOMINATES existing {existing_label} at {next_node_idx}. Removing existing.")
                    
                    new_nd_list_for_node.append(new_generated_label) # Add the new non-dominated label
                    non_dominated_labels_at_node[next_node_idx] = new_nd_list_for_node
                    
                    heapq.heappush(unprocessed_labels_pq, (new_generated_label.cost, pq_counter, new_generated_label))
                    pq_counter += 1
                    if debug_espprc: print(f"      DEBUG_ESPPRC: Added to PQ: {new_generated_label}")
                # else: # new_label was dominated
                    # if debug_espprc: print(f"      DEBUG_ESPPRC: New label to {next_node_idx} was dominated, not added to PQ.")
    
    if extensions_count >= max_extensions_safeguard:
        if debug_espprc: print(f"  DEBUG_ESPPRC: Reached max_extensions_safeguard ({max_extensions_safeguard})")

    if not completed_routes:
        if debug_espprc: print(f"  DEBUG_ESPPRC: No completed routes found. Total extensions: {extensions_count}")
        return None 

    completed_routes.sort(key=lambda x: x[0]) 
    best_modified_cost, best_path_tuple = completed_routes[0]
    
    if debug_espprc: print(f"  DEBUG_ESPPRC: Returning best route: {list(best_path_tuple)} with mod_cost: {best_modified_cost:.2f}. Total completed: {len(completed_routes)}")
    return list(best_path_tuple)


def solve_espprc_placeholder(*args, **kwargs): # Keep signature flexible for now
    # This is the old placeholder, can be removed if not used by any tests.
    # For safety, let's make it return None if called unexpectedly.
    return None


if __name__ == '__main__':
    from common.cvrp_instance import load_cvrp_instance 
    import os

    print("--- Direct ESPPRC Debugging for E-n13-k4 ---")
    
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    instance_file_path = os.path.join(project_root_path, "data", "cvrplib_instances", "E-n13-k4.vrp")

    if not os.path.exists(instance_file_path):
        print(f"ERROR: Instance file not found at {instance_file_path}")
    else:
        print(f"Loading instance: {instance_file_path}")
        e_n13_k4_instance = load_cvrp_instance(instance_file_path)
        print(f"Instance '{e_n13_k4_instance.name}' loaded. Dimension: {e_n13_k4_instance.dimension}, Depot: {e_n13_k4_instance.depot}")

        customer_indices_e_n13_k4 = [i for i in range(e_n13_k4_instance.dimension) if i != e_n13_k4_instance.depot]
        
        debug_rewards = {}
        # Unvisited from log: [3, 4, 5, 7, 8, 9, 10, 11]
        # Let's try to make customer 3 very attractive
        unvisited_target = 3 

        for cust_idx in customer_indices_e_n13_k4:
            if cust_idx == unvisited_target:
                debug_rewards[cust_idx] = 1000.0 # High positive reward for target
            elif cust_idx in {1,2,6,12}: # Customers previously picked
                 debug_rewards[cust_idx] = -100.0 # Slightly penalize
            else:
                debug_rewards[cust_idx] = 0.0   # Neutral reward for others

        print(f"\nAttempting ESPPRC with high reward for customer {unvisited_target}:")
        print(f"Rewards (sample): Cust {unvisited_target}={debug_rewards.get(unvisited_target)}, Cust 1={debug_rewards.get(1)}")

        print("\nESPPRC Call (no tabu, verbose debug):")
        route1 = solve_espprc_with_dominance(
            instance=e_n13_k4_instance,
            modified_node_rewards=debug_rewards,
            tabu_customers=set(),
            debug_espprc=True # Enable verbose debugging for this call
        )
        if route1:
            print(f"\n  RESULT Route 1: {route1}")
            customers_in_route1 = {n for n in route1 if n != e_n13_k4_instance.depot}
            print(f"  Customers in Route 1: {customers_in_route1}")

            if unvisited_target in customers_in_route1:
                print(f"  SUCCESS: Target customer {unvisited_target} was included!")
            else:
                print(f"  INFO: Target customer {unvisited_target} was NOT included in the first route.")
        else:
            print("\n  RESULT Route 1: None")

