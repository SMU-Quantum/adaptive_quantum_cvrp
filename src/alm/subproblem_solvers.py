# cvrp_tripartite_solver/src/alm/subproblem_solvers.py

import random
import heapq 
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field

from common.cvrp_instance import CVRPInstance

@dataclass(order=True, frozen=True) # Ensures __eq__ and __hash__ are based on all fields
class ESPathLabel:
    """
    Represents a label in the ESP labeling algorithm.
    A label corresponds to a partial path from the depot to 'current_node'.
    """
    cost: float = field(compare=True) 
    current_node: int = field(compare=True)
    path: Tuple[int, ...] = field(compare=True) 

    # frozen=True auto-generates __hash__ and __eq__ based on all fields.

    def dominates(self, other_label: "ESPathLabel") -> bool:
        """
        Checks if this label dominates another_label for ESP.
        Dominance: Reaches the same node, with the same path history,
                   and strictly better cost.
        """
        if self.current_node != other_label.current_node:
            return False 
        
        if self.path == other_label.path: 
            if self.cost < other_label.cost: # Strict inequality for cost
                return True
        return False

def solve_esp_with_dominance( 
    instance: CVRPInstance,
    modified_node_rewards: Dict[int, float],
    tabu_customers: Optional[Set[int]] = None,
    debug_esp: bool = False,
    capacity_multiplier: float = 0.0 
) -> Optional[List[int]]:
    depot = instance.depot
    if tabu_customers is None:
        tabu_customers = set()

    if instance.dimension == 1 and depot == 0:
        if debug_esp: print("  DEBUG_ESP: Instance has only depot, returning [0,0]")
        return [0, 0] 
    
    pq_counter = 0 
    initial_label = ESPathLabel(cost=0.0, current_node=depot, path=(depot,))
    unprocessed_labels_pq: List[Tuple[float, int, ESPathLabel]] = []
    
    heapq.heappush(unprocessed_labels_pq, (initial_label.cost, pq_counter, initial_label))
    pq_counter += 1
    
    completed_routes: List[Tuple[float, Tuple[int, ...]]] = []
    
    visited_states_costs: Dict[Tuple[int, frozenset], float] = {}
    visited_states_costs[(depot, frozenset())] = 0.0

    extensions_count = 0
    max_extensions_safeguard = (instance.dimension ** 3) * instance.dimension + 2000 

    if debug_esp: print(f"  DEBUG_ESP: Starting ESP. Depot={depot}, Tabu={tabu_customers}")
    if debug_esp: print(f"  DEBUG_ESP: Initial Rewards (sample): {dict(list(modified_node_rewards.items())[:5])}")

    while unprocessed_labels_pq and extensions_count < max_extensions_safeguard:
        extensions_count += 1
        
        _current_cost_in_pq, _tie_breaker, current_label = heapq.heappop(unprocessed_labels_pq)

        if debug_esp and extensions_count % 1000 == 0 : 
            print(f"    DEBUG_ESP: Ext {extensions_count}, PQ size {len(unprocessed_labels_pq)}, Popped Label: C={current_label.cost:.2f} N={current_label.current_node} P={current_label.path}")

        customers_visited_in_current_path = frozenset(n for n in current_label.path if n != depot)
        state_key = (current_label.current_node, customers_visited_in_current_path)

        if current_label.cost > visited_states_costs.get(state_key, float('inf')): 
            if debug_esp and extensions_count % 1000 == 0: print(f"    DEBUG_ESP: Pruning (cost {current_label.cost:.2f} > stored {visited_states_costs.get(state_key):.2f}) Label to {current_label.current_node}. Path: {current_label.path}")
            continue 
        
        for next_node_idx in range(instance.dimension):
            if next_node_idx == current_label.current_node:
                continue

            if next_node_idx != depot and next_node_idx in tabu_customers:
                continue

            original_arc_cost = instance.distance_matrix[current_label.current_node][next_node_idx]
            reward_for_next_node = 0.0
            if next_node_idx != depot and next_node_idx in modified_node_rewards:
                reward_for_next_node = modified_node_rewards[next_node_idx]
            
            modified_arc_cost = original_arc_cost - reward_for_next_node
            new_path_cost = current_label.cost + modified_arc_cost

            if next_node_idx != depot and next_node_idx in current_label.path:
                continue 

            new_path_tuple = current_label.path + (next_node_idx,)
            new_generated_label = ESPathLabel(cost=new_path_cost,
                                              current_node=next_node_idx,
                                              path=new_path_tuple)
            
            if debug_esp and extensions_count < 100 and len(unprocessed_labels_pq) < 50 : 
                 print(f"    DEBUG_ESP: Try extend {current_label.current_node}->{next_node_idx}. New Label: C={new_generated_label.cost:.2f}, P={new_generated_label.path}")

            if next_node_idx == depot and len(new_generated_label.path) > 2:
                # compute load and violations
                load = sum(instance.demands[n] for n in new_generated_label.path if n != depot)
                violation = max(0, load - instance.capacity)
                # add Lagrange‐multiplier penalty
                penalized_cost = new_generated_label.cost + capacity_multiplier * violation
                completed_routes.append((penalized_cost, new_generated_label.path))
            else: 
                new_customers_visited_for_next_state = customers_visited_in_current_path | {next_node_idx}
                new_state_key = (next_node_idx, new_customers_visited_for_next_state)
                
                if new_generated_label.cost < visited_states_costs.get(new_state_key, float('inf')):
                    visited_states_costs[new_state_key] = new_generated_label.cost
                    heapq.heappush(unprocessed_labels_pq, (new_generated_label.cost, pq_counter, new_generated_label))
                    pq_counter += 1
    
    if extensions_count >= max_extensions_safeguard:
        if debug_esp: print(f"  DEBUG_ESP: Reached max_extensions_safeguard ({max_extensions_safeguard})")

    if not completed_routes:
        if debug_esp: print(f"  DEBUG_ESP: No completed routes found. Total extensions: {extensions_count}")
        return None 

    completed_routes.sort(key=lambda x: x[0]) 
    best_modified_cost, best_path_tuple = completed_routes[0]
    
    if debug_esp: print(f"  DEBUG_ESP: Returning best route: {list(best_path_tuple)} with mod_cost: {best_modified_cost:.2f}. Total completed: {len(completed_routes)}")
    return list(best_path_tuple)

def solve_espprc_placeholder(*args, **kwargs): 
    return None

if __name__ == '__main__':
    from common.cvrp_instance import load_cvrp_instance 
    import os

    print("--- Direct ESP Debugging ---")
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    instance_file_path = os.path.join(project_root_path, "data", "cvrplib_instances", "E-n13-k4.vrp")

    if not os.path.exists(instance_file_path):
        print(f"ERROR: Instance file not found at {instance_file_path}")
    else:
        print(f"Loading instance: {instance_file_path}")
        test_instance = load_cvrp_instance(instance_file_path)
        print(f"Instance '{test_instance.name}' loaded. Dimension: {test_instance.dimension}, Depot: {test_instance.depot}")

        customer_indices_test = [i for i in range(test_instance.dimension) if i != test_instance.depot]
        target_customer = customer_indices_test[2] if len(customer_indices_test) > 2 else (customer_indices_test[0] if customer_indices_test else -1)
        
        debug_rewards = {}
        if target_customer != -1: # Ensure there are customers
            for cust_idx in customer_indices_test:
                if cust_idx == target_customer:
                    debug_rewards[cust_idx] = 1000.0 
                elif customer_indices_test and cust_idx == customer_indices_test[0] and cust_idx != target_customer :
                     debug_rewards[cust_idx] = -100.0
                else:
                    debug_rewards[cust_idx] = 0.0   

            print(f"\nAttempting ESP with high reward for customer {target_customer}:")
            first_cust_for_log_idx = customer_indices_test[0] if customer_indices_test else -1 # Handle empty list
            reward_first_cust_for_log = debug_rewards.get(first_cust_for_log_idx if isinstance(first_cust_for_log_idx, int) else -1 , "N/A")


            print(f"Rewards (sample): Cust {target_customer}={debug_rewards.get(target_customer)}, Cust {first_cust_for_log_idx}={reward_first_cust_for_log}")


            print("\nESP Call (no tabu, verbose debug):")
            route1 = solve_esp_with_dominance(
                instance=test_instance,
                modified_node_rewards=debug_rewards,
                tabu_customers=set(),
                debug_esp=True 
            )
            if route1:
                print(f"\n  RESULT Route 1: {route1}")
            else:
                print("\n  RESULT Route 1: None")
        else:
            print("No customers to target for ESP debug in __main__.")

    instance_no_cust = CVRPInstance(name="no_cust", dimension=1, capacity=100, distance_matrix=[[0]], demands=[0], depot=0)
    route_no_cust = solve_esp_with_dominance(instance_no_cust, {})
    print(f"Route for no_cust instance: {route_no_cust}")