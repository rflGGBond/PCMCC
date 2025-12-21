import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import heapq
import networkx as nx
import copy
from typing import List, Dict, Any

from ..core import graph_ops, evaluator, evolution, merger
from .community import Community
from ..utils.types import CommunityAction, MetaAction

class PCMCCEnvironment:
    def __init__(self, graph_path: str, sn_nodes: List[int], total_budget: int, num_communities: int):
        self.graph_path = graph_path
        self.sn_nodes = sn_nodes
        self.total_budget = total_budget
        self.initial_num_communities = num_communities
        
        # Load Graph
        self.G = nx.Graph()
        with open(graph_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    n, m, w = int(parts[0]), int(parts[1]), float(parts[2])
                    self.G.add_edge(n, m, weight=w)
        
        # Initialize Spaces
        self._init_spaces()
        
        # Initialize Communities
        self.communities: Dict[int, Community] = {}
        self._init_communities()
        
        # Global State
        self.current_gen = 0
        self.global_dpadv_history = []
        self.global_best_seed = []
        self.global_best_dpadv = float('inf')
        
        # Parallel Execution Context
        self.manager = multiprocessing.Manager()
        self.shared_islands = self.manager.list()
        self.shared_islands_effect = self.manager.list()
        self.locks = self.manager.list()
        # ... (Other shared vars would need to be initialized here similarly to original code)
        
        # For simplicity in this refactor, we are not fully re-implementing the complex 
        # shared memory synchronization of the original code in the __init__.
        # We assume the `step` method will set up the necessary structures for `evolution.evolve_community`.

    def _init_spaces(self):
        # 1. Fitness Space
        self.fitness_space = []
        cur_un_ergodic = copy.deepcopy(self.sn_nodes)
        hop_f = 0
        hop = 2 # Hardcoded for now, should be param
        
        while hop_f <= hop:
            cur_ergodic = copy.deepcopy(cur_un_ergodic)
            self.fitness_space += cur_ergodic
            if hop_f == hop:
                break
            else:
                cur_un_ergodic = []
                for u in cur_ergodic:
                    for v in self.G.neighbors(u):
                        cur_un_ergodic.append(v)
                cur_un_ergodic = list(set(cur_un_ergodic) - set(self.fitness_space))
                hop_f += 1
        self.fitness_space = list(set(self.fitness_space) - set(self.sn_nodes))

        # 2. Search Space
        self.search_space = []
        cur_un_ergodic = copy.deepcopy(self.fitness_space)
        hop_s = 0
        while hop_s <= hop:
            cur_ergodic = copy.deepcopy(cur_un_ergodic)
            self.search_space += cur_ergodic
            if hop_s == hop:
                break
            else:
                cur_un_ergodic = []
                for u in cur_ergodic:
                    for v in self.G.neighbors(u):
                        cur_un_ergodic.append(v)
                cur_un_ergodic = list(set(cur_un_ergodic) - set(self.search_space + self.sn_nodes))
                hop_s += 1
                
        # 3. Graph Subgraph (Gs)
        all_nodes = list(set(self.search_space + self.sn_nodes))
        self.Gs = self.G.subgraph(all_nodes).copy()
        
        # 4. Search Space Reduction
        # (Simplified)
        self.search_space = heapq.nlargest(
            min(len(self.search_space), int(12 * self.total_budget)), 
            self.search_space, 
            key=lambda x: self.Gs.degree(x)
        )

    def _init_communities(self):
        # Community Division
        parts = graph_ops.detect_communities(self.Gs, self.initial_num_communities)
        
        # Calculate N_prob (needed for budget assignment)
        N_prob = evaluator.DPADVEvaluator.calculate_negative_probability(
            self.Gs, self.sn_nodes, self.fitness_space, hop=2
        )
        
        # Assign Budgets (Simplified logic from original)
        # ...
        
        for i, part in enumerate(parts):
            # Budget assignment logic would go here
            budget = self.total_budget // len(parts) # Placeholder
            self.communities[i] = Community(i, part, budget)

    def step(self):
        """
        Executes one generation of evolution.
        """
        self.current_gen += 1
        # 1. Parallel Evolution
        # Here we would call evolution.evolve_community using ProcessPoolExecutor
        # This requires setting up all the shared lists that were in the original `undirected.py`
        
        # For the purpose of this refactor structure, we will print a placeholder
        # print(f"Env: Executing step {self.current_gen}...")
        
        # 2. Update Global State
        # self.global_best_dpadv = ...
        pass

    def get_global_observation(self):
        # Return MetaObservation
        pass

    def apply_community_action(self, community_id: int, action: CommunityAction):
        """
        Applies the action from a Community Agent, including strict candidate evaluation.
        """
        com = self.communities.get(community_id)
        if not com: return
        
        # 1. Update Parameters (Mode A)
        if action.parameters:
            com.update_parameters(action.parameters)
        
        # 2. Candidate Generation (Mode B) - Try-Evaluate-Revert Logic
        if action.candidate_seed_set:
            # Validate constraints (e.g., size match)
            if len(action.candidate_seed_set) != com.state.budget:
                print(f"Agent {community_id} provided seed set of wrong size. Ignored.")
                return

            # Construct Global Candidate: Combine new candidate with OTHER communities' current bests
            global_candidate_seed = []
            for cid, other_com in self.communities.items():
                if cid == community_id:
                    global_candidate_seed.extend(action.candidate_seed_set)
                else:
                    global_candidate_seed.extend(other_com.state.current_seed_set)
            
            # Calculate Global DPADV for this candidate configuration
            new_global_dpadv = evaluator.DPADVEvaluator.calculate_fitness(
                global_candidate_seed, self.Gs, self.sn_nodes, self.fitness_space, hop=2
            )
            
            # Compare with current global best
            if new_global_dpadv < self.global_best_dpadv:
                print(f"Agent {community_id} found BETTER global solution (DPADV: {self.global_best_dpadv} -> {new_global_dpadv}). Accepted.")
                # Accept: Update Community Best & Global Best
                com.update_best_solution(action.candidate_seed_set, new_global_dpadv) # Note: Local DPADV is approximation here
                self.global_best_dpadv = new_global_dpadv
                self.global_best_seed = global_candidate_seed
            else:
                # Reject: Do nothing (Revert is implicit by not applying)
                # print(f"Agent {community_id} candidate rejected (DPADV: {new_global_dpadv} >= {self.global_best_dpadv}).")
                pass

    def apply_meta_action(self, action: MetaAction):
        # Update global baselines
        # Update budgets
        # Execute merges
        pass
