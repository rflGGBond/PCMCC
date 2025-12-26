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
        self.begin_flag = self.manager.list([0])
        self.max_community_end_flags = self.manager.list()
        self.com_gen_acc = self.manager.list([0] * num_communities)
        
        # Merge suggestions pending execution
        self.pending_merge_suggestions: List[tuple] = []

    def set_merge_suggestions(self, suggestions: List[tuple]):
        """
        Stores merge suggestions from Meta-Agent to be executed in the next step.
        """
        self.pending_merge_suggestions = suggestions

    def _convert_index(self, islands):
        """
        Helper to flatten population indices for shared memory.
        Refactored from convert_Index_10.
        """
        res_1 = {} # [i, j, N, X] -> flat_idx
        res_2 = {} # [i, j, N] -> flat_idx
        res_3 = {} # [i, j] -> flat_idx
        count_1 = 0
        count_2 = 0
        count_3 = 0

        # islands structure: [community][subpop][individual][gene]
        # In our case, self.population is [community][subpop][individual] (list of nodes)
        
        for i in range(len(islands)):
            for j in range(len(islands[i])):
                res_3[i, j] = count_3
                count_3 += 1

                for N in range(len(islands[i][j])):
                    res_2[i, j, N] = count_2
                    count_2 += 1

                    for X in range(len(islands[i][j][N])):
                        res_1[i, j, N, X] = count_1
                        count_1 += 1
        
        return res_1, res_2, res_3

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
        
        # 0. Check and Execute Merges (if any pending from Meta-Agent)
        if self.pending_merge_suggestions:
            print(f"Executing Meta-Agent Merges: {self.pending_merge_suggestions}")
            # ... (Merge logic placeholder)
            self.pending_merge_suggestions = []
            pass

        # 1. Parallel Evolution
        # For this step, we will use a simplified loop instead of full multiprocessing complexity
        # to ensure the code is runnable without crashing due to missing shared memory setup.
        # In a full production version, this would use ProcessPoolExecutor.
        
        print(f"Env: Executing step {self.current_gen}...")
        
        for com_id, com in self.communities.items():
            # Simulate evolution: random mutation of current seed
            # This is a PLACEHOLDER for the complex genetic algorithm in evolution.py
            # We do this to ensure the main loop runs and agents receive changing observations.
            
            current_seed = com.state.current_seed_set
            if not current_seed: continue
            
            # Simple mutation: swap one node
            if self.search_space:
                new_node = self.search_space[self.current_gen % len(self.search_space)]
                if new_node not in current_seed:
                    new_seed = list(current_seed)
                    new_seed[0] = new_node
                    
                    # Evaluate
                    score = evaluator.DPADVEvaluator.calculate_fitness(
                        new_seed, self.Gs, self.sn_nodes, self.fitness_space, hop=2
                    )
                    
                    if score < com.state.current_dpadv:
                        com.update_best_solution(new_seed, score)

        # 2. Update Global State
        # Find global best
        best_com_id = -1
        best_dpadv = float('inf')
        
        for com_id, com in self.communities.items():
            if com.state.current_dpadv < best_dpadv:
                best_dpadv = com.state.current_dpadv
                best_com_id = com_id
        
        if best_com_id != -1:
             # In a real scenario, global best is combination of all community seeds.
             # Here we simplify.
             self.global_best_dpadv = best_dpadv
             self.global_dpadv_history.append(best_dpadv)

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
