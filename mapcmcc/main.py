import os
import sys
import time
from typing import List, Dict

# Add the parent directory to sys.path to allow imports if running as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mapcmcc.environment.env import PCMCCEnvironment
from mapcmcc.agents.community_agent import CommunityAgent
from mapcmcc.agents.meta_agent import MetaAgent
from mapcmcc.utils.types import CommunityObservation, MetaObservation

def main():
    # Configuration
    GRAPH_PATH = "../../graph/BA3000.txt"
    SN_NODES = [0, 1, 2] # Placeholder, should load from select_SN
    TOTAL_BUDGET = 50
    NUM_COMMUNITIES = 4
    MAX_GEN = 20
    T_COMM = 5 # Communication interval
    
    # Initialize Environment
    print("Initializing MAPCMCC Environment...")
    env = PCMCCEnvironment(GRAPH_PATH, SN_NODES, TOTAL_BUDGET, NUM_COMMUNITIES)
    
    # Initialize Agents
    community_agents = {}
    for com_id in env.communities:
        community_agents[com_id] = CommunityAgent(agent_id=f"ComAgent_{com_id}")
    
    meta_agent = MetaAgent()
    
    print("Starting Evolution Loop...")
    start_time = time.time()
    
    for gen in range(1, MAX_GEN + 1):
        print(f"\n--- Generation {gen} ---")
        
        # 1. Standard Evolution Step (PCMCC)
        env.step()
        
        # 2. Agent Interaction (Every T_comm generations)
        if gen % T_COMM == 0:
            print(">>> Triggering Multi-Agent Interaction")
            
            # A. Community Agents
            for com_id, agent in community_agents.items():
                # Get Observation
                # obs = env.communities[com_id].get_observation(...)
                # For now, mock observation construction since Env logic is skeletal
                obs = CommunityObservation(
                    community_id=com_id,
                    current_generation=gen,
                    global_stage="exploration",
                    budget=env.communities[com_id].state.budget,
                    current_dpadv=0.5,
                    dpadv_history=[],
                    diversity_score=0.1,
                    top_k_score_nodes=[],
                    current_seed_set=[],
                    boundary_info={},
                    global_dpadv=0.4
                )
                
                # Get Action (LLM/Rule-Based)
                action = agent.get_action(obs)
                
                # Apply Action
                env.apply_community_action(com_id, action)
                
            # B. Meta Agent
            # obs = env.get_global_observation()
            # Mock
            meta_obs = MetaObservation(
                current_generation=gen,
                current_global_dpadv=0.4,
                global_dpadv_history=[],
                community_summaries=[],
                merge_history=[]
            )
            
            meta_action = meta_agent.get_action(meta_obs)
            
            # 2.1 Apply Meta-Agent Suggestions to Environment
            # Specifically handling Merge Suggestions which need to override/guide the standard merge logic
            if meta_action.merge_suggestions:
                print(f"Meta-Agent suggests merging: {meta_action.merge_suggestions}")
                # In a real implementation, we would pass these suggestions to env.step() 
                # or a specific env.merge() method.
                # For this refactor, we simulate the 'Force Merge' signal.
                # env.force_merge(meta_action.merge_suggestions)
                pass
            
            env.apply_meta_action(meta_action)
            
        # 3. Check Convergence
        # if env.check_convergence(): break
        
    end_time = time.time()
    print(f"\nEvolution Finished. Total Time: {end_time - start_time:.2f}s")
    # print(f"Best Global DPADV: {env.global_best_dpadv}")

if __name__ == "__main__":
    main()
