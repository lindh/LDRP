import gym
import math
import numpy as np
import random
from typing import Tuple, List
from dataclasses import dataclass
from collections import deque
from collections import defaultdict

@dataclass
class AgentInfo:
    pos: Tuple[float, float]
    current_start: int
    current_goal: int
    action_history: List[int]
    pos_history: List[Tuple[float,float]]
    step_account: int = 0

@dataclass
class OtherAgentsInfo:
    #Store each agent's position for the number of steps
    other_agents_pos: List[List[Tuple[float,float]]]

class PBS:
    def __init__(self, args):
        self.env = gym.make("drp_env:drp-" + str(1) + "agent_" + args.map_name + "-v2", state_repre_flag = "onehot_fov")
        self.num_agents = args.agent_num
        self.time_limit = 100#args.time_limit
        self.schedule_actions = []
        self.goal_rec = []
        self.priority_rec = []
        self.change_rec = [] #List recording agents on nodes (used for changing priority)
        self.no_change_rec = [] #Record agents on edges
        self.tmp_goal_rec = [-1 for _ in range(self.num_agents)]


    #Decide priority (build the priority list)
    #Determine shortest paths in priority order
    #How to find the shortest path
    #Initialize the environment (set goal_array)
    #Take the head data of the deque (popleft)
    #Set the agent's data into the environment
    #Run steps for the number of avail_action; if reached goal, go to @
    #Create a dataclass instance with no collision, update the data, and append to the deque
    #@Refer to action_history to add to schedule_action; refer to pos_history to update pos
    def culc_actions(self, obs ,env):
        #AAAA
        #tmp = 0
        self.schedule_actions = [[] for _ in range(self.num_agents)]
        #Current: step x agent_num
        #other_agents_infos = OtherAgentsInfo(other_agents_pos=[[] for _ in range(self.time_limit - env.step_account)])
        #Revised: agent_num x step
        
        #tmp_list = [[] for _ in range(self.time_limit+1)]
        #other_agents_infos = OtherAgentsInfo(other_agents_pos=[[] for _ in range(self.num_agents)])
        other_agents_infos = OtherAgentsInfo(other_agents_pos=[[[] for _ in range(self.time_limit + 1)]
                                                               for _ in range(self.num_agents)])

        #Handling agents not on a node
        self.fill_non_nodes_agents_pos_history(env, other_agents_infos)
        
        priority_list = self.get_priority(obs, env)
        self.priority_rec = priority_list.copy()
        index = 0
        recal_count = 0
        #print("priority_list", priority_list)
        while index < len(priority_list):
            i = priority_list[index]
            index += 1

            agent_info = AgentInfo(pos=(env.obs[i][0], env.obs[i][1]), 
                                   current_start=env.current_start[i], 
                                   current_goal=env.current_goal[i], 
                                   action_history=[], 
                                   pos_history=[(env.obs[i][0], env.obs[i][1])],
                                   step_account=0)
            
            near_goal_nodes = self.env.get_near_nodes(env.goal_array[i])
            while self.schedule_actions[i] == []:
                #If there is no solution, recompute with that agent at top priority
                #What to do after recomputing agent_num times?
                #Fix to current_goal; if None, current_start
                if len(near_goal_nodes) == 0:
                    """
                    if i in self.change_rec:
                        self.change_rec.remove(i)
                        self.change_rec.insert(0, i)
                    elif i in self.no_change_rec:
                        self.no_change_rec.remove(i)
                        self.no_change_rec.insert(0,i)
                    
                    priority_list = self.change_rec.copy() + self.no_change_rec.copy()
                    """

                    priority_list.remove(i)
                    priority_list.insert(0,i)
                    self.priority_rec = priority_list.copy()
                    #print("priority_list", priority_list)
                    index = 0
                    recal_count += 1
                    
                    if recal_count >= self.num_agents:
                        #print("recompute count exceeded",i)
                        for j in range(self.num_agents):
                            if env.current_goal[j] is not None:
                                self.schedule_actions[j] = [env.current_goal[j]] * self.time_limit
                            else:
                                self.schedule_actions[j] = [env.current_start[j]] * self.time_limit
                    else:
                        self.schedule_actions = [[] for _ in range(self.num_agents)]
                    other_agents_infos = OtherAgentsInfo(other_agents_pos=[[[] for _ in range(self.time_limit + 1)]
                                                               for _ in range(self.num_agents)])
                    self.fill_non_nodes_agents_pos_history(env, other_agents_infos)
                    break
                tmp_goal = near_goal_nodes.pop(0)
                #print("agent", i, "try goal", tmp_goal)

                agent_infos_deque = deque([agent_info])
                current_agent_info = agent_info

                self.env.reset()
                goal_flag = False
                #Processing for overlapping conditions
                step_account_check = agent_info.step_account
                visitted_states = set()

                while not goal_flag and len(agent_infos_deque) > 0 and current_agent_info.step_account+1 < self.time_limit:
                    current_agent_info = agent_infos_deque.popleft()
                    #AAAA
                    #tmp += 1

                    #Processing for overlapping conditions
                    if step_account_check != current_agent_info.step_account:
                        visitted_states = set()
                        step_account_check = current_agent_info.step_account

                    self.env.reset()
                    self.set_env_info(current_agent_info, tmp_goal)

                    avail_actions = self.env.get_avail_agent_actions(0, self.env.n_actions)[1]
                    for action in avail_actions:

                        self.set_env_info(current_agent_info, tmp_goal)

                        obs, reward, done, info = self.env.step([action])
                        new_pos = (self.env.obs[0][0], self.env.obs[0][1])
                        #Processing for overlapping conditions
                        new_state = (round(new_pos[0]), round(new_pos[1]), action)
                        if new_state in visitted_states:
                            continue
                        visitted_states.add(new_state)

                        collision_flag = self.collision_detect(new_pos, other_agents_infos.other_agents_pos, current_agent_info.step_account+1, i)
                        
                        if collision_flag == True:#Reset because a bug occurs when collision and goal happen simultaneously
                            #self.env.reset()
                            pass

                        elif collision_flag == False:
                            new_action_history = current_agent_info.action_history + [action]
                            new_pos_history = current_agent_info.pos_history + [new_pos]
                            new_current_start = self.env.current_start[0]
                            new_current_goal = self.env.current_goal[0]

                            new_agent_info = AgentInfo(pos=new_pos, 
                                                    current_start=new_current_start, 
                                                        current_goal=new_current_goal, 
                                                    action_history=new_action_history, 
                                                    pos_history=new_pos_history,
                                                    step_account=current_agent_info.step_account + 1)
                            
                            if all(done) is True:
                                if info["goal"] == True:
                                    #Implement checking whether stopping after reaching the goal collides with a higher-priority agent
                                    #If it collides, continue
                                    #print("goal found")
                                    if self.check_after_collision(new_agent_info, other_agents_infos.other_agents_pos, i) == True:
                                        #print("collision after goal found")
                                        continue
                                    else:
                                        #print("no collision after goal")
                                        pass
                                    """
                                    print("AAAAAA",new_agent_info.step_account)
                                    print(current_agent_info.pos, other_agents_infos.other_agents_pos[current_agent_info.step_account])
                                    print(new_agent_info.pos_history)
                                    print(other_agents_infos.other_agents_pos)
                                    """
                                    #Current: store per step ()
                                    #Revised: store per agent (just put into the i-th)
                                    self.schedule_actions[i] = new_agent_info.action_history + new_agent_info.action_history[-1:]*(self.time_limit - len(new_agent_info.action_history))

                                    for _ in range(self.time_limit - len(new_agent_info.pos_history)+1):
                                        new_agent_info.pos_history.append(new_agent_info.pos_history[-1])

                                    other_agents_infos.other_agents_pos[i] = new_agent_info.pos_history
                                    
                                    goal_flag = True
                                    
                                    break

                                elif info["collision"] == True:
                                    pass
                                elif info["timeup"] == True:
                                    #print("timeup!!!!!!")
                                    pass

                            else:
                                agent_infos_deque.append(new_agent_info)

            ###
            #print(other_agents_infos.other_agents_pos)
        #AAAA
        #print(other_agents_infos.other_agents_pos)
        self.goal_rec = env.goal_array.copy()
        self.tmp_goal_rec = [self.schedule_actions[i][-1] for i in range(self.num_agents)]
        
    ###############################################
    def fill_non_nodes_agents_pos_history(self, env, other_agents_infos: OtherAgentsInfo) -> None:

        for i in range(self.num_agents):
            # If not on a node, store pos until reaching the node into other_agents_infos
            if env.pos[env.current_start[i]] != [env.obs[i][0], env.obs[i][1]]:
                agent_info = AgentInfo(
                    pos=(env.obs[i][0], env.obs[i][1]),
                    current_start=env.current_start[i],
                    current_goal=env.current_goal[i],
                    action_history=[],
                    pos_history=[(env.obs[i][0], env.obs[i][1])],
                )

                self.env.reset()
                self.set_env_info(agent_info, env.goal_array[i])

                while self.env.pos[self.env.current_start[0]] != [self.env.obs[0][0], self.env.obs[0][1]]:
                    self.set_env_info(agent_info, env.goal_array[i])
                    avail_actions = self.env.get_avail_agent_actions(0, self.env.n_actions)[1]
                    action = avail_actions[0]

                    obs, reward, done, info = self.env.step([action])

                    new_pos = (self.env.obs[0][0], self.env.obs[0][1])
                    agent_info = AgentInfo(
                        pos=new_pos,
                        current_start=self.env.current_start[0],
                        current_goal=self.env.current_goal[0],
                        action_history=agent_info.action_history + [action],
                        pos_history=agent_info.pos_history + [new_pos],
                        step_account=0,
                    )

                agent_info.pos_history.extend([agent_info.pos_history[-1]]*1)
                agent_info.pos_history.extend([] for _ in range(self.time_limit - len(agent_info.pos_history) + 1))

                other_agents_infos.other_agents_pos[i] = agent_info.pos_history

    ###############################################
    def get_priority(self, obs, env):
        priority_list = []
        change_list = []#List recording agents on nodes
        no_change_list = []
        """
        for i in range(self.num_agents):
            path_length = env.get_path_length(env.current_start[i], env.goal_array[i])
            priority_list.append((i, path_length))
        #Prioritize farther ones
        #priority_list.sort(key=lambda x: x[1], reverse=True)
        #Prioritize nearer ones
        priority_list.sort(key=lambda x: x[1], reverse=False)
        priority_list = [x[0] for x in priority_list]
        return priority_list
        """
        #Add agents on nodes at the end
        if self.priority_rec == []:
            for i in range(self.num_agents):
                path_length = env.get_path_length(env.current_start[i], env.goal_array[i])
                priority_list.append((i, path_length))
            #Prioritize farther ones
            #priority_list.sort(key=lambda x: x[1], reverse=True)
            #Prioritize nearer ones
            priority_list.sort(key=lambda x: x[1], reverse=False)
            priority_list = [x[0] for x in priority_list]
            return priority_list
        else:
            priority_list = self.priority_rec.copy()
            no_change_list = self.priority_rec.copy()
            for i in range(self.num_agents):
                if env.pos[env.current_start[i]] == [env.obs[i][0], env.obs[i][1]]:
                    priority_list.remove(i)
                    priority_list.append(i)
                    change_list.append(i)
                    no_change_list.remove(i)

            self.change_rec = change_list.copy()
            self.no_change_rec = no_change_list.copy()
            
            return priority_list
        
    
    #Set the agent's xy coordinates (self.obs) into the environment
    #obs=tuple(array[],array[]...)
    def set_env_info(self, agent_infos, goal_array):
        self.env.set_1agent_info(pos=agent_infos.pos, 
                                 current_start=agent_infos.current_start, 
                                 current_goal=agent_infos.current_goal, 
                                 goal_array = goal_array,)
        return
    
    #Collision detection
    #Whether agent_pos and other_agents_pos collide at step_t
    #Also checking one step before/after prevents collision with the swapped agent
    def collision_detect(self, agent_pos, other_agents_pos, step_t, agent_num):
        speed = 5
        collision_flag = False
        
        for i in range(len(other_agents_pos)):
            if i==agent_num or other_agents_pos[i][step_t] == []:
                continue
            
            distance = math.dist(agent_pos, other_agents_pos[i][step_t])
            distance_before = math.dist(agent_pos, other_agents_pos[i][step_t-1])
            distance_after = speed+1
            if len(other_agents_pos[i])>step_t+1 and other_agents_pos[i][step_t+1] != []:
                distance_after = math.dist(agent_pos, other_agents_pos[i][step_t+1])

            if distance<speed or distance_before<speed or distance_after<speed:
                collision_flag = True

        return collision_flag
    
    #After finding a pass, check there is no risk of collision afterward
    #Check only 2 steps
    def check_after_collision(self, agent_info, other_agents_pos, agent_num):
        collision_flag = False
        """
        for t in range(len(agent_info.pos_history)):
            if self.collision_detect(agent_info.pos_history[t], other_agents_pos, t,  agent_num):
                collision_flag = True
                break
        """
        for t in range(len(agent_info.pos_history), len(agent_info.pos_history)+2):
            if self.collision_detect(agent_info.pos_history[-1], other_agents_pos, t,  agent_num):
                collision_flag = True
                break
        return collision_flag

    def policy(self, obs, env):
        actions = []
        #Initialize when schedule_actions is abnormal
        if self.schedule_actions != []:
            for i in range(self.num_agents):
                if self.schedule_actions[i] == []:
                    self.schedule_actions = []
                    break
            
        #Recompute when someone's goal changes
        if self.goal_rec != env.goal_array:
            self.schedule_actions = []

        #When compromising at a node near the goal, decide when to recompute
        #Idea 1: when an intermediate node is specified as tmp, recompute upon arrival: when someone's current_start matches tmp_goal
        ##Do not exclude those that stay in place (it might be better to exclude them)
        for i in range(self.num_agents):
            if env.current_start[i] == self.tmp_goal_rec[i]:
                self.schedule_actions = []
                break

        #Idea 2: run until none remain: write nothing
        

        if self.schedule_actions == []:
            #print("compute anew")
            self.culc_actions(obs, env)
            #print("schedule_actions", self.schedule_actions)
            #return [0 for _ in range(self.num_agents)]
            
        for i in range(self.num_agents):
            actions.append(self.schedule_actions[i].pop(0))

        return actions