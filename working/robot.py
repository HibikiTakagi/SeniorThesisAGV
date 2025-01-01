import numpy as np
from numpy.typing import NDArray
import random

class Robot:
  def __init__(self, r_id, current_node):
    self.id = r_id
    self.pack_list = []
    self.start_node = current_node
    self.current_node = current_node
    self.pack = None
    self.current_pack = None
        
    self.current_task = None
    
  def __repr__(self):
    return f'Robot(id:{self.id}, pack_list:{self.pack_list}, current_node:{self.current_node})'
    
  @property
  def name(self):
    return f"R{self.id}"

  @property
  def pack_id_queue(self):
    return [pack.id for pack in self.pack_list]  
  
  def reset(self):
    self.pack_list = []
    self.current_node = self.start_node
    self.pack = None
    self.current_pack = None
    self.current_task = None
    
  def queue_time_from_start(self, graph:NDArray[np.int32]):
    end_time = 0
    current_node = self.start_node
        
    for pack in self.pack_list:
      destination = pack.start_node
            
      move_task_durations = graph[current_node, destination]
      end_time += move_task_durations
                  
      pack_durations = pack.time_assume(graph)
      end_time += pack_durations

      current_node = pack.target_node
            
    return end_time

  def queue_job_late_time_from_start(self, graph:NDArray[np.int32]):
    job_late_time = 0
    current_node = self.start_node
    for pack in self.pack_list:
      destination = pack.start_node
      move_task_durations = graph[current_node, destination]
      job_late_time += 0 if move_task_durations==0 else 1/move_task_durations
            
      current_node = pack.start_node
      destination = pack.target_node
            
      task_durations = graph[current_node, destination]
      job_late_time += 0 if task_durations==0 else 1/task_durations
        
    return job_late_time

def generate_robot_list_random(number:int, len_node:int)->list[Robot]:
  assert number <= len_node
    
  current_nodes = random.sample(range(len_node), number)
  robot_list = []
  for r_id in range(number):
    robot_list.append(Robot(r_id=r_id, current_node=current_nodes[r_id]))
  return robot_list