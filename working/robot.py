import numpy as np
from numpy.typing import NDArray
from pack import Pack

class Robot:
  def __init__(self, r_id, current_node):
    self.id = r_id
    self.pack_queue = []
    self.start_node = current_node
    self.current_node = current_node
    self.pack = None
    self.current_pack = None
        
    self.current_task = None
    
  def __repr__(self):
    return f'Robot(id:{self.id}, pack_queue:{self.pack_queue}, current_node:{self.current_node})'
    
  @property
  def name(self):
    return f"R{self.id}"
    
  def queue_time_from_start(self, graph:NDArray[np.int32], pack_list:list[Pack]):
    end_time = 0
    current_node = self.start_node
        
    for pack_id in self.pack_queue:
      pack = pack_list[pack_id]
      destination = pack.start_node
            
      move_task_durations = graph[current_node, destination]
      end_time += move_task_durations
      
      current_node = pack.start_node
      destination = pack.target_node

      task_durations = graph[current_node, destination]
      end_time += task_durations
            
    return end_time
            
    
        