import numpy as np
from numpy.typing import NDArray
from map_maker import NODE_DISTANCES

class Task:
  def __init__(self, t_id:int, start_node:int, target_node:int, birth_time:int, graph:NDArray[np.int32]):
    self.id = t_id
    self.start_node = start_node
    self.target_node = target_node
    #assert start_node != target_node
    self.birth_time = birth_time
    self.is_complete = False
    self.path_distance = graph[start_node, target_node]
    
  def __repr__(self):
    return f'Task(id:{self.id},\n start_node:{self.start_node},\n target_node:{self.target_node},\n birth_time:{self.birth_time},\n is_complete:{self.is_complete},\n path_distance:{self.path_distance})'
    
  def name(self):
    return f'T{self.id}'
    
  def get_waiting_time(self, current_time:int):
    return current_time - self.birth_time
    
  def complete(self):
    self.is_complete = True

class DummyTask(Task):
  def __init__(self, start_node, target_node, birth_time, graph):
    super().__init__(-1, start_node, start_node, birth_time, graph)
    self.is_complete = True

if __name__ == "__main__":
  task_a = Task(0, 0, 1, 0, NODE_DISTANCES)
  print(task_a)
  print(task_a.get_waiting_time(10))
  task_a.complete()
  print(task_a)