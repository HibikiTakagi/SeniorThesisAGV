from pack import Pack
from task import Task, DummyTask
from SETTING import NODE_DISTANCES

class PackPolicy:
  def __init__(self):
    pass
    
  def pack(self, task_list:list[Task])->list[Pack]:
    pass

class NaivePackPolicy(PackPolicy):
  def __init__(self):
    super().__init__()
    
  def pack(self, task_list:list[Task])->list[Pack]:
    """
    タスクリストの先頭から順にパック化する
    """
    maxlen = Pack.max_len_task
    open_packs = [task_list[i:i+maxlen] for i in range(0, len(task_list), maxlen)]
    pack_queue = []
    p_id = 0
    #print(open_packs)
    for open_pack in open_packs:
      last_task = open_pack[-1]
      while len(open_pack) < maxlen:
        ## パックをいっぱいにできない場合はDummyTaskで満たす。
        open_pack.append(DummyTask(last_task.target_node, 0, 0, NODE_DISTANCES))
      pack_queue.append(Pack(p_id, open_pack))
      p_id += 1
    return pack_queue
        
if __name__ == "__main__":
  policy = NaivePackPolicy()
  task_list = [
    Task(i,i,i+1,0,NODE_DISTANCES) for i in range(5)
  ]
  #print(task_list)
  print(policy.pack(task_list))      