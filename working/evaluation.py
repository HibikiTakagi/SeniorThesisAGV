import copy
from env import JobSchedulerEnv
from distribute_policy import DistributePolicy, DQN_DistributePolicy
from robot import Robot
from task import Task
from scheduler import Scheduler
from SETTING import MODEL_PATH, NODE_DISTANCES

class Evaluation:
  def __init__(self):
    pass

  def main(self):
    env = JobSchedulerEnv()
    env.reset()
    if isinstance(env.scheduler.distribute_policy, DQN_DistributePolicy):
      env.scheduler.distribute_policy.load_model(MODEL_PATH)
    env.scheduler.distribute()
    resources, tasks, resource_allocation, start_times, durations, end_times = env.scheduler.dump_schedules(NODE_DISTANCES)
    print(resources)
    print(tasks)
    print(resource_allocation)
    print(start_times)
    print(durations)
    print(end_times)
    return copy.deepcopy((resources, tasks, resource_allocation, start_times, durations, end_times))

if __name__ == "__main__":
  eval = Evaluation()
  eval.main()
