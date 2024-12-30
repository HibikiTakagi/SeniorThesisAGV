import copy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from env import JobSchedulerEnv
from distribute_policy import DistributePolicy, DQN_DistributePolicy
from robot import Robot
from task import Task
from scheduler import Scheduler
from utils_save_fig import save_fig_reward, save_fig_gantt_resource, save_fig_gantt_task
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
    save_fig_gantt_resource(tasks, resource_allocation, start_times, end_times)
    save_fig_gantt_task(tasks, start_times, end_times)
    return copy.deepcopy((resources, tasks, resource_allocation, start_times, durations, end_times))

class StepEvaluation:
  def __init__(self):
    pass

  def main(self):
    env = JobSchedulerEnv()
    assert isinstance(env.scheduler.distribute_policy, DQN_DistributePolicy)
    env.scheduler.distribute_policy.load_model(MODEL_PATH)
    env.scheduler.distribute_policy.eval()
    ts = 0
    state, info = env.reset()
    terminated, truncated = (False, False)
    rewards = []

    while not (terminated or truncated):
      ts += 1
      action = env.scheduler.distribute_policy.act(*state)
      next_state, reward, terminated, truncated, info = env.step(action)
      state = next_state
      rewards.append(reward)

    save_fig_reward(rewards)
    resources, tasks, resource_allocation, start_times, durations, end_times = env.scheduler.dump_schedules(NODE_DISTANCES)

    save_fig_gantt_resource(tasks, resource_allocation, start_times, end_times)
    save_fig_gantt_task(tasks, start_times, end_times)
    return copy.deepcopy((resources, tasks, resource_allocation, start_times, durations, end_times))

if __name__ == "__main__":
  eval = Evaluation()
  eval.main()
