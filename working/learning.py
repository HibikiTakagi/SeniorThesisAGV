import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from env import JobSchedulerEnv
from distribute_policy import DQN_DistributePolicy
from utils_save_fig import save_fig_losses, save_fig_return, save_fig_reward
from SETTING import MODEL_PATH, NODE_DISTANCES

EPISODE = 400
#EPISODE = 2
SYNC = 10
TRAIN_FREQ = 10

class Learning:
  
  def __init__(self):
    pass
  
  #@profile
  def main(self):
    env = JobSchedulerEnv()
    gamma = env.scheduler.distribute_policy.gamma
    ret = 0
    ts = 0
    losses = []
    rets = []
    
    for eps in tqdm(range(EPISODE)):
      state, info = env.reset()
      terminated, truncated = (False, False)
      #for time in range(): # 下記のwhileで事足りそう
      rewards = []
      
      while not (terminated or truncated):
        ts += 1
        action = env.scheduler.distribute_policy.act(*state)
        next_state, reward, terminated, truncated, info = env.step(action)
        env.scheduler.distribute_policy.remember(state, action, reward, next_state, terminated, truncated, info)
        state = next_state
        
        ret = reward + gamma*ret
        rewards.append(reward)
        
        if ts % TRAIN_FREQ == 0:
          ts = 0   
          env.scheduler.distribute_policy.train()
      
      if eps%SYNC==0:
        env.scheduler.distribute_policy.update_target_model()
        env.scheduler.distribute_policy.save_model(MODEL_PATH)
      
      if env.scheduler.distribute_policy.loss is not None:
        losses.append([eps, env.scheduler.distribute_policy.loss])
        rets.append([eps, ret])
    
    save_fig_losses(losses)
    save_fig_return(rets)
    save_fig_reward(rewards)
    
    return copy.deepcopy(env.scheduler.dump_schedules(NODE_DISTANCES))

if __name__ == "__main__":
  train = Learning()
  _ = train.main()
