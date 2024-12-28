from env import JobSchedulerEnv
from distribute_policy import DQN_DistributePolicy
from SETTING import MODEL_PATH

EPISODE = 1000
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
    
    for eps in range(EPISODE):
      state, info = env.reset()
      terminated, truncated = (False, False)
      #for time in range(): # 以下のwhileで事足りそう
      while not (terminated or truncated):
        ts += 0
        action = env.scheduler.distribute_policy.act(*state)
        next_state, reward, terminated, truncated, info = env.step(action)
        env.scheduler.distribute_policy.remember(state, action, reward, next_state, terminated, truncated, info)
        state = next_state
        
        ret = reward + gamma*ret
        if terminated or truncated:
          if env.scheduler.distribute_policy.loss is None:
            print(f"Episode {eps+1}/{EPISODE} - Final reward: {reward:5.3f}, Return:{ret:5.3f} Epsilon: {env.scheduler.distribute_policy.epsilon:.2f}")
          else:
            print(f"Episode {eps+1}/{EPISODE} - Final reward: {reward:5.3f}, Return:{ret:5.3f} Epsilon: {env.scheduler.distribute_policy.epsilon:.2f}, loss:{env.scheduler.distribute_policy.loss:.3f}")
      
      if ts % TRAIN_FREQ == 0:
        ts = 0   
        env.scheduler.distribute_policy.train()
      
      if eps%SYNC==0:
        env.scheduler.distribute_policy.update_target_model()
        env.scheduler.distribute_policy.save_model(MODEL_PATH)
        

if __name__ == "__main__":
  train = Learning()
  train.main()
