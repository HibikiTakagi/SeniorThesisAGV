import matplotlib.pyplot as plt
import plotly.graph_objects as go
from SETTING import FIG_LOSS_PATH, FIG_RET_PATH, FIG_REWARD_PATH, FIG_GANTT_RESOURCE_PATH, FIG_GANTT_TASK_PATH

def save_fig_losses(losses):
  fig_loss, ax = plt.subplots()
  x_eps, y_loss = zip(*losses)
  plt.plot(x_eps, y_loss, '-', label='loss')
  ax.set_xlabel('episodes')
  ax.set_ylabel('loss')
  ax.grid(True)
  plt.legend(loc=0)
  fig_loss.savefig(FIG_LOSS_PATH)
  plt.clf()
  plt.close()

def save_fig_return(returns):
  fig_convergence_return, ax = plt.subplots()
  x_eps, y_return = zip(*returns)
  plt.plot(x_eps, y_return, '-', label="train_return") 
  #plt.plot(target_returns,label="target_return")
  ax.set_xlabel('episodes')
  ax.set_ylabel('Returns')
  ax.grid(True)
  plt.legend(loc=0)
  fig_convergence_return.savefig(FIG_RET_PATH)
  plt.clf()
  plt.close()

def save_fig_reward(rewards):
  fig_rewards = plt.figure()
  plt.plot(rewards, label="rewards")
  plt.legend(loc=0)
  fig_rewards.savefig(FIG_REWARD_PATH)
  plt.close()

def save_fig_gantt_resource(tasks, resource_allocation, start_times, end_times):
  fig = go.Figure()
  for task, resource, start, end in zip(tasks, resource_allocation, start_times, end_times):
    fig.add_trace(go.Bar(
      x=[end - start],  ## タスクの続行時間を幅として設定
      y=[resource],     ## リソースを縦軸として設定
      base=start,       ## タスクの開始タイムステップを設定
      orientation='h',  ## 水平バーを描画
      text=task,        ## タスク名をラベルとして表示
      name=task
    ))

  ## レイアウト調整
  fig.update_layout(
    title="Gantt Chart with Resources",
    xaxis_title="Timesteps",
    yaxis_title="Resources",
    xaxis=dict(
      tickmode="linear",  ## タイムステップの目盛りを線形に設定
      tick0=0,
      dtick=200
    ),
    barmode='stack',
    showlegend=True
  )

  ## グラフ表示
  fig.write_html(FIG_GANTT_RESOURCE_PATH)

def save_fig_gantt_task(tasks, start_times, end_times):
  ## ガントチャート用データ作成
  fig = go.Figure()

  for task, start, end in zip(tasks, start_times, end_times):
    fig.add_trace(go.Bar(
      x=[end - start],  ## タスクの続行時間を幅として設定
      y=[task],         ## タスク名を縦軸として設定
      base=start,       ## タスクの開始タイムステップを設定
      orientation='h',  ## 水平バーを描画
      name=task
    ))

  ## レイアウト調整
  fig.update_layout(
    title="Gantt Chart with Timesteps",
    xaxis_title="Timesteps",
    yaxis_title="Tasks",
    xaxis=dict(
      tickmode="linear",  ## タイムステップの目盛りを線形に設定
      tick0=0,
      dtick=200
    ),
    barmode='stack',
    showlegend=True
  )

  ## グラフ表示
  fig.write_html(FIG_GANTT_TASK_PATH)