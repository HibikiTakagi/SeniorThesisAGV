import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size) ## 入力層から隠れ層への全結合層
    self.fc2 = nn.Linear(hidden_size, output_size) ## 隠れ層から出力層への全結合層

  def forward(self, x, mask=None):
    x = F.relu(self.fc1(x)) ## 隠れ層でReLU活性化関数を使用
    x = self.fc2(x) ## 出力層（スコア）
        
    if mask is not None:
      x = x.masked_fill(mask == 0, float('-inf'))  ## マスクされた部分に-∞を設定
        
    x = F.softmax(x, dim=-1)  ## Softmaxを適用して確率を計算
    return x
    
if __name__ == "__main__":
  ## ハイパーパラメータ
  input_size = 4    ## 状態の次元数
  hidden_size = 128 ## 隠れ層のニューロン数
  output_size = 3   ## 行動の種類（例: 静止、左、右）

  policy_net = Net(input_size, hidden_size, output_size)

  ## 入力のサンプル（状態）
  sample_state = torch.tensor([1.0, 0.5, -0.5, 2.0])

  ## マスク（1の行動は選択可能、0の行動は選択不可）
  mask = torch.tensor([1, 0, 1]) 

  ## ネットワークを通して行動の確率を計算
  pi = policy_net(sample_state, mask)

  ## 結果を表示
  print("行動確率:", pi)