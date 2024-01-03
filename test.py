import torch

# 假设有一个4x3的张量，代表4个状态和3个可能的动作的Q值
q_values = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print("Q values:")
print(q_values)

# 假设有一个长度为4的张量，代表每个状态选择的动作
actions = torch.tensor([0, 2, 1, 0])
print("\nActions:")
print(actions)

# 使用gather函数选择每个状态对应的动作的Q值
selected_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
print("\nSelected Q values:")
print(selected_q_values)