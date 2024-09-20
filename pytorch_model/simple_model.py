import os
import torch
from torch import nn
import torch

"""This is a model that is meant to model a CSGO match. 
The input is just a one hot encoding of the players on each team, the out put will be the number of kills each player on each team gets, 
the purpose is to model how specific unique players perform against eachother. There are over 200 unique players each with an average of 30 datapoints"""
class NeuralNetwork(nn.Module):
    def __init__(self, num_players):
        super().__init__()
        # represents player base kills
        self.bias = torch.nn.Parameter(data=torch.Tensor(num_players), requires_grad=True)

        #represents player influence on team
        self.cum_team_bias = torch.nn.Parameter(data=torch.Tensor(num_players,1), requires_grad=True)

        #represents player influence on enemy team
        self.cum_enemy_bias = torch.nn.Parameter(data=torch.Tensor(num_players,1), requires_grad=True)

        #Should represent the skill level of a player. Number of kills scale up and down based on how close in skill the teams are. Closer teams will have more kills, futher teams will have less
        self.skill_weight = torch.nn.Parameter(data=torch.Tensor(num_players, 1), requires_grad=True)

        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()

        self.bias.data.uniform_(10, 20)
        self.cum_team_bias.data.uniform_(-5, 5)
        self.cum_enemy_bias.data.uniform_(-5, 5)
        self.skill_weight.data.uniform_(0,1)

        nn.init.xavier_uniform_(self.cum_team_bias)
        nn.init.xavier_uniform_(self.cum_enemy_bias)
        nn.init.xavier_uniform_(self.skill_weight)
        # Keep uniform initialization for bias

        self.hidden_layer = nn.Sequential(
            nn.Linear(num_players, num_players),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # x is now expected to be of shape (batch_size, 2, num_players)
        batch_size = x.size(0)
        
        # Compute team and enemy biases
        team_bias = torch.matmul(x, self.cum_team_bias)  # (batch_size, 2, 1)
        enemy_bias = torch.flip(torch.matmul(x, self.cum_enemy_bias), dims=[1])  # (batch_size, 2, 1)
        
        # Compute skills
        skills = torch.matmul(x, self.skill_weight)  # (batch_size, 2, 1)
        kill_scalar = 1 / (torch.abs(skills[:, 0] - skills[:, 1]) + 1)  # (batch_size, 1)
        
        # Compute output
        o1 = self.activation1((x * self.bias.unsqueeze(0).unsqueeze(0)) + team_bias + enemy_bias)  # (batch_size, 2, num_players)
        o2 = torch.sum(o1, dim=1) * kill_scalar  # (batch_size, num_players)
        o2 = self.activation2(self.hidden_layer(o2))

        # Create a mask for active players
        active_mask = x.sum(dim=1) != 0  # (batch_size, num_players)

        # Apply the mask to o2
        o2 = o2 * active_mask  # (batch_size, num_players)
        
        return o2



