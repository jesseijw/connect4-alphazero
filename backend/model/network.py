# neural net that takes board state and outputs move probabilities + win value
# using a small CNN bc the board is only 6x7 so we dont need anything crazy

import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4Net(nn.Module):

    def __init__(self):
        super(Connect4Net, self).__init__()

        # conv layers to pick up patterns on the board
        # like 4 in a row, 3 in a row etc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # policy head - ts figures out which column is best
        # outputs 7 values, one per column
        self.policyConv = nn.Conv2d(64, 2, kernel_size=1)
        self.policyFC = nn.Linear(2 * 6 * 7, 7)

        # value head - figures out whos winning, goes from -1 to 1
        self.valueConv = nn.Conv2d(64, 1, kernel_size=1)
        self.valueFC1 = nn.Linear(1 * 6 * 7, 64)
        self.valueFC2 = nn.Linear(64, 1)

    def forward(self, boardState):
        # boardState shape: (batch, 1, 6, 7)

        # run thru the conv layers first
        x = F.relu(self.conv1(boardState))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # policy head stuff
        policyStuff = F.relu(self.policyConv(x))
        policyStuff = policyStuff.view(policyStuff.size(0), -1)
        policyStuff = self.policyFC(policyStuff)
        # softmax so all the outputs add up to 1 (probabilities)
        policyOut = F.softmax(policyStuff, dim=1)

        # value head stuff
        valueStuff = F.relu(self.valueConv(x))
        valueStuff = valueStuff.view(valueStuff.size(0), -1)
        valueStuff = F.relu(self.valueFC1(valueStuff))
        # tanh squishes the output between -1 and 1
        valueOut = torch.tanh(self.valueFC2(valueStuff))

        return policyOut, valueOut


def prepareBoard(boardState):
    # converts numpy board into a tensor the net can actually use
    # shape goes from (6, 7) to (1, 1, 6, 7)
    boardTensor = torch.FloatTensor(boardState).unsqueeze(0).unsqueeze(0)
    return boardTensor
