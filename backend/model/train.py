# Self-play training loop
# the model plays against itself to generate training data
# then trains on that data, then repeats

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from backend.game.connect4 import Connect4
from backend.model.network import Connect4Net, prepareBoard
from backend.model.mcts import MCTS

# how many self-play games to run per training iteration
GAMES_PER_ITER = 50
# how many training iterations total
NUM_ITERS = 10
# how many simulations MCTS runs per move
NUM_SIMS = 100
# where to save the model
SAVE_PATH = "backend/model/checkpoints/model.pt"


def selfPlayGame(net):
    # plays one full game and returns the training examples
    # each example is (boardState, moveProbs, outcome)

    game = Connect4()
    mcts = MCTS(net, numSimulations=NUM_SIMS)
    gameStuff = []  # stores (boardState, moveProbs, currentPlayer)

    while not game.game_over:
        boardState = game.get_board_state()
        moveProbs = mcts.search(game)

        gameStuff.append((boardState, moveProbs, game.current_player))

        # pick move - sample from probabilities during training for exploration
        validMoves = game.get_valid_moves()
        maskedProbs = np.zeros(7)
        for m in validMoves:
            maskedProbs[m] = moveProbs[m]
        maskedProbs /= maskedProbs.sum()

        chosenMove = np.random.choice(7, p=maskedProbs)
        game.drop_piece(chosenMove)

    # figure out the outcome for each saved state
    trainingExamples = []
    for boardState, moveProbs, player in gameStuff:
        if game.winner is None:
            outcome = 0.0  # draw
        elif game.winner == player:
            outcome = 1.0  # this player won
        else:
            outcome = -1.0  # this player lost

        trainingExamples.append((boardState, moveProbs, outcome))

    return trainingExamples


def trainOnExamples(net, optimizer, allExamples):
    # shuffle and train on all the examples we collected
    np.random.shuffle(allExamples)

    totalLoss = 0.0

    for boardState, moveProbs, outcome in allExamples:
        boardTensor = prepareBoard(boardState)
        policyTarget = torch.FloatTensor(moveProbs).unsqueeze(0)
        valueTarget = torch.FloatTensor([[outcome]])

        policyOut, valueOut = net(boardTensor)

        # policy loss - how wrong were our move probabilities
        policyLoss = -torch.sum(policyTarget * torch.log(policyOut + 1e-8))

        # value loss - how wrong was our win prediction
        valueLoss = nn.MSELoss()(valueOut, valueTarget)

        lossyStuff = policyLoss + valueLoss

        optimizer.zero_grad()
        lossyStuff.backward()
        optimizer.step()

        totalLoss += lossyStuff.item()

    avgLoss = totalLoss / len(allExamples)
    print(f"  avg loss: {avgLoss:.4f}")


def train():
    net = Connect4Net()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # make sure checkpoint folder exists
    os.makedirs("backend/model/checkpoints", exist_ok=True)

    for iteration in range(NUM_ITERS):
        print(f"iteration {iteration + 1}/{NUM_ITERS}")

        # collect self play data
        allExamples = []
        for gameNum in range(GAMES_PER_ITER):
            print(f"  self play game {gameNum + 1}/{GAMES_PER_ITER}")
            gameExamples = selfPlayGame(net)
            allExamples.extend(gameExamples)

        print(f"  training on {len(allExamples)} examples...")
        trainOnExamples(net, optimizer, allExamples)

        # save the model after each iteration
        torch.save(net.state_dict(), SAVE_PATH)
        print(f"  saved model to {SAVE_PATH}")

    print("training done!")


if __name__ == "__main__":
    train()
