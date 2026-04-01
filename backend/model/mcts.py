# Monte Carlo Tree Search - uses neural net to guide the search
# instead of random rollouts like regular MCTS

import math
import numpy as np
from backend.model.network import prepareBoard

# how much to explore vs exploit
# higher = more exploration
C_PUCT = 1.5

class MCTSNode:

    def __init__(self, game, parent=None, moveThatGotHere=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.moveThatGotHere = moveThatGotHere

        # prior probability from the neural net policy head
        self.prior = prior

        # stats we track for this node
        self.visitCount = 0
        self.totalValue = 0.0

        # child nodes, one per valid move
        self.kiddos = {}

    def isLeaf(self):
        return len(self.kiddos) == 0

    def getAvgValue(self):
        if self.visitCount == 0:
            return 0.0
        return self.totalValue / self.visitCount

    def getUCB(self):
        # upper confidence bound - balances exploration and exploitation
        # nodes with high prior or low visits get explored more
        if self.parent is None:
            return 0
        exploration = C_PUCT * self.prior * math.sqrt(self.parent.visitCount) / (1 + self.visitCount)
        return self.getAvgValue() + exploration


class MCTS:

    def __init__(self, net, numSimulations=100):
        self.net = net
        self.numSimulations = numSimulations

    def search(self, game):
        rootNode = MCTSNode(game)

        for _ in range(self.numSimulations):
            self.runSimulation(rootNode)

        # return move probabilities based on visit counts
        moveCounts = np.zeros(7)
        for move, kiddo in rootNode.kiddos.items():
            moveCounts[move] = kiddo.visitCount

        # normalize to get probabilities
        moveProbs = moveCounts / moveCounts.sum()
        return moveProbs

    def runSimulation(self, node):
        # game is over, return the result
        if node.game.game_over:
            if node.game.winner is None:
                return 0.0  # draw
            # winner is the player who just moved (current player already switched)
            return -1.0

        # leaf node - expand it using the neural net
        if node.isLeaf():
            boardState = node.game.get_board_state()
            boardTensor = prepareBoard(boardState)

            policyOut, valueOut = self.net(boardTensor)
            policyOut = policyOut.detach().numpy()[0]
            valueOut = valueOut.detach().item()

            # only consider valid moves
            validMoves = node.game.get_valid_moves()
            for move in validMoves:
                clonedGame = node.game.clone()
                clonedGame.drop_piece(move)
                node.kiddos[move] = MCTSNode(
                    game=clonedGame,
                    parent=node,
                    moveThatGotHere=move,
                    prior=policyOut[move]
                )

            return -valueOut

        # pick the child with the highest UCB score
        bestKiddo = max(node.kiddos.values(), key=lambda k: k.getUCB())
        val = self.runSimulation(bestKiddo)

        # backpropagate - update this node with the result
        node.visitCount += 1
        node.totalValue += val

        return -val

    def getBestMove(self, game):
        moveProbs = self.search(game)
        # just pick the move with the highest visit count
        return int(np.argmax(moveProbs))
