# FastAPI server - connects React frontend to the AlphaZero model
# run with: uvicorn backend.api:app --reload

import numpy as np
import torch
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.game.connect4 import Connect4
from backend.model.network import Connect4Net
from backend.model.mcts import MCTS

# creates da FastAPI server and tells it its oki to have requests come from localhost:3000
# bc w/o this da browser would block React and Python since they on diff ports
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# runs once da server starts, checks if trained model exists and loads it
# if not uses random weights bc untrained. net.eval() turns off train mode
SAVE_PATH = "backend/model/checkpoints/model.pt"
net = Connect4Net()
if os.path.exists(SAVE_PATH):
    net.load_state_dict(torch.load(SAVE_PATH))
    print("loaded trained model!")
else:
    print("no trained model found, using random weights")

net.eval()
mcts = MCTS(net, numSimulations=100)

# ts defines da shape of what React sends us and what we send back
class MoveRequest(BaseModel):
    board: list
    currentPlayer: int

class MoveResponse(BaseModel):
    move: int
    gameOver: bool
    winner: int


@app.post("/move", response_model=MoveResponse)
def getAIMove(req: MoveRequest):
    # reconstruct game state from what frontend sent us
    gamey = Connect4()
    gamey.board = np.array(req.board)
    gamey.current_player = req.currentPlayer

    # check if game already over before we do anything
    if gamey.check_win(1):
        gamey.game_over = True
        gamey.winner = 1
    elif gamey.check_win(-1):
        gamey.game_over = True
        gamey.winner = -1
    elif len(gamey.get_valid_moves()) == 0:
        gamey.game_over = True

    if gamey.game_over:
        return MoveResponse(move=-1, gameOver=True, winner=gamey.winner or 0)

    # run mcts to get da best move, drop it, check if game ended
    bestMove = mcts.getBestMove(gamey)
    gamey.drop_piece(bestMove)

    winnerVal = 0
    if gamey.winner is not None:
        winnerVal = gamey.winner

    return MoveResponse(
        move=bestMove,
        gameOver=gamey.game_over,
        winner=winnerVal
    )


@app.get("/")
def root():
    return {"message": "connect4 alphazero api is running"}
