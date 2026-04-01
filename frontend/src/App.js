import { useState, useEffect, useRef } from 'react';
import './App.css';

const ROWS = 6;
const COLS = 7;
const CELL_SIZE = 75;
const GAP = 10;
const PADDING = 20;

// makes a fresh empty board (6x7 of zeros)
function makeEmptyBoard() {
  return Array(ROWS).fill(null).map(() => Array(COLS).fill(0));
}

function App() {
  // board state, whos turn it is, game over stuff
  const [board, setBoard] = useState(makeEmptyBoard());
  const [currentPlayer, setCurrentPlayer] = useState(1);
  const [gameOver, setGameOver] = useState(false);
  const [winner, setWinner] = useState(null);
  const [isThinking, setIsThinking] = useState(false);
  const [hoveredCol, setHoveredCol] = useState(null);
  const [time, setTime] = useState(0);
  const [winningCells, setWinningCells] = useState([]);
  const timerRef = useRef(null);

  // start timer on mount, stop when game over
  useEffect(() => {
    if (!gameOver) {
      timerRef.current = setInterval(() => setTime(t => t + 1), 1000);
    } else {
      clearInterval(timerRef.current);
    }
    return () => clearInterval(timerRef.current);
  }, [gameOver]);

  // handles when da player clicks a column
  async function handleColClick(col) {
    if (gameOver || isThinking) return;

    // find da lowest empty row in that col and drop the piece
    const newBoard = board.map(row => [...row]);
    let placedRow = -1;
    for (let row = ROWS - 1; row >= 0; row--) {
      if (newBoard[row][col] === 0) {
        newBoard[row][col] = 1;
        placedRow = row;
        break;
      }
    }

    // col is full, do nothing
    if (placedRow === -1) return;

    setBoard(newBoard);

    // check if player just won
    const playerWin = checkWin(newBoard, 1);
    if (playerWin) {
      setWinningCells(playerWin);
      setGameOver(true);
      setWinner(1);
      return;
    }

    // check draw
    if (newBoard[0].every(cell => cell !== 0)) {
      setGameOver(true);
      setWinner(0);
      return;
    }

    // now its AI's turn, call da backend
    setIsThinking(true);
    setCurrentPlayer(-1);

    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL || "http://localhost:8000"}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ board: newBoard, currentPlayer: -1 })
      });
      const data = await res.json();

      // drop AI's piece on da board
      const aiBoard = newBoard.map(row => [...row]);
      for (let row = ROWS - 1; row >= 0; row--) {
        if (aiBoard[row][data.move] === 0) {
          aiBoard[row][data.move] = -1;
          break;
        }
      }

      setBoard(aiBoard);
      setCurrentPlayer(1);

      if (data.gameOver) {
        // find winning cells for the AI too
        const aiWin = checkWin(aiBoard, -1);
        if (aiWin) setWinningCells(aiWin);
        setGameOver(true);
        setWinner(data.winner);
      }
    } catch (err) {
      console.log("something went wrong calling da api:", err);
    }

    setIsThinking(false);
  }

  // checks 4 in a row, returns the 4 winning [row, col] pairs or null
  function checkWin(b, player) {
    // horizontal
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS - 3; c++) {
        if ([0,1,2,3].every(i => b[r][c+i] === player))
          return [[r,c],[r,c+1],[r,c+2],[r,c+3]];
      }
    }
    // vertical
    for (let r = 0; r < ROWS - 3; r++) {
      for (let c = 0; c < COLS; c++) {
        if ([0,1,2,3].every(i => b[r+i][c] === player))
          return [[r,c],[r+1,c],[r+2,c],[r+3,c]];
      }
    }
    // diagonal down-right
    for (let r = 0; r < ROWS - 3; r++) {
      for (let c = 0; c < COLS - 3; c++) {
        if ([0,1,2,3].every(i => b[r+i][c+i] === player))
          return [[r,c],[r+1,c+1],[r+2,c+2],[r+3,c+3]];
      }
    }
    // diagonal down-left
    for (let r = 0; r < ROWS - 3; r++) {
      for (let c = 3; c < COLS; c++) {
        if ([0,1,2,3].every(i => b[r+i][c-i] === player))
          return [[r,c],[r+1,c-1],[r+2,c-2],[r+3,c-3]];
      }
    }
    return null;
  }

  // calculates the center pixel position of a cell on the board
  function cellCenter(r, c) {
    const x = PADDING + c * (CELL_SIZE + GAP) + CELL_SIZE / 2;
    const y = PADDING + r * (CELL_SIZE + GAP) + CELL_SIZE / 2;
    return { x, y };
  }

  // calculates the line style for the winning 4 cells
  function getWinLineStyle() {
    if (winningCells.length < 4) return null;
    const start = cellCenter(winningCells[0][0], winningCells[0][1]);
    const end = cellCenter(winningCells[3][0], winningCells[3][1]);
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const length = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx) * (180 / Math.PI);
    return {
      position: 'absolute',
      left: start.x,
      top: start.y - 4,
      width: length,
      height: 8,
      background: '#ff1744',
      borderRadius: 4,
      opacity: 1,
      boxShadow: '0 0 12px 4px #ff1744',
      transformOrigin: '0 50%',
      transform: `rotate(${angle}deg)`,
      pointerEvents: 'none',
    };
  }

  // resets everything back to a fresh game
  function resetGame() {
    setBoard(makeEmptyBoard());
    setCurrentPlayer(1);
    setGameOver(false);
    setWinner(null);
    setIsThinking(false);
    setHoveredCol(null);
    setTime(0);
    setWinningCells([]);
  }

  // formats seconds into mm:ss
  function formatTime(secs) {
    const m = String(Math.floor(secs / 60)).padStart(2, '0');
    const s = String(secs % 60).padStart(2, '0');
    return `${m}:${s}`;
  }

  // figures out da status message to show at the top
  function getStatusMsg() {
    if (isThinking) return "AI is thinking...";
    if (gameOver) {
      if (winner === 1) return "you won!! yay";
      if (winner === -1) return "AI won :(";
      return "its a draw!";
    }
    return "your turn!";
  }

  // gets da css class for each cell based on whos piece it is
  function getCellClass(r, c, val) {
    const isWinner = winningCells.some(([wr, wc]) => wr === r && wc === c);
    if (val === 1) return `cell player${isWinner ? " winning" : ""}`;
    if (val === -1) return `cell ai${isWinner ? " winning" : ""}`;
    return "cell empty";
  }

  const winLineStyle = getWinLineStyle();

  return (
    <div className="App">
      <h1>Connect 4 vs AlphaZero</h1>
      <p className="status">{getStatusMsg()}</p>

      <div className="timer">{formatTime(time)}</div>

      <div className="gameArea">
        <div className="boardSection">
          {/* arrow indicators above each column */}
          <div className="arrowRow">
            {Array(COLS).fill(null).map((_, c) => (
              <div key={c} className="arrowSlot">
                {hoveredCol === c && !gameOver && !isThinking && (
                  <div className="arrow" />
                )}
              </div>
            ))}
          </div>

          {/* board wrapper so we can overlay stuff on top */}
          <div className="boardWrapper">
            <div className="board">
              {board.map((row, r) =>
                row.map((cell, c) => (
                  <div
                    key={`${r}-${c}`}
                    className={getCellClass(r, c, cell)}
                    onClick={() => handleColClick(c)}
                    onMouseEnter={() => setHoveredCol(c)}
                    onMouseLeave={() => setHoveredCol(null)}
                  />
                ))
              )}
            </div>

            {/* winning line drawn over the board */}
            {winLineStyle && <div style={winLineStyle} />}

            {/* game over overlay */}
            {gameOver && (
              <div className="gameOverOverlay">
                <div className="gameOverBox">
                  <span className="gameOverText">GAME OVER</span>
                  <span className="gameOverSub">
                    {winner === 1 ? "you won!! 🎉" : winner === -1 ? "AI won :(" : "its a draw!"}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* turn indicator panel on the right */}
        <div className="turnPanel">
          <div className="turnCard">
            <span className="turnLabel">
              {isThinking ? "ALPHAZERO" : "PLAYER"} TURN
            </span>
            <div className={`turnDot ${isThinking ? "ai" : "player"}`} />
          </div>
        </div>
      </div>

      <button className="resetBtn" onClick={resetGame}>reset</button>
    </div>
  );
}

export default App;
