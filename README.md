# Rock Paper Scissors

This project implements a Rock Paper Scissors game in Python as part of the freeCodeCamp Machine Learning with Python certification. The goal is to create a `player` function in `RPS.py` that competes against four distinct bots (Quincy, Abbey, Kris, and Mrugesh) and achieves at least a 60% win rate against each in 1000-game matches. The strategy adapts to each bot's behavior by analyzing their move patterns and countering them effectively.

---

### Project Overview
The project processes the opponent's previous move and maintains a history to:
1. Use the `player` function in `RPS.py` to return a move ("R", "P", or "S") based on the opponent's last move (`prev_play`) and optional history (`opponent_history`).
2. Implement strategies to counter four bots:
   - **Quincy**: Repeats a fixed sequence ("R", "P", "P", "S", "R").
   - **Abbey**: Counters the player's most frequent moves.
   - **Kris**: Plays the counter to the player's previous move.
   - **Mrugesh**: Counters the player's most frequent move in the last 10 moves.
3. Achieve at least a 60% win rate against each bot by detecting patterns and adapting strategies.
4. Test the `player` function using `main.py` and verify performance with unit tests in `test_module.py`.

---
👨‍💻 **Author**: Ali Toori – Full-Stack Python Developer  
📺 **YouTube**: [@AliToori](https://youtube.com/@AliToori)  
💬 **Telegram**: [@AliToori](https://t.me/@AliToori)  
📂 **GitHub**: [github.com/AliToori](https://github.com/AliToori)

---

### [Replit Project Link](https://replit.com/@AliToori/RPS)

---

## 🛠 Tech Stack
* Language: Python 3.10+
* Libraries:
  * Standard Python libraries (`random` for random move selection)
  * Testing: `unittest` for automated tests
* Tools:
  * Replit for development and testing
  * GitHub for version control

---

## 📂 Project Structure
```bash
RockPaperScissors/
├── RPS.py                     # Core player function with adaptive strategies
├── main.py                    # Script to test the player against bots
├── test_module.py             # Unit tests to verify win rates
├── RPS_game.py                # Game logic and bot implementations (provided by freeCodeCamp)
└── README.md                  # Project documentation
```

---

## Usage
1. Clone the repository or use the Replit project:
   ```bash
   git clone https://github.com/AliToori/Rock-Paper-Scissors.git
   ```
2. Ensure all files (`RPS.py`, `main.py`, `test_module.py`, `RPS_game.py`) are in the same directory.
3. Run the main script to test the `player` function against each bot:
   ```bash
   python main.py
   ```
   - This runs 1000 games against each bot (Quincy, Abbey, Kris, Mrugesh) with verbose output to show game-by-game results.
   - To disable verbose output, modify `main.py` to set `verbose=False`.
4. Run unit tests to verify the win rate (uncomment `test()` in `main.py` or run directly):
   ```bash
   python test_module.py
   ```
   - Tests check if the `player` function achieves at least a 60% win rate against each bot.

---

## Contributing
Contributions are welcome! Please:
1. Fork the repository: https://github.com/AliToori/Rock-Paper-Scissors.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

**Note**: Do not modify `RPS_game.py`, as it contains the official game logic and bot implementations provided by freeCodeCamp.

---

## 🙏 Acknowledgments
- Built as part of the [freeCodeCamp Machine Learning with Python](https://www.freecodecamp.org/learn/machine-learning-with-python) certification.
- Uses standard Python libraries for strategy implementation and `unittest` for testing.
- Special thanks to freeCodeCamp for providing the challenge framework and bot implementations.

## 🆘 Support
For questions, issues, or feedback:

📺 YouTube: [@AliToori](https://youtube.com/@AliToori)  
💬 Telegram: [@AliToori](https://t.me/@AliToori)  
📂 GitHub: [github.com/AliToori](https://github.com/AliToori)