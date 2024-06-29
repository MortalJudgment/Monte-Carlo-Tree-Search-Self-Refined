# MCTSr: Monte Carlo Tree Search for Self-Refined Question Answering 

- This project is trying to implement a Monte Carlo Tree Search (MCTS) algorithm for question answering, where the system iteratively refines its own answers through self-critique and improvement.
- Based paper [Accessing GPT-4 level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report](https://arxiv.org/pdf/2406.07394).

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Methodology](#methodology)
- [Future Work](#future-work)
- [Contributing](#contributing)

## Project Overview

The MCTSr system aims to improve the accuracy and quality of question answering by incorporating a self-refinement loop. It utilizes the power of Large Language Models (LLMs) not only to generate answers but also to critically evaluate and refine them. 

## Getting Started

### Prerequisites

- Python 3.7+
- A Groq API Key (obtain from [https://groq.com/](https://groq.com/))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/mctsr.git
   cd montecarlo
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Set Environment Variables:**
- Create a .env file in the root directory.
- Add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```
### Usage
The primary way to interact with the MCTSr system is through the Streamlit application:
1. Run the Streamlit app:
  ```
  streamlit run chatbot/app_ver2.py
  ```
2. Use code with caution.
- Input your question and optional correct answer.
- Adjust the parameters:
  - Max Iterations: Controls the number of search iterations.
  - Rollouts: Determines how many simulations are run for each node expansion.
- Click "Run MCTS".
3. The app will display the MCTS search process, including:
- Node Selection
- Expansion with critiques and improved answers.
- Final Answer
- Detailed Logs: Step-by-step breakdown of the search process.

### File Structure
- `.env`: Stores environment variables (API keys).
- `requirement.txt`: Lists project dependencies.
- `monte carlo/`: Contains the core MCTS implementation.
- `monte_carlo.py` or `viet_ver.py`: Implements the MCTS algorithm, node structure, and interactions with the LLM.
- `chatbot/`: Contains the Streamlit app and supporting files.
- `mcts_demo.py`: Demonstrates using the MCTS class.
- `app.py`: Initial Streamlit application (version 1).
- **On process**
  - `app_ver2.py`: Improved Streamlit application with logging (version 2). 
  - `ver2.py`: Contains the core logic for MCTS with logging.

### Methodology
1. **Node Representation:** Each node in the MCTS tree represents a potential answer to the question.
2. **Selection:** The algorithm traverses the tree, selecting the most promising nodes based on their Upper Confidence Bound 1 (UCB1) values, balancing exploration and exploitation.
3. **Expansion:** When a node is selected for expansion, the system generates critiques of the current answer and uses them to create improved answer candidates, which become child nodes.
4. **Simulation:** Each child node undergoes multiple simulations where its answer is evaluated based on various criteria, and a reward score is assigned.
5. **Backpropagation:** The simulation results (rewards) are propagated back up the tree, updating the visit counts and Q-values of parent nodes.
6. **Final Answer Selection:** After a set number of iterations, the algorithm selects the most visited node's answer as the final, refined answer.

### Future Work
1. **Experiment with Different LLMs:** Evaluate performance with various LLMs for different aspects (answer generation, critique, rating).
2. **Implement Different Search Strategies:** Explore variations of MCTS, such as using different selection policies.
3. **Incorporate User Feedback:** Allow users to provide feedback on generated answers to further enhance the refinement process.
4. **Knowledge Base Integration:** Integrate external knowledge bases to improve answer accuracy and provide more comprehensive responses.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.
