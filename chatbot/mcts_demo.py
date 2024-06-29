import os
import re
import random
from groq import Groq
from typing import List, Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import math
import numpy as np
from dotenv import load_dotenv

console = Console()
max_children = 3

class Node:
    def __init__(self, question: str, answer: str, parent: Optional['Node'] = None):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.rewards: List[float] = []
        self.q_value = 0.0
        self.critique = None  # Initialize critique

    def is_fully_expanded(self) -> bool:
        return len(self.children) >= max_children
    
    def best_child(self, exploration_weight: float = 1.41) -> 'Node':
        node_scores = []
        for child in self.children:
            if child.visits == 0:
                uct_score = float('inf')
            else:
                uct_score = child.q_value + exploration_weight * math.sqrt((math.log(self.visits + 1)) / (child.visits + 1e-10))
            node_scores.append(uct_score)
        return self.children[np.argmax(node_scores)]

    def add_child(self, child_node: 'Node') -> 'Node':
        self.children.append(child_node)
    
    def most_visited_child(self):
        return max(self.children, key = lambda child: child.visits)
    
    def __str__(self):
        return f"Question: {self.question}, Answer: {self.answer}, Visits: {self.visits}, Q_value: {self.q_value}"

class MCTSr:
    def __init__(self, question, seed_answers, max_iterations: int = 5, rollouts: int = 1):
        self.question = question
        self.seed_answer = seed_answers
        self.max_iterations = max_iterations
        self.rollouts = rollouts
        self.root = Node(question, random.choice(seed_answers))

    def select(self, node: Node) -> Node:
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def expand(self, node: Node, streamlit_display_callback) -> Tuple[List[Node], dict, str]:
        children = {}
        for i in range(max_children - len(node.children)):
            child_node = Node(self.question, node.answer, parent=node)
            node.add_child(child_node)
            critique = self.critique(self.question, child_node.answer)
            child_node.critique = critique  # Store the critique
            improved_answer = self.improve_answer(self.question, child_node.answer, critique)
            child_node.answer = improved_answer
            # Update Streamlit display after each iteration
            children.update(streamlit_display_callback(child_node, i + 1))
        return random.choice(node.children), children, node.answer


    def simulate(self, node: Node) -> float:
        total_reward = 0.0
        for _ in range(self.rollouts):
            rating = self.rate_answer(self.question, node.answer)
            total_reward += rating
            node.rewards.append(rating)
        return total_reward / self.rollouts 
        
    def backpropagate(self, node: Node):
        while node is not None:
            node.visits += 1
            if node.rewards:
                node.q_value = 0.5 * (min(node.rewards) + (sum(node.rewards) / len(node.rewards)))
            node = node.parent

    def search(self, streamlit_display_callback, print_to_screen):
        for i in range(self.max_iterations):
            node = self.select(self.root)
            if not node.is_fully_expanded():
                children, list_children, parent = self.expand(node, streamlit_display_callback)
            self.simulate(node)
            try:
                final_answers = [self.refined_answer(child.answer) for child in children]
                for child, final_answer in zip(children, final_answers):
                    child.answer = final_answer
            except Exception as e:
                print(f"Error extracting rating: {e}")
            print(len(list_children))
            print_to_screen(list_children, parent, i+1)

        self.backpropagate(node)  # Backpropagate average reward
        
        return self.root.most_visited_child().answer

    def critique(self, question, draft_answer: str) -> str:
        prompt = (
            f"Question: {question}\n"
            f"Draft answer: {draft_answer}\n"
            "Please provide a comprehensive critique of the draft answer, focusing on the following key areas:\n\n"
            "1. Logical Reasoning:\n"
            "   - Identify any flaws in the logical steps taken to arrive at the answer.\n"
            "   - Explain the correct logical approach to solving this problem.\n\n"
            "2. Mathematical Understanding:\n"
            "   - Assess if the answer demonstrates proper understanding of relevant mathematical concepts.\n"
            "   - Point out any misapplications of mathematical principles.\n\n"
            "3. Common Misconceptions:\n"
            "   - Highlight any common misconceptions or errors that the answer might be based on.\n"
            "   - Explain why these misconceptions are incorrect and how to avoid them.\n\n"
            "4. Problem Interpretation:\n"
            "   - Evaluate if the question was interpreted correctly.\n"
            "   - Clarify any aspects of the question that might have been misunderstood.\n\n"
            "5. Underlying Assumptions:\n"
            "   - Identify any unstated assumptions in the answer.\n"
            "   - Discuss whether these assumptions are valid and how they affect the solution.\n\n"
            "6. Real-World Application:\n"
            "   - Assess if the answer aligns with real-world scenarios and practical considerations.\n"
            "   - Explain how real-world factors might influence the solution.\n\n"
            "7. Alternative Approaches:\n"
            "   - Suggest alternative methods to solve the problem.\n"
            "   - Compare these methods to the one used in the draft answer.\n\n"
            "8. Verification and Testing:\n"
            "   - Propose ways to verify the correctness of the answer.\n"
            "   - Suggest test cases or scenarios to validate the solution.\n\n"
            "9. Clarity and Communication:\n"
            "   - Evaluate how well the answer is explained and communicated.\n"
            "   - Suggest improvements in the presentation of the solution.\n\n"
            "10. Generalization and Extension:\n"
            "    - Discuss how the solution might apply to similar problems or be generalized.\n"
            "    - Explore any interesting extensions or related questions.\n\n"
            "Please provide specific, actionable feedback for each point. Your critique should help in understanding why the answer might be incorrect and how to approach similar problems correctly in the future.\n"
            "Remember: Focus on critiquing and providing recommendations for improvement, not on giving the correct answer directly."
        )
        critique = LLM_teacher(prompt)
        return critique

    def improve_answer(self, question, draft_answer: str, critique: str) -> str:
        prompt = (
            f"Task: Improve the draft answer based on the given critique while ensuring relevance to the original question.\n\n"
            f"Original Question: {question}\n"
            f"Draft Answer: {draft_answer}\n"
            f"Critique: {critique}\n\n"
            "Please provide an improved answer following this format:\n\n"
            "1. Reasoning Process:\n"
            "   - Analyze the critique and identify areas for improvement\n"
            "   - Provide a step-by-step explanation of how you'll address each point\n"
            "   - Ensure your reasoning directly relates to the original question\n\n"
            "2. Verification:\n"
            "   - Fact-check the information in the improved answer\n"
            "   - Cite reliable sources if applicable\n"
            "   - Ensure all statements are accurate and well-supported\n\n"
            "3. Final Answer:\n"
            "   - Provide a concise, clear, and improved answer\n"
            "   - Incorporate the insights from your reasoning and verification\n"
            "   - Ensure the answer directly addresses the original question\n\n"
            "Remember to be concise, clear, and directly address all points raised in the critique. Do not start with 'Here is the improved answer' or some equivalent start text.\n"
            "Your improved answer should be more accurate, comprehensive, and relevant to the original question."
            )
        improved_answer = LLM_student(prompt)
        return improved_answer.strip()

    def rate_answer(self, question, answer: str) -> float:
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            "Please analyze the answer based on the following criteria:\n"
            "1. Accuracy of information\n"
            "2. Completeness of the response\n"
            "3. Relevance to the question\n"
            "4. Clarity and coherence\n"
            "5. Use of evidence or examples (if applicable)\n\n"
            "Provide a detailed critique addressing each of these aspects. Be specific about strengths and weaknesses, "
            "and point out any factual errors or misconceptions. Do not suggest improvements or provide a corrected answer.\n\n"
            "After your critique, rate the answer on a scale of 0 to 100, where:\n"
            "0-20: Poor (major flaws, largely incorrect or irrelevant)\n"
            "21-40: Below Average (significant issues, partially relevant or correct)\n"
            "41-60: Average (some flaws, mostly relevant and partially correct)\n"
            "61-80: Good (minor issues, relevant and mostly correct)\n"
            "81-100: Excellent (minimal to no issues, highly relevant and correct)\n\n"
            "Your response should follow this format:\n"
            "Critique: <detail critique>\n"
            "Rating: <rating>"
            )
        rating_response = LLM_student(prompt)
        # Extract the rating 
        try:
            match = re.search(r"Rating:\s*(\d+)", rating_response)
            if match:
                rating = int(match.group(1))
                if rating > 95:
                    rating = 95
                rating = float(rating)/100
            else:
                raise ValueError("Rating not found in the response") 
        except Exception as e:
            print(f"Error extracting rating: {e}")
            print(f"Rating response was {rating_response}")
            rating = 0.0
        return rating
    
    def refined_answer(self, answer: str) -> str:
        prompt = (
            f"Given answer: {answer}\n\n"
            "Your job is to extract the final answer from the given answer.\n"
            "Please provide only the final answer, starting with 'Final Answer:' and including all text that follows it.\n"
            "If you find out there is no 'Final Answer:', return the origin answer.\n"
            "Ensure the response maintains the original wording and does not add or remove or replace any information from the given answer.\n"
            "Do not start with 'Here is the improved answer' or some equivalent start text."
            "Your response should follow this format:\n"
            "<final-answer>\n"
        )
        refined_answer = LLM_student(prompt)
        return refined_answer

    
def LLM_student(prompt: str, model_name: str = "llama3-8b-8192", temperature: float = 0.3, max_tokens: int = 16384) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return chat_completion.choices[0].message.content

def LLM_teacher(prompt: str, model_name: str = "llama3-8b-8192", temperature: float = 0.3, max_tokens: int = 16384) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return chat_completion.choices[0].message.content


def main():
    load_dotenv()
    question = "If I hang 5 shirts outside and it takes them 5 hours to dry, how long would it take to dry 30 shirts?"
    dummy_answers = ["I Don't Know", "I can't understand this question.",
                      "I can't help with this question.", "I don't know how to solve this question.",
                      "I don't know the answer to this question.", "I don't know the answer to this question, sorry."]


    mcts = MCTSr(question, dummy_answers, max_iterations=2, rollouts=4)
    best_answer = mcts.search()
    console.print(f"\nFinal Answer: {best_answer}", style="bold green")

if __name__ == "__main__":
    main()
