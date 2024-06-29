import os
import re
import random
from groq import Groq
from typing import List, Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

import math
import numpy as np
from dotenv import load_dotenv

console = Console()
max_children = 3

class Node:
    def __init__(self, question: str, answer: List[str], refined_answer: str = None, parent: Optional['Node'] = None):
        self.question = question
        self.answer = answer
        self.refined_answer = refined_answer
        self.parent = parent
        self.children: List[Node] = []
        self.visits = 0
        self.rewards: List[float] = []
        self.q_value = 0.0
        self.uct_value = float('inf')

    def is_fully_expanded(self) -> bool:
        return len(self.children) >= max_children
    
    def best_child(self) -> 'Node':
        return max(self.children, key=lambda c: c.uct_value)

    def add_child(self, child_node: 'Node') -> 'Node':
        self.children.append(child_node)
    
    def most_visited_child(self):
        return max(self.children, key = lambda child: child.visits)
    
    def __str__(self):
        return f"Question: {self.question}, Answer: {self.refined_answer}, Visits: {self.visits}, Q_value: {self.q_value}"

class MCTSr:
    def __init__(self, question, seed_answers, max_iterations: int = 5, rollouts: int = 1):
        self.question = question
        self.seed_answer = seed_answers
        self.max_iterations = max_iterations
        self.rollouts = rollouts
        self.root = Node(question, random.choice(seed_answers))

    def select(self, node: Node) -> Node:
        while node.children:
            if not node.is_fully_expanded():
                return node
            node = node.best_child()
        return node

    def expand(self, node: Node) -> List[Node]:
        new_children = []
        for i in range(max_children - len(node.children)):
            child_node = Node(self.question, node.answer, parent=node)
            critique = self.critique(self.question, child_node.answer)
            console.print(
                Panel(
                    Markdown(critique),
                    title=f"[bold yellow]Critique {i+1}[/bold yellow]",
                    title_align="left",
                    border_style="yellow",
                )
            )
            improved_answer = self.improve_answer(self.question, child_node.answer, critique)
            console.print(
                Panel(
                    Markdown(improved_answer),
                    title=f"[bold blue]Improved Answer {i+1}[/bold blue]",
                    title_align="left",
                    border_style="blue",
                )
            )
            child_node.answer = improved_answer
            try:
                pattern = r'\*\*Final Answer:\*\*(.*?)$'
                match = re.search(pattern, improved_answer, re.DOTALL)
                if match:
                    refined_answer = match.group(1).strip()
                    child_node.refined_answer = refined_answer
                else: 
                    refined_answer = self.refined_answer(improved_answer)
                    child_node.refined_answer = refined_answer
            except Exception as e:
                print(f"Error extracting final answer: {e}")
            
            node.add_child(child_node)
            new_children.append(child_node)
        return new_children
    
    def simulate(self, nodes: List[Node]) -> List[float]:
        rewards = []
        for i, node in enumerate(nodes):
            total_reward = 0.0
            console.print(
                Panel(
                    Markdown(node.answer),
                    title=f"[bold blue]Simulate reward for child {i+1}[/bold blue]",
                    title_align="left",
                    border_style="blue",
                )
            )
            for _ in range(self.rollouts):
                rating = self.rate_answer(self.question, node.answer)
                console.print(f"\nRating: {rating}", style="bold cyan")
                total_reward += rating
                node.rewards.append(rating)
            avg_reward = total_reward / self.rollouts
            console.print(f"\nAverage reward {avg_reward}", style="bold cyan")
            rewards.append(avg_reward)
        return rewards
        
    def backpropagate(self, node: Node, reward: float):
        while node is not None:
            node.visits += 1
            node.rewards.append(reward)
            node.q_value = 0.5 * (min(node.rewards) + (sum(node.rewards) / len(node.rewards)))
            node = node.parent

    def update_uct(self, node: Node, exploration_weight: float = 1.41):
        for child in node.children:
            child.uct_value = child.q_value + exploration_weight * math.sqrt(math.log(node.visits) / (child.visits + 1e-10))

    def search(self):
        for i in range(self.max_iterations):
            console.print(f"\nIteration {i+1}/{self.max_iterations}", style="bold yellow")
            # Selection
            node = self.select(self.root)
            if node.refined_answer is not None:
                console.print(f"\nSelect Node: {node.refined_answer}", style="bold magenta")
            else:
                console.print(f"\nSelect Node: {node.answer}", style="bold magenta")
            
            # Expansion
            if not node.is_fully_expanded():
                new_children = self.expand(node)
                rewards = self.simulate(new_children)
                for child, reward in zip(new_children, rewards):
                    self.backpropagate(child, reward)
            else:
                reward = self.simulate([node])[0]
                self.backpropagate(node, reward)
            
            # UCT Update
            self.update_uct(self.root)

            
        best_child = max(self.root.children, key=lambda c: c.visits)
        print(f"Visits to most visited child: {best_child.visits}")
        return best_child.refined_answer

    def critique(self, question, draft_answer: str) -> str:
        prompt = (
            f"Question: {question}\n"
            f"Draft answer: {draft_answer}\n"
            "Please provide a comprehensive critique of the draft answer\n"
            "Please provide specific, actionable feedback for each point. Your critique should help in understanding why the answer might be incorrect and how to approach similar problems correctly in the future.\n"
            "Remember: Focus on critiquing and providing recommendations for improvement, not on giving the correct answer directly."
            "Let's work this out in a step by step way to be sure we have the right answer."
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
            "1. Reasoning Process: <step-by-step reasoning>\n"
            "2. Verification: <verify the given reasoning corresponds to the original question>\n"
            "3. Final Answer: <improved answer>\n"
            "Remember to be concise, clear, and directly address all points raised in the critique.\n"
            "Your improved answer should be more accurate, comprehensive, and relevant to the original question."
            "Let's work this out in a step by step way to be sure we have the right answer."
            )
        improved_answer = LLM_student(prompt)
        return improved_answer.strip()

    def rate_answer(self, question, answer: str) -> float:
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            "Please analyze the answer corresponds to the question\n"
            "Provide a detailed critique. Be extremely critical in your evaluation, "
            "highlighting any shortcomings, no matter how minor. Be specific about strengths but focus more on weaknesses, "
            "and point out any factual errors or misconceptions. Do not suggest improvements or provide a corrected answer.\n\n"
            "After your critique, rate the answer on a scale of 0 to 100.\n"
            "Your response should follow this format:\n"
            "Critique: <detailed critique>\n"
            "Rating: <rating>"
            )
        rating_response = LLM_teacher(prompt)
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
            {"role": "user", "content": prompt},
        ],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return chat_completion.choices[0].message.content


def main():
    load_dotenv()
    # question = "If I hang 5 shirts outside and it takes them 5 hours to dry, how long would it take to dry 30 shirts?"
    question = "Completely factor the expression: $$x^8-256$$"
    correct_answer = "(x^4+16)(x^2+4)(x+2)(x-2)"
    dummy_answers = ["I Don't Know", "I can't understand this question.",
                      "I can't help with this question.", "I don't know how to solve this question.",
                      "I don't know the answer to this question.", "I don't know the answer to this question, sorry."]


    mcts = MCTSr(question, dummy_answers, max_iterations=2, rollouts=1)
    best_answer = mcts.search()
    text = Text()
    text.append("Question: ", style="bold blue")
    text.append(question + "\n", style="blue")
    text.append("Correct answer: \n", style="bold yellow")
    text.append(correct_answer + "\n", style="yellow")
    text.append("MCTSr Final Answer: \n", style="bold green")
    text.append(best_answer, style="green")  
    console.print(
        Panel(text)
    )

if __name__ == "__main__":
    main()