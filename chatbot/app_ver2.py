import streamlit as st
from ver2 import MCTSr
import os
from typing import List, Tuple, Optional
from dotenv import load_dotenv


class MCTSrWithLogging(MCTSr):
    def __init__(self, question, seed_answers, max_iterations=5, rollouts=1):
        super().__init__(question, seed_answers, max_iterations, rollouts)
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def expand(self, node):
        children_details = []
        for child in super().expand(node):
            child_detail = {
                "node": child,
                "critique": child.critique,
                "improved_answer": child.answer,
                "uct_score": child.uct_value
            }
            children_details.append(child_detail)
        self.log({
            "type": "expansion",
            "node": node.question,
            "children": children_details
        })
        return children_details

    def simulate(self, nodes: List[dict]) -> List[float]:
        rewards = []
        for i, node_detail in enumerate(nodes):
            node = node_detail["node"]
            total_reward = 0.0
            for _ in range(self.rollouts):
                rating = self.rate_answer(self.question, node.answer)
                total_reward += rating
                node.rewards.append(rating)
            avg_reward = total_reward / self.rollouts
            rewards.append(avg_reward)
        return rewards

    def backpropagate(self, node_detail: dict, reward: float):
        node = node_detail["node"]
        while node is not None:
            node.visits += 1
            node.rewards.append(reward)
            node.q_value = 0.5 * (min(node.rewards) + (sum(node.rewards) / len(node.rewards)))
            node = node.parent

    def search(self):
        for i in range(self.max_iterations):
            self.log({"type": "iteration", "iteration": i + 1})
            # Selection
            node = self.select(self.root)
            self.log({"type": "selection", "node": node.question})
            
            # Expansion
            if not node.is_fully_expanded():
                new_children = self.expand(node)
                rewards = self.simulate(new_children)
                for child, reward in zip(new_children, rewards):
                    self.backpropagate(child, reward)
            else:
                reward = self.simulate([{"node": node}])[0]
                self.backpropagate({"node": node}, reward)

            # UCT Update
            self.update_uct(self.root)

        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.refined_answer

def display_logs(logs):
    for log in logs:
        if log["type"] == "expansion":
            st.markdown(f"#### Selection\n- **Select Node {log['node'][:10]}...**: {log['node']}")
            for i, child in enumerate(log["children"]):
                st.write(f"**Child {i+1}**")
                with st.expander("**Critique:**"):
                    st.write(child["critique"])
                
                with st.expander("**Improved Answer:**"):
                    st.write(child["improved_answer"])

                st.write(f"**UCT Score:** {child['uct_score']}")

def main():
    load_dotenv()
    st.title("Monte Carlo Tree Search (MCTS) Self-Refined Demo")
    
    question = st.text_input("Enter your question:", "Completely factor the expression: $$x^8-256$$")
    correct_answer = st.text_input("Enter the correct answer (for comparison):", "(x^4+16)(x^2+4)(x+2)(x-2)")
    seed_answers = ["I Don't Know", "I can't understand this question.",
                "I can't help with this question.", "I don't know how to solve this question.",
                "I don't know the answer to this question.", "I don't know the answer to this question, sorry."]
    max_iterations = st.slider("Max Iterations:", 1, 10, 2)
    rollouts = st.slider("Rollouts:", 1, 8, 1)
    if st.button("Run MCTS"):
        mcts = MCTSrWithLogging(question, seed_answers, max_iterations = max_iterations, rollouts = rollouts)
        best_answer = mcts.search()
        
        st.subheader("Results")
        st.markdown(f"**Question:** {question}")
        st.markdown(f"**Correct answer:** {correct_answer}")
        st.markdown(f"**MCTS Final Answer:** {best_answer}")
        
        st.subheader("Detailed Logs")
        display_logs(mcts.logs)

if __name__ == "__main__":
    main()
