import streamlit as st
from mcts_demo import MCTSr

st.title("Monte Carlo Tree Search (MCTS) Self-Refined Demo")

question = st.text_input("Enter your question:", "If I hang 5 shirts outside and it takes them 5 hours to dry, how long would it take to dry 30 shirts?")
seed_answers = ["I Don't Know", "I can't understand this question.",
                "I can't help with this question.", "I don't know how to solve this question.",
                "I don't know the answer to this question.", "I don't know the answer to this question, sorry."]
max_iterations = st.slider("Max Iterations:", 1, 10, 2)
rollouts = st.slider("Rollouts:", 1, 8, 1)

if st.button("Run MCTS"):
    if question:    
        def streamlit_display_callback(node, num_child):
            return {
                f"Child {num_child}": {
                    "Critique": node.critique,
                    "Answer": node.answer
                }
            }
        
        def print_to_screen(children, parent_answer, iteration):
            st.write(f"Iteration {iteration}/{max_iterations}")
            st.write(f"Parent Node Answer: {parent_answer}")
            
            for child_key, child_data in children.items():
                st.write(f"{child_key}")
                with st.expander("Critique"):
                    st.write(child_data["Critique"])
                
                with st.expander("Improved Answer"):
                    st.write(child_data["Answer"])
        
        mcts = MCTSr(question, seed_answers, max_iterations, rollouts)
        best_answer = mcts.search(streamlit_display_callback, print_to_screen)
        
        st.write("Best Answer:")
        st.write(best_answer)
    else:
        st.warning("Please enter both a question and seed answers.")
