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
                console.print(f"\nĐiểm rating: {rating}", style="bold cyan")
                total_reward += rating
                node.rewards.append(rating)
            q_value = 0.5* (min(node.rewards) + (total_reward / len(node.rewards)))
            console.print(f"\nĐiểm reward: {q_value}", style="bold cyan")
            rewards.append(q_value)
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
                console.print(f"\nChọn Node: {node.refined_answer}", style="bold magenta")
            else:
                console.print(f"\nChọn Node: {node.answer}", style="bold magenta")
            
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
        console.print(f"Visits to most visited child: {best_child.visits}", style = "bright_green")
        return best_child.refined_answer

    def critique(self, question, draft_answer: str) -> str:
        prompt = (
            f"Câu hỏi: {question}\n"
            f"Bản nháp câu trả lời: {draft_answer}\n"
            "Vui lòng đưa ra một bài phê bình toàn diện về bản nháp câu trả lời, tập trung vào các lĩnh vực chính sau đây:\n\n"
            "1. Lập luận Logic:\n"
            " - Xác định bất kỳ lỗi nào trong các bước logic được thực hiện để đi đến câu trả lời.\n"
            " - Giải thích cách tiếp cận logic đúng để giải quyết vấn đề này.\n\n"
            "2. Hiểu biết về Toán học:\n"
            " - Đánh giá xem câu trả lời có thể hiện sự hiểu biết đúng đắn về các khái niệm toán học liên quan không.\n"
            " - Chỉ ra bất kỳ sự áp dụng sai lầm nào của các nguyên tắc toán học.\n\n"
            "3. Những Hiểu lầm Phổ biến:\n"
            " - Nêu bật bất kỳ hiểu lầm hoặc lỗi phổ biến nào mà câu trả lời có thể dựa trên.\n"
            " - Giải thích tại sao những hiểu lầm này là không chính xác và làm thế nào để tránh chúng.\n\n"
            "4. Diễn giải Vấn đề:\n"
            " - Đánh giá xem câu hỏi có được diễn giải chính xác không.\n"
            " - Làm rõ bất kỳ khía cạnh nào của câu hỏi có thể đã bị hiểu sai.\n\n"
            "5. Giả định Cơ bản:\n"
            " - Xác định bất kỳ giả định ngầm nào trong câu trả lời.\n"
            " - Thảo luận xem những giả định này có hợp lệ không và chúng ảnh hưởng như thế nào đến giải pháp.\n\n"
            "6. Ứng dụng Thực tế:\n"
            " - Đánh giá xem câu trả lời có phù hợp với các tình huống thực tế và cân nhắc thực tiễn không.\n"
            " - Giải thích cách các yếu tố thực tế có thể ảnh hưởng đến giải pháp.\n\n"
            "7. Cách Tiếp cận Thay thế:\n"
            " - Đề xuất các phương pháp thay thế để giải quyết vấn đề.\n"
            " - So sánh các phương pháp này với phương pháp được sử dụng trong bản nháp câu trả lời.\n\n"
            "8. Xác minh và Kiểm tra:\n"
            " - Đề xuất cách để xác minh tính chính xác của câu trả lời.\n"
            " - Đề xuất các trường hợp thử nghiệm hoặc kịch bản để xác nhận giải pháp.\n\n"
            "9. Rõ ràng và Giao tiếp:\n"
            "   - Đánh giá mức độ giải thích và truyền đạt câu trả lời.\n"
            "   - Đề xuất cải thiện trong cách trình bày giải pháp.\n\n"
            "10. Tổng quát hóa và Mở rộng:\n"
            "    - Thảo luận về cách giải pháp có thể áp dụng cho các vấn đề tương tự hoặc được tổng quát hóa.\n"
            "    - Khám phá bất kỳ phần mở rộng thú vị hoặc câu hỏi liên quan nào.\n\n"
            "Vui lòng đưa ra phản hồi cụ thể, có thể thực hiện được cho mỗi điểm. Bài phê bình của bạn nên giúp hiểu tại sao câu trả lời có thể không chính xác và cách tiếp cận đúng đắn các vấn đề tương tự trong tương lai.\n"
            "Lưu ý: Tập trung vào việc phê bình và đưa ra các đề xuất cải thiện, không đưa ra câu trả lời đúng trực tiếp."
            "Làm việc theo từng bước để đạt được kết quả tốt nhất."
        )
        critique = LLM_teacher(prompt)
        return critique

    def improve_answer(self, question, draft_answer: str, critique: str) -> str:
        prompt = (
            f"Nhiệm vụ: Cải thiện bản nháp câu trả lời dựa trên phê bình đã cho, đồng thời đảm bảo sự liên quan đến câu hỏi ban đầu.\n\n"
            f"Câu hỏi gốc: {question}\n"
            f"Bản nháp câu trả lời: {draft_answer}\n"
            f"Phê bình: {critique}\n\n"
            "Vui lòng cung cấp một câu trả lời cải thiện theo định dạng sau:\n\n"
            "1. Quá trình Lập luận:\n"
            " - Phân tích bài phê bình và xác định các lĩnh vực cần cải thiện\n"
            " - Cung cấp giải thích từng bước về cách bạn sẽ giải quyết từng điểm\n"
            " - Đảm bảo lập luận của bạn liên quan trực tiếp đến câu hỏi ban đầu\n\n"
            "2. Xác minh:\n"
            " - Kiểm tra thông tin trong câu trả lời đã cải thiện\n"
            " - Trích dẫn nguồn đáng tin cậy nếu có thể\n"
            " - Đảm bảo tất cả các phát biểu đều chính xác và được hỗ trợ tốt\n\n"
            "3. Câu trả lời Cuối cùng:\n"
            " - Cung cấp một câu trả lời ngắn gọn, rõ ràng và đã được cải thiện\n"
            " - Kết hợp các hiểu biết từ quá trình lập luận và xác minh của bạn\n"
            " - Đảm bảo câu trả lời trực tiếp giải quyết câu hỏi ban đầu\n\n"
            "Hãy nhớ phải ngắn gọn, rõ ràng và trực tiếp giải quyết tất cả các điểm được nêu ra trong bài phê bình.\n"
            "Câu trả lời cải thiện của bạn nên chính xác hơn, toàn diện hơn và liên quan hơn đến câu hỏi ban đầu."
            "Làm việc theo từng bước để đạt được kết quả tốt nhất."
            )
        improved_answer = LLM_student(prompt)
        return improved_answer.strip()

    def rate_answer(self, question, answer: str) -> float:
        prompt = (
            f"Câu hỏi: {question}\n"
            f"Câu trả lời: {answer}\n"
            "Vui lòng phân tích câu trả lời dựa trên các tiêu chí sau:\n"
            "1. Độ chính xác của thông tin\n"
            "2. Tính đầy đủ của câu trả lời\n"
            "3. Sự liên quan đến câu hỏi\n"
            "4. Tính rõ ràng và mạch lạc\n"
            "5. Sử dụng bằng chứng hoặc ví dụ (nếu có)\n\n"
            "Đưa ra một bài phê bình chi tiết đề cập đến từng khía cạnh này. Hãy cực kỳ nghiêm khắc trong đánh giá của bạn, "
            "nêu bật bất kỳ thiếu sót nào, dù nhỏ đến đâu. Hãy cụ thể về điểm mạnh nhưng tập trung nhiều hơn vào điểm yếu, "
            "và chỉ ra bất kỳ lỗi thực tế hoặc hiểu lầm nào. CHÚ Ý, TUYỆT ĐỐI KHÔNG đề xuất cải tiến hoặc đưa ra câu trả lời đã được sửa chữa.\n\n"
            "Sau bài phê bình của bạn, hãy đánh giá câu trả lời trên thang điểm từ 0 đến 100. Việc chấm điểm nên phản ánh một tiêu chuẩn nghiêm ngặt, trong đó:\n"
            "0-20: Rất kém (có lỗi lớn, phần lớn không chính xác hoặc không liên quan)\n"
            "21-40: Kém (có vấn đề đáng kể, nhiều khía cạnh chỉ đúng một phần hoặc không chính xác)\n"
            "41-60: Dưới trung bình (có lỗi đáng chú ý, nhiều hơn một vài khía cạnh không hoàn toàn chính xác hoặc liên quan)\n"
            "61-80: Trung bình (có vấn đề nhỏ, nhìn chung liên quan và chính xác, nhưng thiếu chi tiết hoặc độ chính xác)\n"
            "81-90: Tốt (có vấn đề nhỏ, cấu trúc tốt và hầu hết đều chính xác, nhưng chưa xuất sắc)\n"
            "91-99: Rất tốt (có vấn đề tối thiểu, rất gần với lý tưởng, nhưng chưa hoàn hảo)\n"
            "100: Dành cho sự hoàn hảo (cực kỳ hiếm, không có lỗi nào có thể nhận thấy)\n\n"
            "Câu trả lời của bạn nên theo định dạng này:\n"
            "Phê bình: <phê bình chi tiết>\n"
            "Đánh giá: <điểm số>"
        )
        rating_response = LLM_student(prompt)
        # Extract the rating 
        try:
            rating_response = self.refined_rating(rating_response)
            match = re.search(r"(\d+)", rating_response)
            if match:
                rating = int(match.group(1))
                if rating > 95:
                    rating = 95
                rating = float(rating)/100
            else:
                raise ValueError("Rating not found in the response") 
        except Exception as e:
            print(f"Error extracting rating: {e}")
            print(f"Số điểm được chấm: {rating_response}")
            rating = 0.0
        return rating
    
    def refined_answer(self, answer: str) -> str:
        prompt = (
            f"Câu trả lời đã cho: {answer}\n\n"
            "Nhiệm vụ của bạn là trích xuất câu trả lời cuối cùng từ câu trả lời đã cho.\n"
            "Vui lòng chỉ cung cấp câu trả lời cuối cùng, bắt đầu bằng 'Câu trả lời cuối cùng:' và bao gồm tất cả văn bản theo sau nó.\n"
            "Nếu bạn phát hiện không có 'Câu trả lời cuối cùng:', hãy trả về câu trả lời gốc.\n"
            "Đảm bảo rằng phản hồi giữ nguyên cách diễn đạt ban đầu và không thêm, xóa hoặc thay thế bất kỳ thông tin nào từ câu trả lời đã cho.\n"
            "Không bắt đầu bằng 'Đây là câu trả lời đã cải thiện' hoặc một số văn bản bắt đầu tương đương.\n"
            "Phản hồi của bạn nên theo định dạng này:\n"
            "<câu-trả-lời-cuối-cùng>\n"
        )
        refined_answer = LLM_student(prompt)
        return refined_answer
    
    def refined_rating(self, answer: str) -> str:
        prompt = (
            f"Câu trả lời đã cho: {answer}\n\n"
            "Nhiệm vụ của bạn là trích xuất điểm được đánh giá cho câu hỏi này.\n"
            "Vui lòng chỉ cung cấp điểm số lấy được từ câu trả lời, không cung cấp thông tin gì thêm"
            "Ví dụ, nếu điểm là 80/100 thì câu trả lời của bạn là 80"
            "<điểm đánh giá>\n"
        )
        refined_answer = LLM_student(prompt)
        return refined_answer
    
def LLM_student(prompt: str, model_name: str = "gemma-7b-it", temperature: float = 0.3, max_tokens: int = 16384) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Give a response in Vietnamse"},
            {"role": "user", "content": prompt},
        ],
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return chat_completion.choices[0].message.content

def LLM_teacher(prompt: str, model_name: str = "gemma-7b-it", temperature: float = 0.3, max_tokens: int = 16384) -> str:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Give a response in Vietnamse"},
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
    question = "Phân tích đa thức sau thành nhân tử: $$x^8-256$$"
    correct_answer = "(x^4+16)(x^2+4)(x+2)(x-2)"
    dummy_answers = ["Tôi không biết", "Tôi không hiểu câu hỏi này.",
                      "Tôi không thể giúp với câu hỏi này.", "Tôi không biết cách để có thể trả lời câu hỏi này.",
                      "Tôi không biết câu trả lời cho câu hỏi này.", "Tôi không biết câu trả lời cho câu hỏi này, thành thật xin lỗi."]


    mcts = MCTSr(question, dummy_answers, max_iterations=2, rollouts=1)
    best_answer = mcts.search()
    text = Text()
    text.append("Câu hỏi: ", style="bold blue")
    text.append(question + "\n", style="blue")
    text.append("Câu trả lời đúng: \n", style="bold yellow")
    text.append(correct_answer + "\n", style="yellow")
    text.append("MCTSr Final Answer: \n", style="bold green")
    text.append(best_answer, style="green")  
    console.print(
        Panel(text)
    )

if __name__ == "__main__":
    main()
