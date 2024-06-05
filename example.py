
from inference import generate_response, naive_prompt
from transformers import AutoTokenizer , AutoModelForCausalLM


question = "What is the sum of 2 and 3?"
prompt = naive_prompt(question)
answer = generate_response(prompt)


print(answer)