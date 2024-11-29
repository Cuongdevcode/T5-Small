import logging
import warnings
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
# Tắt logging cảnh báo của transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# Tắt tất cả cảnh báo
warnings.filterwarnings("ignore")

# import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize the tokenizer from Hugging Face Transformers library
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained('cssupport/t5-small-awesome-text-to-sql')
model = model.to(device)
model.eval()

def generate_sql(input_prompt):
    # Tokenize the input prompt
    inputs = tokenizer(input_prompt, padding=True, truncation=True, return_tensors="pt").to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    
    # Decode the output IDs to a string (SQL query in this case)
    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_sql

# Test the function
# Request the user to enter a prompt for generating SQL
input_prompt = input("Enter a natural language query for SQL generation: ")

# Prepend the tables structure for context
tables_info = "tables:\nCREATE TABLE product (id LONG); CREATE TABLE product_details (id LONG PRIMARY KEY, title VARCHAR(200), price VARCHAR(11), rating VARCHAR(18), reviews VARCHAR(14), availability VARCHAR(40), aboutIt TEXT, description TEXT)"
input_prompt = f"{tables_info}\nquery for: {input_prompt}"

# Generate SQL
generated_sql = generate_sql(input_prompt)

# Output the generated SQL
print(f"The generated SQL query is: {generated_sql}")