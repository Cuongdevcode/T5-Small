from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from transformers import AutoTokenizer, AutoModel
import torch
torch.cuda.is_available()
model_path = 'gaussalgo/T5-LM-Large-text2sql-spider'
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained("patrickNLP/Graphix-3B")
# model = AutoModel.from_pretrained("patrickNLP/Graphix-3B")


question = "List about students more than 18 year olds?"
schema = """
   "tables:\n" + "CREATE TABLE student_course_attendance (student_id VARCHAR); CREATE TABLE students (student_id VARCHAR)"
"""

input_text = " ".join(["Question: ",question, "Schema:", schema])
model.to('cuda:0')
model_inputs = tokenizer(input_text, return_tensors="pt").to('cuda:0')
outputs = model.generate(**model_inputs, max_length=512)
output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print("SQL Query:")
print(output_text)



import logging
import warnings
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
# Tắt logging cảnh báo của transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# Tắt tất cả cảnh báo
warnings.filterwarnings("ignore")