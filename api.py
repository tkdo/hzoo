from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

class ReqParam(BaseModel):
    query: str

# 加载模型到 CPU
model = AutoModelForCausalLM.from_pretrained(
    "/Users/ss/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device_map="cpu",  # 强制 CPU 加载
    torch_dtype=torch.float32  # 明确指定精度
)

tokenizer = AutoTokenizer.from_pretrained("/Users/ss/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")


def root_func(query):
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7)
    result =  tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)
    return result
 

@app.post("/")
async def root_post(req_param: ReqParam):
    return root_func(req_param.query)
@app.get("/")
async def root_get(query:str = Query("")):
    return root_func(query)

