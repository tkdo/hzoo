from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI, Request
from fastapi import BackgroundTasks
import uvicorn

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
 
async def parse_param(request:Request):
    param = {}
    if request.method == "GET":
        param = request.query_params
    if request.method == "POST":
        param = await request.json()
    return dict(param)

async def home(background_tasks: BackgroundTasks, request:Request):
    result  = {"code":1, "message":"success"}
    try:
        param = await parse_param(request)
        query = param.get("query")
        result["data"] = root_func(query)
    except Exception as e:
        result["message"] = str(e)
    return result

app = FastAPI()
app.add_api_route(path="/", endpoint=home, methods=["POST", "GET"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

