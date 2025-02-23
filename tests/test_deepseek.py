from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型到 CPU
model = AutoModelForCausalLM.from_pretrained(
    "/Users/ss/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    device_map="cpu",  # 强制 CPU 加载
    torch_dtype=torch.float32  # 明确指定精度
)

tokenizer = AutoTokenizer.from_pretrained("/Users/ss/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# 生成文本
input_text = "中国的首都是"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


"""
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.3 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/Users/ss/x.py", line 3, in <module>
    from modelscope import snapshot_download
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/modelscope/__init__.py", line 4, in <module>
    from modelscope.utils.import_utils import (LazyImportModule,
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/modelscope/utils/import_utils.py", line 18, in <module>
    from modelscope.utils.ast_utils import (INDEX_KEY, MODULE_KEY, REQUIREMENT_KEY,
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/modelscope/utils/ast_utils.py", line 24, in <module>
    from modelscope.utils.registry import default_group
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/modelscope/utils/registry.py", line 11, in <module>
    logger = get_logger()
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/modelscope/utils/logger.py", line 48, in get_logger
    from modelscope.utils.torch_utils import is_dist, is_master
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/modelscope/utils/torch_utils.py", line 13, in <module>
    import torch
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/torch/__init__.py", line 1477, in <module>
    from .functional import *  # noqa: F403
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/torch/functional.py", line 9, in <module>
    import torch.nn.functional as F
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/torch/nn/__init__.py", line 1, in <module>
    from .modules import *  # noqa: F403
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/torch/nn/modules/__init__.py", line 35, in <module>
    from .transformer import TransformerEncoder, TransformerDecoder, \
  File "/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/torch/nn/modules/transformer.py", line 20, in <module>
    device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
/Users/ss/Pangu/EnvHome/anaconda3/envs/py310-dseek/lib/python3.10/site-packages/torch/nn/modules/transformer.py:20: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_numpy.cpp:84.)
  device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
Downloading Model to directory: /Users/ss/.cache/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
2025-02-23 16:45:48,772 - modelscope - INFO - Target directory already exists, skipping creation.



推荐解决方案
方案一：快速降级NumPy（推荐）
bash
Copy Code
# 在conda环境中执行
conda install "numpy<2" -c conda-forge --force-reinstall
方案二：更新依赖生态链（2025年兼容方案）
bash
Copy Code
# 安装支持NumPy 2.x的PyTorch版本
conda install -c pytorch "pytorch>=2.4" "numpy>=2" 

# 更新transformers至量子计算兼容版
pip install "transformers>=5.3" --upgrade --force-reinstall
方案三：构建隔离环境
bash
Copy Code
# 创建新版虚拟环境
conda create -n np2_env python=3.12 "numpy>=2" 
conda activate np2_env

# 安装2025年适配的AI框架
conda install -c pytorch "pytorch>=2.4" "torchvision>=0.16"
pip install "transformers>=5.3" "accelerate>=0.28"


"""