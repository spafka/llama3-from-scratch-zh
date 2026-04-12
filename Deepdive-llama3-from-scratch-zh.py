#!/usr/bin/env python
# coding: utf-8
import os.path
# 如果在 Google Colab 运行此单元格，请安装所需的库
# ! pip install tiktoken blobfile


# 导入相关库
from pathlib import Path  # 用于从文件路径中获取文件名/模型名
import tiktoken  # openai开发的开源库，用于文本编解码（文本和tokenid的相互转换）
from IPython.core.debugger import prompt
from tiktoken.load import load_tiktoken_bpe  # 加载BPE模型
import torch  # 用于搭建模型和矩阵计算
import json  # 用于加载配置文件
import matplotlib.pyplot as plt  # 用于绘图
from modelscope.hub.snapshot_download import snapshot_download  # 使我们可以从 HuggingFace 下载权重



def calc_model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024 ** 3)

    return total_memory_gb


# 登录 HuggingFace，以便我们可以下载权重


# 选择模型在什么设备上进行加载和计算
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 如果有GPU则运行在GPU上，否则运行在CPU上
print(f"使用设备: {device}")
torch.set_default_device(device)

# 选择要下载和使用的模型

# 注意，默认推荐的模型为 Meta-Llama-3-8B
# 本项目中的模型架构、超参数等，均是基于该模型进行的分析，不同模型间可能会有一点差异是正常的

# 要从 HuggingFace 获取权重，您需要访问模型的 URL 并请求访问权限。
# 例如，对于模型 Meta-Llama-3-8B，您需要访问 https://huggingface.co/meta-llama/Meta-Llama-3-8B
# 并请求被授予访问权限。

# 选择任何版本或变体的 Llama。
# 您可以选择 3.0、3.1 或 3.2 版本。
# 3.2 版本提供 3B 或 1B 参数大小。
# 您还可以选择 "Instruct" 版本。

# "Instruct" 版本与普通版本有什么区别？
# "Vanilla"（普通版）是尚未对齐的 "基础模型"。
# "Instruct" 版本经过对齐，并且能够回答问题。
# 代码是相同的，唯一的区别是权重的不同，这对学习来说很有意义。

model_path = "Meta-Llama-3-8B"  # https://huggingface.co/meta-llama/Meta-Llama-3-8B

# model_path = "Llama-3.1-8B"                # https://huggingface.co/meta-llama/Llama-3.1-8B

# model_path = "Llama-3.2-3B"                # https://huggingface.co/meta-llama/Llama-3.2-3B
# model_path = "Llama-3.2-1B"                  # https://huggingface.co/meta-llama/Llama-3.2-1B

# model_path = "Meta-Llama-3-8B-Instruct"    # https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

# model_path = "Llama-3.1-8B-Instruct"       # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct

# model_path = "Llama-3.2-3B-Instruct"       # https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
# model_path = "Llama-3.2-1B-Instruct"       # https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct


# 下载模型文件

repo_id = f"LLM-Research/{model_path}"

files = [
    "original/consolidated.00.pth",  # PyTorch 格式的模型权重
    "original/params.json",  # 模型配置
    "original/tokenizer.model",  # 分词器模型
]

for file in files:
    if os.path.exists(f"./{model_path}/{file}"):
        continue
    file_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=file,
        local_dir=f"./{model_path}",
    )
    print(f"已下载 {file} 到 {file_path}")

# 此时，我们需要确保 model_path 目录中包含以下 3 个文件：
assert Path(f"{model_path}/original/tokenizer.model").exists()
assert Path(f'{model_path}/original/consolidated.00.pth').exists()
assert Path(f"{model_path}/original/params.json").exists()

# 加载基于BPE的tokenizer

tokenizer_path = f"{model_path}/original/tokenizer.model"  # 分词器模型的路径

# 常规词典外的特殊token
# 在"Meta-Llama-3-8B/"路径下的'tokenizer.json'和'tokenizer_config.json'的added_tokens字段下都有这些特殊token
special_tokens = [
                     "<|begin_of_text|>",
                     "<|end_of_text|>",
                     "<|reserved_special_token_0|>",  # 保留了从0到250的特殊token
                     "<|reserved_special_token_1|>",
                     "<|reserved_special_token_2|>",
                     "<|reserved_special_token_3|>",
                     "<|start_header_id|>",  # 头部信息的开始，用于标记包裹结构化数据的头部信息，如元数据
                     "<|end_header_id|>",  # 头部信息的结束
                     "<|reserved_special_token_4|>",
                     "<|eot_id|>",  # end of turn，多轮对话里标记当前轮次对话的结束
                 ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]

# 加载BPE模型（实际是一个字典）
# 一个字典，子词(bytes类型，用utf-8解码)-rank(id)对，128000词，不包含上面的256个特殊token（所以模型的总词典大小是128256）
# 其中rank值是从0递增的序列，用于决定子词单元合并的优先顺序，优先级越高的会优先合并，因此这里的名字是mergeable ranks而非BPE或字典等类似的名字
# 没把特殊token加到字典里应该是出于灵活性考虑，便于面对不同模型架构或任务有不同特殊token时添加特定的token，而且保持字典大小不变
mergeable_ranks = load_tiktoken_bpe(tokenizer_path)

# 创建一个文本编解码器对象
# 其中的pat_str大致分为三个类型：带缩写的单词 & 单词、中文片段、1-3位的数字 & 其他特殊字符
tokenizer = tiktoken.Encoding(
    name=Path(tokenizer_path).name,  # 编码器名称，便于调试和日志记录使用的不同的编码器
    pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    # 用于初步的粗分割文本为token序列的正则表达式
    mergeable_ranks=mergeable_ranks,  # 传入加载的BPE模型
    special_tokens={token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)},  # 添加特殊token-id对的字典
)

# 测试是否创建成功，即编解码器是否能正确运行
print(tokenizer.decode(tokenizer.encode("create tokenizer successed!")))

# 下面是一个案例测试，来测试pat_str粗分割和tokenizer细分割的效果和区别
# pat_str的正则只是提供了一个初步的分割，一些长句子或中文等不会分割，会在tokenizer中进一步基于BPE算法进行细化分割
import regex  # 由于pat_str中用到了Unicode的一些语法，如\p{L}，所以不能用re库

## 创建正则
pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
pattern = regex.compile(pat_str)

## 文本切分
text = "Hello world! It's a test. 这是一个测试. alongwords. a long words. 123 456 789."  # 测试文本
re_tokens = pattern.findall(text)  # 使用正则表达式分割字符串
merge_tokens_id = tokenizer.encode(text)  # 使用tokenizer分割字符串
merge_tokens = [tokenizer.decode([i]) for i in merge_tokens_id]  # 将tokenizer分割结果的id序列转换为实际的子词序列

## 结果输出
print("原始字符串:", text)
print("正则分割结果:", re_tokens)
print("tokenizer分割结果:", merge_tokens)
print("tokenizer分割结果id:", list(zip(merge_tokens, merge_tokens_id)))

## 从结果将会看到所有单词的前缀空格都被保留了下来，而非单独一个空格token或将其删除，有利于模型正确理解单词间的边界信息，如例子中的alongwords


_ = """
输出结果：

create tokenizer successed!
原始字符串: Hello world! It's a test. 这是一个测试. alongwords. a long words. 123 456 789.
正则分割结果: ['Hello', ' world', '!', ' It', "'s", ' a', ' test', '.', ' 这是一个测试', '.', ' alongwords', '.',
               ' a', ' long', ' words', '.', ' ', '123', ' ', '456', ' ', '789', '.']
tokenizer分割结果: ['Hello', ' world', '!', ' It', "'s", ' a', ' test', '.', ' 这', '是一个', '测试', '.',
                    ' along', 'words', '.', ' a', ' long', ' words', '.', ' ', '123', ' ', '456', ' ', '789', '.']
tokenizer分割结果id: [('Hello', 9906), (' world', 1917), ('!', 0), (' It', 1102), ("'s", 596), (' a', 264),
                      (' test', 1296), ('.', 13), (' 这', 122255), ('是一个', 122503), ('测试', 82805), ('.', 13),
                      (' along', 3235), ('words', 5880), ('.', 13), (' a', 264), (' long', 1317), (' words', 4339),
                      ('.', 13), (' ', 220), ('123', 4513), (' ', 220), ('456', 10961), (' ', 220), ('789', 16474), ('.', 13)]

"""

# 加载模型，一个网络层名称-tensor类型参数的字典
model = torch.load(f"{model_path}/original/consolidated.00.pth", weights_only=False, map_location=device)

#size = calc_model_memory_size(model, input_dtype=torch.bfloat16)
#print(size)
# 输出前20层网络名，验证是否正确加载
print(json.dumps(list(model.keys())[:20], indent=4))

# 加载配置文件，每个配置的具体含义见下节
with open(f"{model_path}/original/params.json", "r") as f:
    config = json.load(f)
config

# 记录这些配置，后面将逐渐用到
dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])

# 将输入的文本prompt转换为token id序列
prompt = "the answer to the ultimate question of life, the universe, and everything is "  # 输入文本
tokens = [128000] + tokenizer.encode(prompt)  # 做子词切分，并在文本开头加入指示文本开始了的特殊token：<|begin_of_text|>，维度：[17]
print(tokens)  # 查看下切分后的结果
tokens = torch.tensor(tokens)  # 转换为张量类型，便于后续矩阵计算，[17]

# 将token id序列转换为具体的token子词序列，仅作为展示用，实际不需要
prompt_split_as_tokens = [tokenizer.decode([token]) for token in tokens]
print(prompt_split_as_tokens)

# 创建一个嵌入层网络，用于将离散的token id映射到连续的向量空间
embedding_layer = torch.nn.Embedding(vocab_size, dim)

# 将嵌入层网络的参数替换为llama3中预训练好的参数值
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])

# 使用嵌入层网络，将输入的token id序列转换为向量表示
# 嵌入层网络仅是基于id查字典来找到对应的向量，不涉及token间的交互
# [17] -> [17x4096]
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)  # 默认是float32全精度，这里换成半精度格式，降低内存占用

token_embeddings_unnormalized.shape

# 展示第一层transformer块的所有权重参数及形状
for k, v in model.items():
    if not k.startswith('layers'):
        continue
    if k.startswith('layers.1'):
        break
    print(k, v.shape)


# 定义RMS归一化的计算函数
# 会将每个token进行独立的归一化
# norm_weights为预训练的缩放因子（即公式中的gi），以增强模型的表达能力。可以从模型文件中加载，4096维
# torch.rsqrt计算tensor的平方根的倒数，即1/RMS(a)
def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + norm_eps)) * norm_weights


# 归一化输入
token_embeddings = rms_norm(token_embeddings_unnormalized,
                            model["layers.0.attention_norm.weight"])  # [17x4096] & [4096] -> [17x4096]
model["layers.0.attention_norm.weight"].shape, token_embeddings.shape

# 展示当前的q,k,v和o的注意力权重矩阵形状
print(
    model["layers.0.attention.wq.weight"].shape,  # [4096x4096]
    model["layers.0.attention.wk.weight"].shape,  # [1024x4096]
    model["layers.0.attention.wv.weight"].shape,  # [1024x4096]
    model["layers.0.attention.wo.weight"].shape  # [4096x4096]
)

# 加载并修改layers.0的query权重矩阵的形状，使其以多头的形式展开
q_layer0 = model["layers.0.attention.wq.weight"]  # 默认形状为[4096x4096]
head_dim = q_layer0.shape[0] // n_heads  # 注意力头的维度，4096/32=128
q_layer0 = q_layer0.view(n_heads, head_dim, dim)  # 展开后的维度，[32x128x4096]
q_layer0.shape

# 取出第一个头的权重
q_layer0_head0 = q_layer0[0]  # [32x128x4096] -> [128x4096]
q_layer0_head0.shape

# 计算输入在第一个query头上得到的query值
# Q0_head0 = XW0_Q_head0.T
q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T)  # [17x4096] x [4096x128] = [17x128]
q_per_token.shape

# 加载并修改layers.0的key权重矩阵的形状，使其以多头的形式展开
# 与query权重矩阵不同，key为8个注意力头，因此参数量为query矩阵的1/4
k_layer0 = model["layers.0.attention.wk.weight"]  # [1024x4096]
k_layer0 = k_layer0.view(n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim)  # [8x128x4096]
k_layer0.shape

# 取出第一个头的权重矩阵
k_layer0_head0 = k_layer0[0]  # [8x128x4096] -> [128x4096]
k_layer0_head0.shape

# 计算第一个头的token嵌入对应的key向量
# K0_head0 = XW0_K_head0.T
k_per_token = torch.matmul(token_embeddings, k_layer0_head0.T)  # [17x4096] x [4096x128] = [17x128]
k_per_token.shape

# 加载并修改layers.0的value权重矩阵的形状，使其以多头的形式展开
# 与key权重矩阵一样，value同样为8个注意力头，因此参数量同样为query矩阵的1/4
v_layer0 = model["layers.0.attention.wv.weight"]  # [1024x4096]
v_layer0 = v_layer0.view(n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim)  # [1024x4096] -> [8x128x4096]
v_layer0.shape

# 取出第一个头的权重矩阵
v_layer0_head0 = v_layer0[0]  # [8x128x4096] -> [128x4096]
v_layer0_head0.shape

# 计算第一个头的token嵌入对应的value向量
# V0_head0 = XW0_V_head0.T
v_per_token = torch.matmul(token_embeddings, v_layer0_head0.T)  # [17x4096] x [4096x128] = [17x128]
v_per_token.shape

# 将query向量在维度方向上两两分组
# [17x128] -> [17x64x2]
q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)  # 换回了全精度，保证后续三角函数计算时的精度和数值稳定性
q_per_token_split_into_pairs.shape

# 计算θ，第一步，得到 i/D
# [64]
n_split = head_dim // 2
zero_to_one_split_into_64_parts = torch.tensor(range(n_split)) / n_split  # 每个特征分割后具有64个维度对，因此需要64个θ值
zero_to_one_split_into_64_parts

# 计算θ，第二步，得到 θ
# rope_theta用于控制位置编码的周期性等信息，详情可见配置信息章节
freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)  # [64]
freqs

# 计算mθ
# outer为外积计算，arange(17)即每个向量对应的m（输入为17个token，因此需要17个m值）
# 结果为[17x64]的形状，即每个token对应的向量都有64个mθ值，用于计算64个维度对各自的旋转
freqs_for_each_token = torch.outer(torch.arange(17), freqs)  # [17] & [64] -> [17x64]

# 得到(cosmθ + sinmθi)，即将mθ转换为复数形式
# 将旋转角度mθ看做是模长为1的极坐标形式，从而将其转换为复数表示
# polar的两个输入分别表示模长（设为1，即只改变角度，不影响长度）和角度（即mθ）
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)  # [17x64] -> [17x64]
print(freqs_cis.shape)

# 查看部分位置的freqs_cis，仅展示用
token_to_show = list(range(17))  # 查看第2,4,6行
fig, axs = plt.subplots(1, len(token_to_show), figsize=(5 * len(token_to_show), 4))  # 生成1行3列的3个子图图形窗口
for i, index in enumerate(token_to_show):
    value = freqs_cis[index]
    for j, element in enumerate(value):
        element = element.cpu()
        axs[i].plot([0, element.real], [0, element.imag], color='blue', linewidth=1,
                    label=f"Index: {j}")  # 以实部为横坐标，虚部为纵坐标，绘制从原点到坐标点的蓝线
        axs[i].annotate(f"{j}", xy=(element.real, element.imag), color='red')  # 绘制红色的数字注释，表示第i对维度
    axs[i].set_xlabel('Real')
    axs[i].set_ylabel('Imaginary')
    axs[i].set_title(f'Plot of {index + 1}th of freqs_cis')
plt.show()

"""
注：从展示图可以看出，token位置越靠后的旋转角度越大，而单个token内的向量维度越靠前的旋转角度越大。
    其中是否有数学上的考量可自行进一步探索X_X
"""

# 得到(x + yi)
# 即把维度对转换为复数，转换后的维度将从[17x64x2]变为[17x64]
q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)  # [17x64x2] -> [17x64]
q_per_token_as_complex_numbers.shape

# 计算(x + yi) * (cosmθ + sinmθi)
# 即执行旋转操作，得到最终结果
q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis  # [17x64] * [17x64] = [17x64]
q_per_token_as_complex_numbers_rotated.shape

# 将复数结果还原回实数的维度对形式
q_per_token_split_into_pairs_rotated = torch.view_as_real(
    q_per_token_as_complex_numbers_rotated)  # [17x64] -> [17x64x2]
q_per_token_split_into_pairs_rotated.shape

# 将维度对结果还原回原始query向量形式
q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)  # [17x64x2] -> [17x128]
q_per_token_rotated.shape

# 将key向量在维度方向上两两切分，形成维度对（修改形状）
k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)  # [17x128] -> [17x64x2]
k_per_token_split_into_pairs.shape

# 得到(x + yi)
# 即将维度对转换为复数形式
k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)  # [17x64x2] -> [17x64]
k_per_token_as_complex_numbers.shape

# 得到(x + yi) * (cosmθ + sinmθi)，即旋转位置编码后的最终结果
# 并将结果还原回实数形式
k_per_token_split_into_pairs_rotated = torch.view_as_real(
    k_per_token_as_complex_numbers * freqs_cis)  # [17x64] * [17x64] = [17x64] -> [17x64x2]
k_per_token_split_into_pairs_rotated.shape

# 将维度对还原回原始key向量形式，得到最终的key向量
k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)  # [17x64x2] -> [17x128]
k_per_token_rotated.shape

# 计算注意力分数
# 同时进行归一化，防止后续softmax计算结果过于偏向0或1（维度较大时的点积值可能过大），而导致梯度消失或梯度爆炸，以维持数值稳定性
qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (
    head_dim) ** 0.5  # [17x128] x [128x17] = [17x17]
qk_per_token.shape


# 首先查看一下屏蔽前的分数矩阵
def display_qk_heatmap(qk_per_token):
    _, ax = plt.subplots()  # 创建图形窗口

    # imshow通常用于显示二维数组或矩阵形式的数据，将矩阵元素映射为灰度或彩色值，因此可用于绘制热力图
    # 将张量转换回全精度，然后从计算图中分离出来（detach），避免涉及到潜在的梯度计算和存储问题
    # 并指定使用viridis颜色映射方案来显示图像（蓝->绿->黄）
    im = ax.imshow(qk_per_token.cpu().to(float).detach(), cmap='viridis')

    # 设定xy轴的刻度数量和标签，保证正确的一一对应
    ax.set_xticks(range(len(prompt_split_as_tokens)))
    ax.set_yticks(range(len(prompt_split_as_tokens)))
    ax.set_xticklabels(prompt_split_as_tokens)
    ax.set_yticklabels(prompt_split_as_tokens)

    # 添加侧方颜色条
    # 指定im以识别正确的颜色映射和值的范围
    # 指定所属的子图为ax（若有多个子图，则ax=ax[i]）
    ax.figure.colorbar(im, ax=ax)


display_qk_heatmap(qk_per_token)

# 生成屏蔽矩阵
# 让需要屏蔽的元素位置设置为负无穷，而不需要屏蔽的位置设为0，后续将其与分数矩阵相加即可实现屏蔽效果（负无穷在计算softmax时将趋于0）

# torch.full用于生成指定形状和指定填充值的张量，这里首先生成了全为负无穷的[17x17]矩阵
# 指定该矩阵位置与之前token的位置相同，以确保后续计算时不出错（例如：如果之前的token在gpu上，而这里不指定设备，则mask将新建在cpu上，二者相加时将会报错）
mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)  # [17x17]

# torch.triu用于返回矩阵的上三角部分，其余部分置为0（取下三角使用torch.tril）
# diagonal为对角线的偏移量，等于1时表示从主对角线向上偏移1个位置开始取上三角部分，以避免屏蔽自身token
mask = torch.triu(mask, diagonal=1)  # [17x17]

mask, mask.shape

# 屏蔽未来token的分数
qk_per_token_after_masking = qk_per_token + mask  # [17x17] + [17x17] = [17x17]
display_qk_heatmap(qk_per_token_after_masking)  # 展示屏蔽后的注意力分数

# 计算注意力权重
# 即计算分数的softmax值
# dim=1表示按行进行softmax的计算，结果转换为半精度，与之后的value向量保持一致
qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(
    torch.bfloat16)  # [17x17] -> [17x17]
display_qk_heatmap(qk_per_token_after_masking_after_softmax)

# 计算单头注意力的最终结果
qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)  # [17x17] x [17x128] = [17x128]
qkv_attention.shape

# 计算多头注意力结果
# 即之前的单头注意力计算过程的循环
qkv_attention_store = []

for head in range(n_heads):
    # 取出当前头对应的qkv权重矩阵
    q_layer0_head = q_layer0[head]  # [32x128x4096] -> [128x4096]
    k_layer0_head = k_layer0[head // (n_heads // n_kv_heads)]  # 每4个头共享一个key权重，[8x128x4096] -> [128x4096]
    v_layer0_head = v_layer0[head // (n_heads // n_kv_heads)]  # 每4个头共享一个value权重，[8x128x4096] -> [128x4096]

    # 计算XW，得到qkv向量
    # [17x4096] x [4096x128] = [17x128]
    q_per_token = torch.matmul(token_embeddings, q_layer0_head.T)
    k_per_token = torch.matmul(token_embeddings, k_layer0_head.T)
    v_per_token = torch.matmul(token_embeddings, v_layer0_head.T)

    # 给query向量加入位置信息（RoPE）
    q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1,
                                                            2)  # 维度方向进行两两分组，组成维度对，[17x128] -> [17x64x2]
    q_per_token_as_complex_numbers = torch.view_as_complex(
        q_per_token_split_into_pairs)  # 转换为复数表示，(x,y) -> (x+yi)，[17x64x2] -> [17x64]
    q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis[:len(
        tokens)]  # 计算(x+yi)*(cosmθ+sinmθi)，完成旋转操作，[17x64] * [17x64] = [17x64]
    q_per_token_split_into_pairs_rotated = torch.view_as_real(
        q_per_token_as_complex_numbers_rotated)  # 结果还原回实数表示，(x+yi) -> (x,y)，[17x64] -> [17x64x2]
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(
        q_per_token.shape)  # 结果还原回原始向量形状，得到最终的query向量，[17x64x2] -> [17x128]

    # 给key向量加入位置信息（RoPE）
    k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1,
                                                            2)  # 维度方向进行两两分组，组成维度对，[17x128] -> [17x64x2]
    k_per_token_as_complex_numbers = torch.view_as_complex(
        k_per_token_split_into_pairs)  # 转换为复数表示，(x,y) -> (x+yi)，[17x64x2] -> [17x64]
    k_per_token_as_complex_numbers_rotated = k_per_token_as_complex_numbers * freqs_cis[:len(
        tokens)]  # 计算(x+yi)*(cosmθ+sinmθi)，完成旋转操作，[17x64] * [17x64] = [17x64]
    k_per_token_split_into_pairs_rotated = torch.view_as_real(
        k_per_token_as_complex_numbers_rotated)  # 结果还原回实数表示，(x+yi) -> (x,y)，[17x64] -> [17x64x2]
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(
        k_per_token.shape)  # 结果还原回原始向量形状，得到最终的key向量，[17x64x2] -> [17x128]

    # 计算注意力分数，同时归一化分数（即QxK/sqrt(dim)）
    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (
        head_dim) ** 0.5  # [17x128] x [128x17] = [17x17]

    # 屏蔽未来token的分数
    mask = torch.full(qk_per_token.shape, float("-inf"),
                      device=tokens.device)  # 创建和注意力分数相同形状，值全为负无穷的矩阵，并保证存储位置和其他向量一致，防止后续计算时错误，[17x17]
    mask = torch.triu(mask,
                      diagonal=1)  # 保留上三角部分的负无穷，将下三角设为0（即上三角区域为未来token需要屏蔽，下三角为当前及以前的token无需屏蔽），对角线偏移量为1，避免屏蔽自身token，[17x17]
    qk_per_token_after_masking = qk_per_token + mask  # 注意力分数与屏蔽矩阵相加，使分数矩阵的上三角变为负无穷，后续softmax后将趋于0，[17x17]

    # 计算注意力权重（即softmax(score)）
    # 同时变换回半精度（因为后续要和value向量v_per_token相乘，需要保证数据类型相同）
    qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(
        torch.bfloat16)  # 按行计算softmax，[17x17]

    # 计算注意力机制的最终结果（即softmax(score) x v）
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)  # [17x17] x [17x128] = [17x128]

    # 记录该头结果
    qkv_attention_store.append(qkv_attention)

len(qkv_attention_store)

# 合并多头注意力矩阵
stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)  # 将第二维合并，即32个[17x128] -> [17x4096]
stacked_qkv_attention.shape

# 加载layers.0的output权重矩阵
w_layer0 = model["layers.0.attention.wo.weight"]  # [4096x4096]
w_layer0.shape

# 进行注意力矩阵的线性映射
# 这就是注意力层的最终输出结果了
embedding_delta = torch.matmul(stacked_qkv_attention, w_layer0.T)  # [17x4096] x [4096x4096] = [17x4096]
embedding_delta.shape

# 将attention层的输出与原始输入相加，完成残差操作
embedding_after_edit = token_embeddings_unnormalized + embedding_delta  # [17x4096] + [17x4096] = [17x4096]
embedding_after_edit.shape

# 将残差结果再进行一下归一化
embedding_after_edit_normalized = rms_norm(embedding_after_edit,
                                           model["layers.0.ffn_norm.weight"])  # [17x4096] & [4096] -> [17x4096]
embedding_after_edit_normalized.shape

# 计算前馈网络层
# 隐藏层维度大小为14336
w1 = model["layers.0.feed_forward.w1.weight"]  # [14336x4096]
w3 = model["layers.0.feed_forward.w3.weight"]  # [14336x4096]
w2 = model["layers.0.feed_forward.w2.weight"]  # [4096x14336]
print(w1.shape, w3.shape, w2.shape)

# output = (silu(XW1) * XW3)W2
# [17x4096] x [4096x14336] x [14336x4096] = [17x4096]
output_after_feedforward = torch.matmul(
    torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(
        embedding_after_edit_normalized, w3.T), w2.T)
output_after_feedforward.shape

# 将前馈层的输出与原始输入相加，完成残差操作
# 这就是一个transformer块的最终结果了
layer_0_embedding = embedding_after_edit + output_after_feedforward  # [17x4096] + [17x4096] = [17x4096]
layer_0_embedding.shape

# 现在开始完成全部的32层transformer块的计算！

# 输入token的嵌入向量作为初始值
final_embedding = token_embeddings_unnormalized  # [17x4096]

# 32层transformer块逐层计算
for layer in range(n_layers):
    #########################################################################################################################
    ###################################### 第一轮 归一化-特征变换-残差 ######################################################

    ########################### 第一次归一化 ###################################################

    # 第一次归一化
    layer_embedding_norm = rms_norm(final_embedding,
                                    model[f"layers.{layer}.attention_norm.weight"])  # [17x4096] & [4096] -> [17x4096]

    ########################### 第一次特征变换-多头自注意力 #####################################

    # 获取当前层注意力机制的qkv权重矩阵
    q_layer = model[f"layers.{layer}.attention.wq.weight"]  # [4096x4096]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)  # [32x128x4096]
    k_layer = model[f"layers.{layer}.attention.wk.weight"]  # [1024x4096]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)  # [8x128x4096]
    v_layer = model[f"layers.{layer}.attention.wv.weight"]  # [1024x4096]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)  # [8x128x4096]

    # 用于存储每个头的注意力机制计算结果
    qkv_attention_store = []

    # 计算每个头的注意力机制结果
    for head in range(n_heads):
        # 取出当前头的qkv权重矩阵
        q_layer_head = q_layer[head]  # [32x128x4096] -> [128x4096]
        k_layer_head = k_layer[head // (n_heads // n_kv_heads)]  # 每4个头共享一个key权重，[8x128x4096] -> [128x4096]
        v_layer_head = v_layer[head // (n_heads // n_kv_heads)]  # 每4个头共享一个value权重，[8x128x4096] -> [128x4096]

        # 计算XW，得到输入token的qkv向量
        # [17x4096] x [4096x128] = [17x128]
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)

        # 给query向量加入位置信息（RoPE）
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1,
                                                                2)  # 维度方向进行两两分组，组成维度对，[17x128] -> [17x64x2]
        q_per_token_as_complex_numbers = torch.view_as_complex(
            q_per_token_split_into_pairs)  # 转换为复数表示，(x,y) -> (x+yi)，[17x64x2] -> [17x64]
        q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis  # 计算(x+yi)*(cosmθ+sinmθi)，完成旋转操作，[17x64] * [17x64] = [17x64]
        q_per_token_split_into_pairs_rotated = torch.view_as_real(
            q_per_token_as_complex_numbers_rotated)  # 结果还原回实数表示，(x+yi) -> (x,y)，[17x64] -> [17x64x2]
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(
            q_per_token.shape)  # 结果还原回原始向量形状，得到最终的query向量，[17x64x2] -> [17x128]

        # 给key向量加入位置信息（RoPE）
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1,
                                                                2)  # 维度方向进行两两分组，组成维度对，[17x128] -> [17x64x2]
        k_per_token_as_complex_numbers = torch.view_as_complex(
            k_per_token_split_into_pairs)  # 转换为复数表示，(x,y) -> (x+yi)，[17x64x2] -> [17x64]
        k_per_token_as_complex_numbers_rotated = k_per_token_as_complex_numbers * freqs_cis  # 计算(x+yi)*(cosmθ+sinmθi)，完成旋转操作，[17x64] * [17x64] = [17x64]
        k_per_token_split_into_pairs_rotated = torch.view_as_real(
            k_per_token_as_complex_numbers_rotated)  # 结果还原回实数表示，(x+yi) -> (x,y)，[17x64] -> [17x64x2]
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(
            k_per_token.shape)  # 结果还原回原始向量形状，得到最终的key向量，[17x64x2] -> [17x128]

        # 计算注意力分数，同时归一化分数（即QxK/sqrt(dim)）
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (
            head_dim) ** 0.5  # [17x128] x [128x17] = [17x17]

        # 屏蔽未来token的分数
        mask = torch.full(qk_per_token.shape, float("-inf"),
                          device=qk_per_token.device)  # 创建和注意力分数相同形状，值全为负无穷的矩阵，并保证存储位置和其他向量一致，防止后续计算时错误，[17x17]
        mask = torch.triu(mask,
                          diagonal=1)  # 保留上三角部分的负无穷，将下三角设为0（即上三角区域为未来token需要屏蔽，下三角为当前及以前的token无需屏蔽），对角线偏移量为1，避免屏蔽自身token，[17x17]
        qk_per_token_after_masking = qk_per_token + mask  # 注意力分数与屏蔽矩阵相加，使分数矩阵的上三角变为负无穷，后续softmax后将趋于0，[17x17]

        # 计算注意力权重（即softmax(score)）
        # 同时变换回半精度（因为后续要和value向量v_per_token相乘，需要保证数据类型相同）
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(
            torch.bfloat16)  # 按行计算softmax，[17x17]

        # 计算注意力机制的最终结果（即softmax(score)xV）
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax,
                                     v_per_token)  # [17x17] x [17x128] = [17x128]

        # 记录该头结果
        qkv_attention_store.append(qkv_attention)

    # 合并多头注意力结果
    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)  # 将第二维合并，即32个[17x128] -> [17x4096]

    # 结果做线性映射，生成最终的多头自注意力机制结果
    o_layer = model[f"layers.{layer}.attention.wo.weight"]
    embedding_delta = torch.matmul(stacked_qkv_attention, o_layer.T)  # [17x4096] x [4096x4096] = [17x4096]

    ########################### 第一次残差连接 ################################################

    # 第一次残差
    # 将attention层的输出与原始输入相加，完成残差操作
    embedding_after_edit = final_embedding + embedding_delta  # [17x4096] + [17x4096] = [17x4096]

    #########################################################################################################################
    ###################################### 第二轮 归一化-特征变换-残差 ######################################################

    ########################### 第二次归一化 ###################################################

    # 第二次归一化
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[
        f"layers.{layer}.ffn_norm.weight"])  # [17x4096] & [4096] -> [17x4096]

    ########################### 第二次特征变换-前馈神经网络 #####################################

    # 加载前馈网络（SwiGLU）参数矩阵
    w1 = model[f"layers.{layer}.feed_forward.w1.weight"]  # [14336x4096]
    w3 = model[f"layers.{layer}.feed_forward.w3.weight"]  # [14336x4096]
    w2 = model[f"layers.{layer}.feed_forward.w2.weight"]  # [4096x14336]

    # 计算前馈网络结果（output = (silu(XW1) * XW3)W2）
    # [17x4096] x [4096x14336] x [14336x4096] = [17x4096]
    output_after_feedforward = torch.matmul(
        torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(
            embedding_after_edit_normalized, w3.T), w2.T)

    ########################### 第二次残差连接 ##################################################

    # 第二次残差，得到当前transformer块的最终输出结果
    # 将前馈层的输出与原始输入相加，完成残差操作
    final_embedding = embedding_after_edit + output_after_feedforward  # [17x4096] + [17x4096] = [17x4096]

# 进行整个模型中最后一次的归一化
final_embedding = rms_norm(final_embedding, model["norm.weight"])  # [17x4096] & [4096] -> [17x4096]
final_embedding.shape

# 最后一次的线性映射，将嵌入向量映射成词典维度大小，以作为对下一个token的预测
logits = torch.matmul(final_embedding[-1],
                      model["output.weight"].T)  # [17x4096] -> [4096] -> [4096] x [4096x128256] = [128256]
logits.shape

# 取出概率最大的那一维对应的id，即预测的下一个token的id
next_token = torch.argmax(logits, dim=-1)  # 获取最大值对应的下标，即预测的下一个token id，[128256] -> [1]
next_token

# 基于预测的id，还原回具体预测的值
tokenizer.decode([next_token.item()])

# 让我们先看一下top-k的预测结果
logits_sort, logits_idx = torch.sort(logits, dim=-1, descending=True)  # 对预测结果做排序，使最大可能性的预测token放前面，[128256]
[tokenizer.decode([i]) for i in logits_idx[:10]]  # 查看前10大概率的结果

# 让我们再试一下用其他token的嵌入向量做预测能得到什么
logits_all_token = torch.matmul(final_embedding,
                                model["output.weight"].T)  # 将嵌入向量映射成词典大小的维度，[17x4096] x [4096x128256] = [17x128256]
logits_all_token_sort, logits_all_token_idx = torch.sort(logits_all_token, dim=-1,
                                                         descending=True)  # 对预测结果做排序，使最大可能性的预测token放前面，[17x128256]

print('输入的token：', prompt_split_as_tokens)  # 展示输入的token，[17]

# 展示基于每个token的嵌入向量所做的下一个token预测的结果
for i in range(len(final_embedding)):
    print(f'基于第{i + 1}个token的嵌入向量的预测结果：',
          [tokenizer.decode([j]) for j in logits_all_token_idx[i][:10]])  # 输出前10大概率的结果

_ = """
可以看到的是，基于每个token进行预测时，其预测结果为当前token后的下一个token的可能结果，而不是整个完整输入的预测结果，因此在实际预测时，会只使用最后一个token的嵌入向量来做预测
"""

# 最后，我们看一下如果在计算attention时不屏蔽未来token，预测结果会变成什么
# 此时基于每个token的预测结果将变为如下所示
# 可以看到，由于未来token的可见性，基于每个token的嵌入向量都将更准确的预测出“对于它来说的”下一个token（有一点点“作弊”了）

_ = """
输入的token： ['<|begin_of_text|>', 'the', ' answer', ' to', ' the', ' ultimate', ' question', ' of', ' life', ',', ' the', ' universe', ',', ' and', ' everything', ' is', ' ']
基于第1个token的嵌入向量的预测结果： ['://', '.Forms', '_REF', ' Angeles', '.swing', '�', 'php', 'во', 'ysics', '�']
基于第2个token的嵌入向量的预测结果： [' answer', ' Hitch', ' universe', ' question', ' ultimate', ' meaning', ' hitch', ' Universe', ' Answer', ' reason']
基于第3个token的嵌入向量的预测结果： [' to', ' is', ',', ':', ' was', '\n', ' ', ' (', '\n\n', ' of']
基于第4个token的嵌入向量的预测结果： [' the', ' life', ' this', ' which', ' everything', ' that', ' how', ' why', ' ', ' all']
基于第5个token的嵌入向量的预测结果： [' ultimate', ' question', ' great', ' meaning', ' universe', ' Ultimate', ' everything', ' life', ' holy', ' greatest']
基于第6个token的嵌入向量的预测结果： [' question', ' answer', ' is', ' was', '\n', ' questions', ' mystery', '\n\n', ' what', ' Question']
基于第7个token的嵌入向量的预测结果： [' of', ' is', '\n', ',', ' about', ':', ' to', ' in', ' (', '<|end_of_text|>']
基于第8个token的嵌入向量的预测结果： [' life', ' existence', ' everything', ' Life', ' the', ' death', ' time', ' all', ' why', ' which']
基于第9个token的嵌入向量的预测结果： [',', ' is', ' the', '\n', ':', ' (', '...', ' and', ' ,', ' -']
基于第10个token的嵌入向量的预测结果： [' the', ' and', ' is', ' death', ' The', ' which', ' or', '\xa0', ' existence', ' don']
基于第11个token的嵌入向量的预测结果： [' universe', ' answer', ' cosmos', ' world', ' existence', ' Universe', ' everything', ' un', ' meaning', ' question']
基于第12个token的嵌入向量的预测结果： [',', ' and', ' is', ' &', '\n', ' ,', '.', '...', ' (', ' ']
基于第13个token的嵌入向量的预测结果： [' and', ' &', ' don', ' the', ' is', ' a', ' or', ' Douglas', '\xa0', '<|end_of_text|>']
基于第14个token的嵌入向量的预测结果： [' everything', ' dough', ' don', ' ever', ' deep', ' Douglas', ' the', ' every', ' all', ' death']
基于第15个token的嵌入向量的预测结果： ['\n', ' is', ',', '.', ' ', ' (', ':', '<|end_of_text|>', '\n\n', '.\n']
基于第16个token的嵌入向量的预测结果： [' ', '\n', ' forty', '...', ' "', '42', ' the', ':', '\xa0', ' to']
基于第17个token的嵌入向量的预测结果： ['42', '6', '4', '41', '1', '2', '3', '7', '5', '43']
"""
