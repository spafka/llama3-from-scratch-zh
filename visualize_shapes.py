import torch
from llama3.model import Transformer, ModelArgs

def visualize_transformer_flow():
    # 使用极小参数快速初始化一个模型用于演示
    args = ModelArgs(
        dim=128,
        n_layers=1,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=1000,
        max_seq_len=64
    )
    
    device = "cpu"
    model = Transformer(args).to(device)
    
    # 模拟输入: Batch=1, Seq_Len=8
    tokens = torch.randint(0, 1000, (1, 8)).to(device)
    print(f"1. 输入 Tokens 形状: {tokens.shape} (Batch=1, Seq=8)")
    
    # 跟踪 Embedding
    h = model.tok_embeddings(tokens)
    print(f"2. Embedding 后形状: {h.shape} (Batch, Seq, Dim=128)")
    
    # 跟踪 Transformer 层
    layer = model.layers[0]
    
    # 预计算 RoPE
    freqs_cos, freqs_sin = model.freqs_cos[:8], model.freqs_sin[:8]
    
    # Attention 前的归一化
    norm_h = layer.attention_norm(h)
    
    # Attention 内部投影
    xq = layer.attention.wq(norm_h)
    xk = layer.attention.wk(norm_h)
    xv = layer.attention.wv(norm_h)
    print(f"3. Attention 投影后:")
    print(f"   - Query (xq): {xq.shape} (32 heads * 4 dim/head)")
    print(f"   - Key   (xk): {xk.shape} (8 heads * 4 dim/head - GQA 模式)")
    
    # 执行 Attention
    attn_out = layer.attention(norm_h, freqs_cos, freqs_sin)
    print(f"4. Attention 输出形状: {attn_out.shape} (与输入一致)")
    
    # FeedForward
    ffn_out = layer.feed_forward(layer.ffn_norm(attn_out))
    print(f"5. FFN 输出形状: {ffn_out.shape}")
    
    # 最终 Logits
    logits = model.output(model.norm(h + attn_out + ffn_out))
    print(f"6. 最终 Logits 形状: {logits.shape} (Batch, Seq, Vocab_Size)")

if __name__ == "__main__":
    visualize_transformer_flow()
