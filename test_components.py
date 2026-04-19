import torch
import math
from llama3.model import RMSNorm, precompute_freqs_cis, apply_rotary_emb

def test_rmsnorm():
    print("测试 RMSNorm...")
    dim = 64
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim)
    output = norm(x)
    
    # RMSNorm 的输出在维度维度上的平方均值应该接近 1
    rms = torch.sqrt(output.pow(2).mean(-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)
    print("✅ RMSNorm 测试通过！")

def test_rope_property():
    print("测试 RoPE 旋转性质...")
    # 维度设为 2，方便手动验证
    dim = 2
    seq_len = 2
    theta = 10000.0
    
    cos, sin = precompute_freqs_cis(dim, seq_len, theta)
    
    # 创建一个简单的向量 [1, 0]
    xq = torch.tensor([[[[1.0, 0.0]]]] ) # [Batch=1, Seq=1, Head=1, Dim=2]
    xk = xq.clone()
    
    # 应用旋转（位置 1）
    # 旋转角度应为 1 * theta^(0) = 1 弧度
    # 旋转后的向量应为 [cos(1), sin(1)]
    xq_out, _ = apply_rotary_emb(xq, xk, cos[1:2], sin[1:2])
    
    expected = torch.tensor([math.cos(1.0), math.sin(1.0)])
    assert torch.allclose(xq_out.flatten(), expected, atol=1e-5)
    print("✅ RoPE 旋转性质测试通过！")

if __name__ == "__main__":
    test_rmsnorm()
    test_rope_property()
