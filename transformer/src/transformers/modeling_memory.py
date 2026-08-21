import math
import torch
import torch.nn as nn


class MemorySlotAttention(nn.Module):
    """Multi-head attention, handles [B, V, N, D] or [B, N, D] inputs."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.norm_q = nn.RMSNorm(dim)
        self.norm_kv = nn.RMSNorm(dim)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        if query.dim() == 3:
            B, Nq, D = query.shape
            query = query.unsqueeze(1)
            key_value = key_value.unsqueeze(1)
            squeeze_back = True
        else:
            squeeze_back = False
            
        B, V, Nq, D = query.shape
        Nkv = key_value.shape[2]
        
        q = self.q_proj(self.norm_q(query.reshape(B * V * Nq, D))).reshape(B, V, Nq, D)
        k = self.k_proj(self.norm_kv(key_value.reshape(B * V * Nkv, D))).reshape(B, V, Nkv, D)
        v = self.v_proj(self.norm_kv(key_value.reshape(B * V * Nkv, D))).reshape(B, V, Nkv, D)
        
        q = q.view(B, V, Nq, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B * V * self.num_heads, Nq, self.head_dim)
        k = k.view(B, V, Nkv, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B * V * self.num_heads, Nkv, self.head_dim)
        v = v.view(B, V, Nkv, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B * V * self.num_heads, Nkv, self.head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = out.view(B, V, self.num_heads, Nq, self.head_dim).permute(0, 1, 3, 2, 4).reshape(B, V, Nq, D)
        out = self.o_proj(out.reshape(B * V * Nq, D)).reshape(B, V, Nq, D)
        
        if squeeze_back:
            out = out.squeeze(1)
            
        return out


class ShortTermMemoryBank(nn.Module):
    """
    Short-term memory for multi-level (3 levels) dual-view features with batch dimension.
    预设时间步 T=5，3个层级独立处理，支持 batch 中不同 timestep。
    
    Input:
        - memory: [B, 3, T, 2, 64, D]，3个层级，每层级T步历史，双视角
        - visual: [B, 3, 2, 64, D]，3个层级，当前帧，双视角
        - timestep: [B] 或 int，每个样本的绝对时间步
        
    Output:
        - new_memory: [B, 3, 2, 64, D]，更新后的最新记忆状态（3个层级）
    """
    def __init__(
        self,
        dim: int,
        num_slots: int = 64,
        num_heads: int = 8,
        num_levels: int = 3,
        num_timesteps: int = 5,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_levels = num_levels
        self.num_timesteps = num_timesteps
        
        self.write_attns = nn.ModuleList([
            MemorySlotAttention(dim, num_heads) for _ in range(num_levels)
        ])
        self.read_attns = nn.ModuleList([
            MemorySlotAttention(dim, num_heads) for _ in range(num_levels)
        ])
        
        self.write_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
            for _ in range(num_levels)
        ])
        
        self.norms = nn.ModuleList([nn.RMSNorm(dim) for _ in range(num_levels)])
        
        self.register_buffer(
            "_temporal_div",
            torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        )
        
    def get_temporal_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        获取正弦时间编码，支持 batch。
        
        Args:
            timestep: [B] 或 scalar，每个样本的时间步
        
        Returns:
            emb: [B, D] 或 [D]
        """
        # 确保是 tensor
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=torch.float32, device=self._temporal_div.device)
        # 处理 scalar 情况
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        
        # [B, 1] * [D//2] -> [B, D//2]
        angles = timestep.unsqueeze(-1) * self._temporal_div.to(timestep.device)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [B, D]
        
        return emb
    
    def forward(
        self,
        memory: torch.Tensor,           # [B, 3, T, 2, 64, D]
        visual: torch.Tensor,           # [B, 3, 2, 64, D]
        timestep: torch.Tensor,         # [B] 或 int，每个样本的时间步
    ) -> torch.Tensor:
        """
        更新记忆，3个层级分别处理，支持 batch 中不同 timestep。
        """
        assert memory.dim() == 6, f"memory must be [B,3,T,V,S,D], got {memory.shape}"
        assert visual.dim() == 5, f"visual must be [B,3,V,S,D], got {visual.shape}"
        B, L, T, V, S, D = memory.shape
        assert L == self.num_levels
        assert T == self.num_timesteps
        assert S == self.num_slots
        assert visual.shape == (B, L, V, S, D)
        
        # 处理 timestep：统一为 [B]
        if isinstance(timestep, int):
            timestep = torch.full((B,), timestep, dtype=torch.float32, device=memory.device)
        elif timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(B)
        elif timestep.shape[0] != B:
            raise ValueError(f"timestep batch size {timestep.shape[0]} != memory batch size {B}")
        
        # 获取时间编码: [B, D]
        t_emb = self.get_temporal_embedding(timestep)
        
        # 处理每个层级
        new_memories = []
        for level in range(self.num_levels):
            level_memory = memory[:, level, :, :, :, :]   # [B, T, 2, 64, D]
            level_visual = visual[:, level, :, :, :]       # [B, 2, 64, D]
            
            # 添加时间编码: [B, D] -> [B, 1, 1, D]
            level_visual_t = level_visual + t_emb.view(B, 1, 1, D)  # [B, 2, 64, D]
            
            # 聚合历史: [B, 5, 2, 64, D] -> [B, 2, 5*64, D]
            memory_per_view = level_memory.permute(0, 2, 1, 3, 4).reshape(B, V, T * S, D)
            
            # 用最后一帧 attend 历史
            last_memory = level_memory[:, -1, :, :, :]  # [B, 2, 64, D]
            history_agg = self.write_attns[level](last_memory, memory_per_view)  # [B, 2, 64, D]
            
            # 门控融合
            gate_input = torch.cat([history_agg, level_visual_t], dim=-1)  # [B, 2, 64, 2D]
            gate = self.write_gates[level](gate_input)  # [B, 2, 64, D]
            
            level_new_memory = gate * level_visual_t + (1 - gate) * history_agg
            level_new_memory = self.norms[level](level_new_memory)
            
            new_memories.append(level_new_memory)
        
        # 堆叠: [B, 3, 2, 64, D]
        new_memory = torch.stack(new_memories, dim=1)
        
        return new_memory


# ============== 使用示例 ==============

def demo():
    B, D = 4, 2560  # batch=4
    num_slots = 64
    T = 10
    device = torch.device('cpu')
    memory_bank = ShortTermMemoryBank(dim=D, num_slots=num_slots,num_timesteps=T).to(device)
    
    # 输入
    memory = torch.randn(B, 3, T, 2, num_slots, D).to(device)
    visual = torch.randn(B, 3, 2, num_slots, D).to(device)
    
    # 情况1: 所有样本相同 timestep (int)
    new_memory = memory_bank(memory, visual, timestep=10)
    print(f"Same timestep (int): {new_memory.shape}")
    
    # 情况2: batch 中不同 timestep [B]
    timesteps = torch.tensor([5, 10, 15, 20], dtype=torch.float32).to(device)
    new_memory = memory_bank(memory, visual, timestep=timesteps)
    print(f"Different timesteps [B]: {new_memory.shape}")
    
    # 情况3: scalar tensor
    new_memory = memory_bank(memory, visual, timestep=torch.tensor(7))
    print(f"Scalar tensor: {new_memory.shape}")


if __name__ == "__main__":
    demo()
