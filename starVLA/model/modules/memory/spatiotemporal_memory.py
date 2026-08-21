"""
Spatiotemporal Memory Module inspired by RynnBrain.

Provides the model with continuity across multi-step tasks by maintaining
a compressed memory of past visual observations and their spatial context.
This enables the robot to:
- Remember where objects were seen in previous frames
- Track object persistence across occlusions
- Maintain consistent spatial references for multi-step manipulation

The memory uses a fixed-size bank of learned slots that are updated via
cross-attention from incoming visual features, implementing a form of
"working memory" for embodied reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemorySlotAttention(nn.Module):
    """
    Cross-attention from memory slots to visual features.

    Memory slots attend to incoming visual tokens to update their
    representations, while visual tokens attend to memory for
    context from previous timesteps.
    """

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

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: [B, Nq, D] - the tokens that attend (memory or visual)
            key_value: [B, Nkv, D] - the tokens being attended to

        Returns:
            updated query: [B, Nq, D]
        """
        B, Nq, D = query.shape
        Nkv = key_value.shape[1]

        q = self.q_proj(self.norm_q(query))
        k = self.k_proj(self.norm_kv(key_value))
        v = self.v_proj(self.norm_kv(key_value))

        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Nkv, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        return self.o_proj(out)


class SpatiotemporalMemory(nn.Module):
    """
    Fixed-size memory bank that maintains spatiotemporal context across
    sequential observations (RynnBrain-inspired).

    Architecture:
    1. Memory bank: [num_slots, dim] learnable embeddings that persist
       across timesteps
    2. Write path: memory slots attend to new visual features (cross-attn)
       to absorb new information
    3. Read path: visual/action tokens attend to memory slots to retrieve
       historical context
    4. Gated update: learned gate controls how much new info overwrites
       existing memory (prevents catastrophic forgetting within an episode)

    Usage in the VLA pipeline:
    - At each timestep, visual features are written to memory
    - Action head reads from memory for temporally-coherent predictions
    - Memory persists across frames during rollout, reset between episodes
    """

    def __init__(
        self,
        dim: int,
        num_slots: int = 64,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots

        # Learnable memory slot initialization
        self.slot_init = nn.Parameter(torch.randn(1, num_slots, dim) * 0.02)

        # Write: memory attends to visual features
        self.write_attn = MemorySlotAttention(dim, num_heads)
        self.write_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )

        # Read: query tokens attend to memory
        self.read_attn = MemorySlotAttention(dim, num_heads)

        # Temporal position encoding for memory timesteps
        self.temporal_embed = nn.Embedding(256, dim)  # up to 256 timesteps

        self.norm = nn.RMSNorm(dim)

    def init_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize memory slots for a new episode."""
        return self.slot_init.expand(batch_size, -1, -1).to(device)

    def write(
        self,
        memory: torch.Tensor,
        visual_features: torch.Tensor,
        timestep: int = 0,
    ) -> torch.Tensor:
        """
        Update memory with new visual observations.

        Args:
            memory: [B, num_slots, D] current memory state
            visual_features: [B, N_vis, D] new visual tokens
            timestep: current timestep for temporal encoding

        Returns:
            updated memory: [B, num_slots, D]
        """
        B = memory.shape[0]
        # Add temporal encoding to visual features
        t_emb = self.temporal_embed(
            timestep
        ).unsqueeze(1).expand(B, visual_features.shape[1], -1)
        vis_with_time = visual_features + t_emb

        # Memory attends to new visual features
        new_info = self.write_attn(memory, vis_with_time)

        # Gated update: blend old memory with new information
        gate_input = torch.cat([memory, new_info], dim=-1)
        gate = self.write_gate(gate_input)
        memory = gate * new_info + (1 - gate) * memory

        return self.norm(memory)

    def read(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Read from memory to provide temporal context.

        Args:
            query: [B, N_query, D] tokens seeking memory context
            memory: [B, num_slots, D] current memory state

        Returns:
            context: [B, N_query, D] memory-augmented representations
        """
        return query + self.read_attn(query, memory)


class ChainOfPointReasoner(nn.Module):
    """
    Chain-of-Point (CoP) spatial reasoning module inspired by RynnBrain.

    Interleaves textual reasoning with physical coordinate grounding:
    the model generates reasoning tokens and spatial point predictions
    alternately, anchoring each reasoning step to real coordinates.

    This bridges the gap between language understanding and physical
    action by ensuring every plan step has concrete spatial grounding.

    Implementation: predicts (x, y, z) coordinates at designated
    reasoning steps, using a lightweight MLP head on the hidden states.
    """

    def __init__(self, dim: int, num_points: int = 8):
        super().__init__()
        self.num_points = num_points

        # Point prediction head: hidden state -> (x, y, z) coordinates
        self.point_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 3),  # (x, y, z)
        )

        # Confidence head: how confident is each point prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

        # Point embedding: encode predicted points back into hidden space
        self.point_embed = nn.Sequential(
            nn.Linear(3, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Predict spatial points from reasoning hidden states.

        Args:
            hidden_states: [B, N, D] from language backbone
            point_mask: [B, N] binary mask indicating where to predict points

        Returns:
            dict with:
                points: [B, N, 3] predicted (x, y, z) coordinates
                confidence: [B, N, 1] prediction confidence
                point_embeddings: [B, N, D] point representations for re-injection
        """
        points = self.point_head(hidden_states)  # [B, N, 3]
        confidence = self.confidence_head(hidden_states)  # [B, N, 1]
        point_embeddings = self.point_embed(points)  # [B, N, D]

        if point_mask is not None:
            mask = point_mask.unsqueeze(-1)
            points = points * mask
            confidence = confidence * mask
            point_embeddings = point_embeddings * mask

        return {
            "points": points,
            "confidence": confidence,
            "point_embeddings": point_embeddings,
        }