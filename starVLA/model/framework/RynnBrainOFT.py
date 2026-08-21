from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model as get_mlp_action_model
from starVLA.model.modules.action_model.DiTActionHeader import get_action_model as get_dit_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)
IGNORE_INDEX = -100
DIT_CONDITION_DIMS = {"DiT-S": 384, "DiT-B": 768, "DiT-L": 1024}


# 你之前实现的 wrapper
# from starVLA.model.modules.vlm.rynnbrain_interface import _RynnBrain_Interface


@FRAMEWORK_REGISTRY.register("RynnBrainOFT")
class RynnBrain_OFT(baseframework):
    """
    OFT on top of RynnBrain:
      - RynnBrain backbone provides hidden states
      - Inject action placeholder tokens into prompt
      - Gather their hidden states and regress continuous actions via MLP head
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config

        # --- 1) Init RynnBrain backbone ---
        self.vlm_interface = get_vlm_model(config=self.config)

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.action_model_type = config.framework.action_model.get("action_model_type", "L1RegressionActionHead")
        self.use_dit_action_model = self.action_model_type in DIT_CONDITION_DIMS
        if self.use_dit_action_model and self.past_action_window_size != 0:
            raise ValueError("RynnBrainOFT DiT action head currently requires past_action_window_size == 0.")

        # --- 2) Init action head ---
        self.vlm_hidden_size = self.vlm_interface.model.config.hidden_size
        if self.use_dit_action_model:
            self.config.framework.action_model.n_condition_token = self.chunk_len
            self.config.framework.action_model.action_hidden_dim = DIT_CONDITION_DIMS[self.action_model_type]
            self.action_condition_projector = nn.Linear(
                self.vlm_hidden_size,
                self.config.framework.action_model.action_hidden_dim,
            )
            self.action_model = get_dit_action_model(config=self.config)
        elif self.action_model_type == "L1RegressionActionHead":
            self.config.framework.action_model.action_hidden_dim = self.vlm_hidden_size
            self.action_condition_projector = nn.Identity()
            self.action_model = get_mlp_action_model(config=self.config)
        else:
            raise ValueError(
                f"Unsupported action_model_type `{self.action_model_type}` for RynnBrainOFT. "
                f"Use one of {sorted(DIT_CONDITION_DIMS)} or `L1RegressionActionHead`."
            )

        # --- 3) Choose action placeholder token ---
        self.action_token = "🔍"
        ids = self.vlm_interface.processor.tokenizer(
            self.action_token, add_special_tokens=False
        )["input_ids"]

        if len(ids) != 1:
            raise RuntimeError(
                f"action_token '{self.action_token}' is not a single token (got ids={ids}). "
                f"Please choose another token or add it as a special token."
            )
        self.action_token_id = ids[0]

        # --- 4) Memory mode ---
        self.memory_mode = config.framework.qwenvl.memory

        self.l1_loss = nn.L1Loss()

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        """
        Train forward: L1 regression on future actions
        examples[i] requires:
          - image: List[PIL.Image]  (multi-view)
          - lang: str
          - action: np.ndarray [T, action_dim]
        """
        batch_images = [ex["image"] for ex in examples]          # [B, [PIL,...]]
        instructions = [ex["lang"] for ex in examples]          # [B]
        actions = [ex["action"] for ex in examples]             # [B, T, A]
        if self.memory_mode:
            memorys = [ex["memory"] for ex in examples]
            steps = [ex["step"] for ex in examples]

        # step 0: append action placeholders
        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [ins + prompt_suffix for ins in instructions]

        # step 1: build inputs
        if not self.memory_mode:
            rb_inputs = self.vlm_interface.build_rynnbrain_inputs(
                images=batch_images, instructions=instructions
            )
        else:
            rb_inputs = self.vlm_interface.build_rynnbrain_inputs_with_memorys(
                images=batch_images, instructions=instructions, memorys=memorys, steps=steps
            )

        # step 2: run backbone
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm_interface(
                **rb_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]  # [B, L, H]

        # step 3: gather action token embeddings -> action head -> loss
        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = rb_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )  # [B, chunk_len, H]

            action_condition = self.action_condition_projector(action_queries)
            actions = torch.tensor(np.array(actions), device=action_condition.device, dtype=action_condition.dtype)
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]  # [B, chunk_len, action_dim]

            if self.use_dit_action_model:
                repeated_diffusion_steps = (
                    self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
                )
                actions_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                action_condition = action_condition.repeat(repeated_diffusion_steps, 1, 1)
                noise_pred, noise, timestep = self.action_model(actions_repeated, action_condition)
                action_loss = self.action_model.loss(noise_pred, noise)
            else:
                pred_actions = self.action_model.predict_action(action_condition)  # [B, chunk_len, action_dim]
                action_loss = self.l1_loss(pred_actions, actions_target)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> np.ndarray:
        """
        Inference: regress continuous actions
        """
        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        if self.memory_mode:
            # Memory is shipped over msgpack as numpy (read-only, buffer-backed).
            # Convert to PIL to (i) match the training path where `_pack_sample`
            # stores PIL frames, (ii) copy the data so the HF image processor
            # does not see a non-writable array.
            memorys = [to_pil_preserve(ex["memory"]) for ex in examples]
            steps = [ex["step"] for ex in examples]

            # ---- [MEMORY PROBE] prints first N eval calls then a periodic sample ----
            if not hasattr(self, "_mem_probe_count"):
                self._mem_probe_count = 0
            if self._mem_probe_count < 3 or self._mem_probe_count % 50 == 0:
                try:
                    b = len(memorys)
                    outer = len(memorys[0])
                    inner = len(memorys[0][0])
                    leaf = memorys[0][0][0]
                    leaf_info = (
                        f"type={type(leaf).__name__} size={getattr(leaf, 'size', None)} "
                        f"mode={getattr(leaf, 'mode', None)}"
                    )
                    main_leaf = batch_images[0][0]
                    main_info = (
                        f"type={type(main_leaf).__name__} size={getattr(main_leaf, 'size', None)} "
                        f"mode={getattr(main_leaf, 'mode', None)}"
                    )
                    print(
                        f"[MEM PROBE #{self._mem_probe_count}] "
                        f"batch={b} | memory[{b}][{outer}][{inner}] leaf: {leaf_info} | "
                        f"main_image[{len(batch_images[0])}] leaf: {main_info} | "
                        f"steps={steps}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[MEM PROBE #{self._mem_probe_count}] structure print failed: {e!r}", flush=True)
            self._mem_probe_count += 1
            # ---- [/MEMORY PROBE] ----

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        action_tokens = self.action_token * self.chunk_len
        prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        instructions = [ins + prompt_suffix for ins in instructions]

        if not self.memory_mode:
            rb_inputs = self.vlm_interface.build_rynnbrain_inputs(
                images=batch_images, instructions=instructions
            )
        else:
            rb_inputs = self.vlm_interface.build_rynnbrain_inputs_with_memorys(
                images=batch_images, instructions=instructions, memorys=memorys, steps=steps
            )

            # ---- [MEMORY PROBE] post-processor shapes actually entering the model ----
            if self._mem_probe_count <= 3 or self._mem_probe_count % 50 == 1:
                try:
                    pv_mem = rb_inputs.get("memorys", None)
                    pv_main = rb_inputs.get("pixel_values", None)
                    print(
                        f"[MEM PROBE #{self._mem_probe_count - 1}] post-processor: "
                        f"pixel_values(main)={tuple(pv_main.shape) if pv_main is not None else None}, "
                        f"memorys={tuple(pv_mem.shape) if pv_mem is not None else None}, "
                        f"memorys_length={rb_inputs.get('memorys_length')}, "
                        f"steps_field={rb_inputs.get('steps')}, "
                        f"input_ids={tuple(rb_inputs['input_ids'].shape)}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[MEM PROBE post] failed: {e!r}", flush=True)
            # ---- [/MEMORY PROBE] ----

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.vlm_interface(
                **rb_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            input_ids = rb_inputs.get("input_ids", None)
            action_queries = self._gather_action_token_embeddings(
                last_hidden, input_ids, action_token_id=self.action_token_id
            )
            action_condition = self.action_condition_projector(action_queries)
            if self.use_dit_action_model:
                pred_actions = self._sample_dit_actions(action_condition, **kwargs)
            else:
                pred_actions = self.action_model.predict_action(action_condition)

        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    def _sample_dit_actions(
        self,
        action_condition: torch.Tensor,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 5,
        **kwargs,
    ) -> torch.Tensor:
        using_cfg = cfg_scale > 1.0
        model_dtype = next(self.action_model.net.parameters()).dtype
        B = action_condition.shape[0]

        noise = torch.randn(
            B,
            self.future_action_window_size + 1,
            self.action_model.in_channels,
            device=action_condition.device,
            dtype=model_dtype,
        )

        if using_cfg:
            noise = torch.cat([noise, noise], dim=0)
            uncondition = self.action_model.net.z_embedder.uncondition.unsqueeze(0).expand(B, -1, -1)
            z = torch.cat([action_condition, uncondition], dim=0)
            model_kwargs = dict(z=z, cfg_scale=cfg_scale)
            sample_fn = self.action_model.net.forward_with_cfg
        else:
            model_kwargs = dict(z=action_condition)
            sample_fn = self.action_model.net.forward

        if use_ddim:
            if self.action_model.ddim_diffusion is None:
                self.action_model.create_ddim(ddim_step=num_ddim_steps)
            samples = self.action_model.ddim_diffusion.ddim_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=action_condition.device,
                eta=0.0,
            )
        else:
            samples = self.action_model.diffusion.p_sample_loop(
                sample_fn,
                noise.shape,
                noise,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=action_condition.device,
            )

        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)
        return samples

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,   # [B, L, H]
        input_ids: torch.Tensor,     # [B, L]
        action_token_id=None,
    ) -> torch.Tensor:
        """
        Same as your Qwenvl_OFT version (vectorized gather of last chunk_len action tokens)
        """
        if action_token_id is None:
            raise ValueError("action_token_id 不能为空")

        device = input_ids.device
        B, L, H = last_hidden.shape

        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)

        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"以下样本动作 token 数量不足 {self.chunk_len}: {insufficient} | counts={counts.tolist()}"
            )

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))

        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values
        selected_pos = topk_pos.sort(dim=-1).values

        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)
        return last_hidden.gather(dim=1, index=expanded_index)
