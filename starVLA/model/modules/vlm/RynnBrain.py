import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast

from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100

# Action token range — same as QWen3.py (RynnBrain is Qwen3-VL based).
# Only meaningful when using a Fast/Action tokenizer variant.
_ACTION_TOKEN_MIN = 151669
_ACTION_TOKEN_MAX = 153716

class _RynnBrain_Interface(nn.Module):
    """
    Lightweight wrapper around RynnBrain checkpoints (HF: Alibaba-DAMO-Academy/RynnBrain-*)
    Goal: keep same interface style as  _QWen3_VL_Interface.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        rb_cfg = config.framework.get("qwenvl", {}) if config is not None else {}
        model_id = rb_cfg.get("base_vlm", "Alibaba-DAMO-Academy/RynnBrain-8B")
        memory_mode = rb_cfg.get('memory', False)
        max_memory_step = rb_cfg.get('max_memory_step', 5)
        if memory_mode:
            pass

        # also can be:
        #  - Alibaba-DAMO-Academy/RynnBrain-2B
        #  - Alibaba-DAMO-Academy/RynnBrain-30B-A3B
        #  - Alibaba-DAMO-Academy/RynnBrain-Plan-8B / Nav-8B / CoP-8B ...

        # RynnBrain checkpoint is qwen3_vl architecture — load via Qwen3VLForConditionalGeneration
        # so the local transformer (with ShortTermMemoryBank) is used directly.
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            memory_mode=memory_mode,
            max_memory_step=max_memory_step,
        )
        self.max_memory_step = max_memory_step
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        self.processor.tokenizer.padding_side = "left"

        self.config = config
        self.model_id = model_id

        # Align hidden_size (safe no-op if attr missing)
        if hasattr(self.model.config, "text_config") and hasattr(self.model.config.text_config, "hidden_size"):
            self.model.config.hidden_size = self.model.config.text_config.hidden_size

        # Only bind action token range when using a Fast/Action tokenizer variant
        if "-Action" in model_id:
            self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
            self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

        



    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model(**kwargs)

    def generate(self, **kwargs):
        # keep same behavior as your Qwen wrapper
        with torch.autocast("cuda", dtype=torch.float16):
            return self.model.generate(**kwargs)

    def build_rynnbrain_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        RynnBrain is Qwen3-VL based and uses similar chat-template style inputs:
        messages = [{"role":"user","content":[{"type":"image",...},{"type":"text",...}]}]
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]

            # keep your CoT prompt hook
            if self.config is not None and hasattr(self.config, "datasets") and hasattr(self.config.datasets, "vla_data"):
                if "CoT_prompt" in self.config.datasets.vla_data:
                    cot = self.config.datasets.vla_data.get("CoT_prompt", "")
                    prompt = cot.replace("{instruction}", instruction)
                else:
                    prompt = instruction
            else:
                prompt = instruction

            content.append({"type": "text", "text": prompt})

            msg = [{"role": "user", "content": content}]
            if solutions is not None:
                # supervised finetune case
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solutions[len(messages)]}]})
            messages.append(msg)

        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # TODO: add solutions
        # 如果你也想像 starVLA 一样做 label masking：
        # 这里无法假设 action token range（RynnBrain 默认并不是 -Action tokenizer）。
        # 最稳妥：用 assistant 段落的 offset（或 special separator）来 mask。
        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            # 简单策略：把 pad mask 掉；其余是否 mask 取决于你模板里 assistant 起始位置
            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
            batch_inputs["labels"] = labels

        return batch_inputs.to(self.model.device)

    def build_rynnbrain_inputs_with_memorys(
            self,
            images,
            instructions,
            memorys,  # List[List[List[Image.Image]]], [B][X_i][2]
            solutions=None,
            **kwargs
        ):
        """
        Build model inputs from raw data.
        Also preprocess memory images into visual features.
        """
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        B = len(images)

        # --- Step 1: Build chat messages (unchanged) ---
        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            if "CoT_prompt" in self.config.datasets.vla_data:
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction
            content.append({"type": "text", "text": prompt})
            msg = [{"role": "user", "content": content}]
            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # --- Step 2: Tokenize main inputs ---
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # --- Step 3: Process memory images into features ---
        # 检查这里的memorys格式，训练和测试的结果有什么不同
        # Flatten all memory images and collect metadata
        all_memory_images = []
        for mem in memorys:
            for frame in mem:
                # assert len(frame) == 2, "Each memory frame must have exactly 2 views"
                all_memory_images.extend(frame)  # [view0, view1]
        # Preprocess all memory images at once
        processed_mem = self.processor.image_processor(
            images=all_memory_images,
            return_tensors="pt"
        )
        pixel_values_mem = processed_mem["pixel_values"]
        batch_inputs['memorys'] = pixel_values_mem
        batch_inputs['memorys_length'] = len(memorys[0])
        batch_inputs['steps'] = kwargs['steps']
        # --- Step 4: Handle labels (unchanged) ---
        if solutions is not None:
            action_token_min = _ACTION_TOKEN_MIN
            action_token_max = _ACTION_TOKEN_MAX
            labels = batch_inputs['input_ids'].clone()
            for i in range(labels.size(0)):
                seq = labels[i]
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    seq[:] = IGNORE_INDEX
                    RuntimeWarning("Action tokens not found in tokenizer.")
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch_inputs['labels'] = labels

        return batch_inputs.to(self.model.device)    

if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse
    import debugpy
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config"
    )
    args, clipargs = parser.parse_known_args()


    cfg = OmegaConf.load(args.config_yaml)

    # 建议改成你自己的配置字段；如果你暂时还沿用 qwenvl，也能兼容
    if "rynnbrain" not in cfg.framework:
        cfg.framework.rynnbrain = {}

    # 你本地或 HF 模型路径二选一
    cfg.framework.qwenvl.base_vlm = "/home/jiangnan-luxitech/workspace/starVLA/playground/Pretrained_models/RynnBrain-2B"
    # 例如本地路径：
    # cfg.framework.rynnbrain.base_vlm = "./playground/Pretrained_models/RynnBrain-8B"

    print("========== Init RynnBrain Interface ==========")
    rynn_vl = _RynnBrain_Interface(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rynn_vl = rynn_vl.to(device)
    rynn_vl.eval()

    print(f"Model device: {next(rynn_vl.parameters()).device}")
    print(f"Model id: {rynn_vl.model_id}")
    print(f"Hidden size: {rynn_vl.model.config.hidden_size}")

    print("\n========== Tokenizer Check ==========")
    tokenizer = rynn_vl.processor.tokenizer

    test_strings = [
        "🔍",
        "🔍🔍",
        "🔍🔍🔍",
        "🔍 🔍 🔍",
        "<action>",
        "<ACTION>",
    ]

    for s in test_strings:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        print(f"text={repr(s)}")
        print(f"input_ids={ids}")
        print(f"num_tokens={len(ids)}")
        print("-" * 50)

    print("\n========== Fake Sample Build ==========")
    image = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )

    batch_images = [
        [image],   # sample 1: 单视角
        [image],   # sample 2: 单视角
    ]
    instructions = [
        "Pick up the red block and place it into the basket.",
        "Move the object to the target area.",
    ]

    # 模拟 OFT prompt
    chunk_len = 8
    action_token = "🔍"
    action_tokens = action_token * chunk_len
    prompt_suffix = f" Please predict the next {chunk_len} robot actions: <action>{action_tokens}<action>."
    instructions = [ins + prompt_suffix for ins in instructions]

    batch_inputs = rynn_vl.build_rynnbrain_inputs(
        images=batch_images,
        instructions=instructions,
    )

    print("Built input keys:", batch_inputs.keys())
    for k, v in batch_inputs.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
        else:
            print(f"{k}: {type(v)}")

    if "input_ids" in batch_inputs:
        input_ids = batch_inputs["input_ids"]
        print("\n========== Decode Check ==========")
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        print(decoded_text)

        action_token_ids = tokenizer(action_token, add_special_tokens=False)["input_ids"]
        print("\nSingle action token ids:", action_token_ids)

        if len(action_token_ids) == 1:
            action_token_id = action_token_ids[0]
            action_count = (input_ids == action_token_id).sum(dim=1)
            print("Action token counts per sample:", action_count.tolist())
        else:
            print("WARNING: action token is not a single token, current OFT gather logic will fail.")

    print("\n========== Forward Check ==========")
    with torch.inference_mode():
        outputs = rynn_vl(
            **batch_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

    print("Forward success.")
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        print(f"Num hidden states: {len(outputs.hidden_states)}")
        print(f"Last hidden shape: {tuple(outputs.hidden_states[-1].shape)}")
    else:
        print("WARNING: outputs.hidden_states is None")

    print("\n========== Generate Check (Optional) ==========")
    try:
        gen_inputs = dict(batch_inputs)
        if "labels" in gen_inputs:
            gen_inputs.pop("labels")

        with torch.inference_mode():
            generated = rynn_vl.generate(
                **gen_inputs,
                max_new_tokens=16,
                do_sample=False,
            )

        print("Generate success.")
        print("Generated shape:", tuple(generated.shape))
        print("Decoded output:")
        print(tokenizer.decode(generated[0], skip_special_tokens=False))
    except Exception as e:
        print("Generate failed:", repr(e))

    print("\n========== Done ==========")
