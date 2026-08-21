"""Robolatent data config."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)


class RobolatentSingleArmJointConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    video_keys = [
        "video.cam_head",
        "video.cam_high",
        "video.cam_left_wrist",
        "video.cam_right_wrist",
    ]

    state_keys = [
        "state.joints",
        "state.gripper",
    ]

    action_keys = [
        "action.joints",
        "action.gripper",
    ]

    state_key_dims = {
        "state.joints": 6,
        "state.gripper": 1,
    }

    action_key_dims = {
        "action.joints": 6,
        "action.gripper": 1,
    }

    language_keys = ["annotation.human.action.task_description"]

    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        "state.joints": "min_max",
                        "state.gripper": "binary",
                    },
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.joints": "min_max",
                        "action.gripper": "binary",
                    },
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "robolatent_single_arm_joint": RobolatentSingleArmJointConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    # 单独训练 PickXtimes
    "robolatent_pickxtimes_left": [
        ("robolatent_pickxtimes_left", 1.0, "robolatent_single_arm_joint"),
    ],

    # 单独训练 UncoverBlock
    "robolatent_uncoverblock_left": [
        ("robolatent_uncoverblock_left", 1.0, "robolatent_single_arm_joint"),
    ],

    # 单独训练 PutBackBlock
    "robolatent_putbackblock_left": [
        ("robolatent_putbackblock_left", 1.0, "robolatent_single_arm_joint"),
    ],

    # 三个任务混合训练
    "robolatent_all_left": [
        ("robolatent_pickxtimes_left", 1.0, "robolatent_single_arm_joint"),
        ("robolatent_uncoverblock_left", 1.0, "robolatent_single_arm_joint"),
        ("robolatent_putbackblock_left", 1.0, "robolatent_single_arm_joint"),
    ],
}

