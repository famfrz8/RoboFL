from collections import defaultdict

from lerobot.utils.constants import OBS_STATE, ACTION, OBS_IMAGES, OBS_IMAGE
from .utils import make_bool_mask


MASK_MAPPING = {
    # a1 old
    "piper": make_bool_mask(6, -1, 6, -1),  # split_aloha
    "arx_lift2": make_bool_mask(6, -1, 6, -1), 
    "split_aloha": make_bool_mask(6, -1, 6, -1), 
    "a2d": make_bool_mask(14, -2),  # agibotworld
    "genie1": make_bool_mask(14, -2), 
    "franka": make_bool_mask(7, -1), 
    "frankarobotiq": make_bool_mask(7, -1), 
    # a1 new
    "Franka": make_bool_mask(7, -1), 
    "ARX Lift-2": make_bool_mask(6, -1, 6, -1), 
    "AgileX Split Aloha": make_bool_mask(6, -1, 6, -1), 
    "Genie-1": make_bool_mask(14, -2), 
    "ARX AC One": make_bool_mask(6, -1, 6, -1), 
    # others
    "aloha": make_bool_mask(6, -1, 6, -1), 
    "nero_dual_arm": make_bool_mask(7, -1, 7, -1),
    "panda": make_bool_mask(7, ), 
    "FrankaEEF": make_bool_mask(7, -1),
    "FrankaEEF3Cam": make_bool_mask(7, -1),
    "mole_rlbench_single_view": make_bool_mask(6, -1),
    "intern_gello_7dof_robotiq": make_bool_mask(7, -1),
}


FEATURE_MAPPING = defaultdict(
    lambda : {
        OBS_STATE: ["observation.state"],
        ACTION: ["action"],
    }, 
    a2d={
        OBS_STATE: [
            "observation.states.joint.position", 
            "observation.states.effector.position", 
        ], 
        ACTION: [
            "actions.joint.position", 
            "actions.effector.position", 
        ], 
    }, 
    genie1={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.right_joint.position", 
            "states.left_gripper.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.right_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    arx_lift2={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_joint.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    piper={
        OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
        ACTION: [
            "actions.left_joint.position", 
            "actions.left_gripper.position", 
            "actions.right_joint.position", 
            "actions.right_gripper.position", 
        ], 
    }, 
    r1lite={
        OBS_STATE: [
            'observation.state.left_arm', 
            'observation.state.right_arm', 
            'observation.state.left_gripper', 
            'observation.state.right_gripper',
        ], 
        ACTION: [
            "action.left_arm", 
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
        ], 
    },
    nero_dual_arm={
        OBS_STATE: ["observation.state"],
        ACTION: ["action"],
    },
    aloha={
        OBS_STATE: [
            'observation.state',
        ], 
        ACTION: [
            'action',
        ], 
    },
    franka={
        OBS_STATE: [
            "states.joint.position", 
            "states.gripper.position",
        ], 
        ACTION: [
            "actions.joint.position", 
            "actions.gripper.position", 
        ], 
    }, 
    panda={
        OBS_STATE: [
            "observation.state", 
        ], 
        ACTION: [
            "action", 
        ], 
    }
)
# a1 new
FEATURE_MAPPING["Franka"] = {
    OBS_STATE: [
            "states.joint.position", 
            "states.gripper.position",
    ], 
    ACTION: [
        "actions.joint.position", 
        "actions.gripper.position", 
    ], 
}
FEATURE_MAPPING["FrankaEEF"] = {
    OBS_STATE: [
        "states.effector.position",
        "states.gripper.position",
    ],
    ACTION: [
        "actions.effector.position",
        "actions.gripper.position",
    ],
}
FEATURE_MAPPING["FrankaLIFT3D"] = {
    OBS_STATE: [
        "states.joint.position",
        "states.effector.position",
        "states.gripper.position",
    ],
    ACTION: [
        "actions.effector.position",
        "actions.gripper.position",
    ],
}
FEATURE_MAPPING["FrankaLIFT3D3Cam"] = FEATURE_MAPPING["FrankaLIFT3D"]
FEATURE_MAPPING["FrankaEEF3Cam"] = FEATURE_MAPPING["FrankaEEF"]
FEATURE_MAPPING["mole_rlbench_single_view"] = FEATURE_MAPPING["FrankaEEF"]
FEATURE_MAPPING["ARX Lift-2"] = {
    OBS_STATE: [
            "states.left_joint.position", 
            "states.left_gripper.position", 
            "states.right_joint.position", 
            "states.right_gripper.position", 
        ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_joint.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["Genie-1"] = {
    OBS_STATE: [
        "states.left_joint.position", 
        "states.right_joint.position", 
        "states.left_gripper.position", 
        "states.right_gripper.position", 
    ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.right_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["AgileX Split Aloha"] = {
    OBS_STATE: [
        "states.left_joint.position", 
        "states.left_gripper.position", 
        "states.right_joint.position", 
        "states.right_gripper.position", 
    ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_joint.position", 
        "actions.right_gripper.position", 
    ], 
}
FEATURE_MAPPING["ARX AC One"] = {
    OBS_STATE: [
        "states.left_joint.position", 
        "states.left_gripper.position", 
        "states.right_joint.position", 
        "states.right_gripper.position", 
    ], 
    ACTION: [
        "actions.left_joint.position", 
        "actions.left_gripper.position", 
        "actions.right_joint.position", 
        "actions.right_gripper.position", 
    ], 
}


FEATURE_MAPPING["intern_gello_7dof_robotiq"] = {
    OBS_STATE: [OBS_STATE], ACTION: [ACTION],
}

IMAGE_MAPPING = defaultdict(
    lambda : {
        "observation.image": f"{OBS_IMAGES}.image0", 
    }, 
    arx_lift2={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    piper={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    },
    genie1={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
        "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    a2d={
        "observation.images.head": f"{OBS_IMAGES}.image0", 
        "observation.images.hand_left": f"{OBS_IMAGES}.image1", 
        "observation.images.hand_right": f"{OBS_IMAGES}.image2", 
    }, 
    # todo, make sure what the key names are for franka
    franka={
        "images.rgb.head": f"{OBS_IMAGES}.image0", 
        "images.rgb.hand": f"{OBS_IMAGES}.image1", 
    }, 
    r1lite={
        "observation.images.head_rgb": f"{OBS_IMAGES}.image0", 
        "observation.images.left_wrist_rgb": f"{OBS_IMAGES}.image1", 
        "observation.images.right_wrist_rgb": f"{OBS_IMAGES}.image2", 
    },

    nero_dual_arm={
        "observation.images.third_person": f"{OBS_IMAGES}.image0",
        "observation.images.left_wrist": f"{OBS_IMAGES}.image1",
        "observation.images.right_wrist": f"{OBS_IMAGES}.image2",
    },
    aloha={
        "observation.images.cam_high": f"{OBS_IMAGES}.image0", 
        "observation.images.cam_left_wrist": f"{OBS_IMAGES}.image1", 
        "observation.images.cam_right_wrist": f"{OBS_IMAGES}.image2", 
    },
    panda={
        "observation.images.image": f"{OBS_IMAGES}.image0", 
        "observation.images.image2": f"{OBS_IMAGES}.image1", 
    }
)
# a1 new
IMAGE_MAPPING["Franka"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand": f"{OBS_IMAGES}.image1", 
}
IMAGE_MAPPING["FrankaEEF"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0",
    "images.rgb.hand": f"{OBS_IMAGES}.image1",
}
IMAGE_MAPPING["FrankaLIFT3D"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0",
    "images.rgb.hand": f"{OBS_IMAGES}.image1",
}
IMAGE_MAPPING["FrankaLIFT3D3Cam"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0",
    "images.rgb.hand": f"{OBS_IMAGES}.image1",
    "images.rgb.overhead": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["FrankaEEF3Cam"] = {
    "images.rgb.front": f"{OBS_IMAGES}.image0",
    "images.rgb.left_shoulder": f"{OBS_IMAGES}.image1",
    "images.rgb.right_shoulder": f"{OBS_IMAGES}.image2",
}
IMAGE_MAPPING["mole_rlbench_single_view"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0",
}
IMAGE_MAPPING["intern_gello_7dof_robotiq"] = {
    "observation.images.cam_high": f"{OBS_IMAGES}.image0",
    "observation.images.cam_front": f"{OBS_IMAGES}.image1",
}
IMAGE_MAPPING["ARX Lift-2"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["Genie-1"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["AgileX Split Aloha"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
IMAGE_MAPPING["ARX AC One"] = {
    "images.rgb.head": f"{OBS_IMAGES}.image0", 
    "images.rgb.hand_left": f"{OBS_IMAGES}.image1", 
    "images.rgb.hand_right": f"{OBS_IMAGES}.image2", 
}
