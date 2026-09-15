"""Export a CTS / MoE-CTS training checkpoint to deployment TorchScript + ONNX.

Runs on CPU only (no IsaacLab needed): reconstructs ActorCriticMoECTS with the
dimensions inferred from the checkpoint state dict, loads the weights, and calls
the repository's own exporter helpers so the output matches `play.py` exactly.

Usage:
    python export_cts_standalone.py <run_dir> <checkpoint_name> [out_dir]
"""
from __future__ import annotations

import os
import sys

import torch
from tensordict import TensorDict

REPO = "/home/robot/go2_rl_robotlab"
sys.path.insert(0, os.path.join(REPO, "source", "rsl_rl"))
sys.path.insert(0, os.path.join(REPO, "scripts", "rsl_rl"))

from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS  # noqa: E402
from utils import export_cts_policy_as_jit, export_cts_policy_as_onnx  # noqa: E402

NUM_ACTIONS = 12


def infer_dims(sd: dict) -> dict:
    def layer_outs(prefix: str) -> list[int]:
        keys = [k for k in sd if k.startswith(prefix) and k.endswith(".weight")]
        keys.sort(key=lambda k: int(k.split(".")[-2]))
        return [sd[k].shape[0] for k in keys]

    actor_layers = layer_outs("actor.network.")
    critic_layers = layer_outs("critic.network.")
    teacher_layers = layer_outs("teacher_encoder.0.network.")
    gating_layers = layer_outs("student_moe_encoder.moe.gating_network.0.network.")

    num_actor_obs = sd["student_moe_encoder.moe.experts.backbone.network.0.weight"].shape[1]
    latent_dim = teacher_layers[-1]
    num_single_obs = sd["actor.network.0.weight"].shape[1] - latent_dim
    num_critic_obs = teacher_layers[0] if False else sd["teacher_encoder.0.network.0.weight"].shape[1]
    expert_num = gating_layers[-1]
    return {
        "num_actor_obs": num_actor_obs,
        "num_single_obs": num_single_obs,
        "num_critic_obs": num_critic_obs,
        "latent_dim": latent_dim,
        "expert_num": expert_num,
        # hidden dims exclude the final output layer of each MLP
        "actor_hidden": actor_layers[:-1],
        "critic_hidden": critic_layers[:-1],
        "teacher_hidden": teacher_layers[:-1],
        # MoE hidden_dims drive both gating (full list) and expert backbone (list[:-1])
        "student_hidden": gating_layers[:-1],
    }


def main() -> None:
    run_dir = sys.argv[1]
    ckpt_name = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(run_dir, "exported")
    ckpt_path = os.path.join(run_dir, ckpt_name)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    print(f"[info] checkpoint iter = {ckpt.get('iter')}")
    print(f"[info] state dict keys ({len(sd)}):")
    for key, value in sd.items():
        print(f"   {key} {tuple(value.shape) if hasattr(value, 'shape') else value}")

    dims = infer_dims(sd)
    print(f"[info] inferred dims: {dims}")
    if dims["num_actor_obs"] % dims["num_single_obs"] != 0:
        raise SystemExit("num_actor_obs must be divisible by num_single_obs")

    obs = TensorDict(
        {
            "policy": torch.zeros(1, dims["num_actor_obs"]),
            "critic": torch.zeros(1, dims["num_critic_obs"]),
            "single_obs": torch.zeros(1, dims["num_single_obs"]),
        },
        batch_size=[1],
    )
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = ActorCriticMoECTS(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=NUM_ACTIONS,
        actor_hidden_dims=dims["actor_hidden"],
        critic_hidden_dims=dims["critic_hidden"],
        teacher_encoder_hidden_dims=dims["teacher_hidden"],
        student_encoder_hidden_dims=dims["student_hidden"],
        expert_num=dims["expert_num"],
        latent_dim=dims["latent_dim"],
    )
    policy.load_state_dict(sd, strict=False)
    model_sd = policy.state_dict()
    missing = [k for k in model_sd if k not in sd]
    unexpected = [k for k in sd if k not in model_sd]
    mismatch = [k for k in sd if k in model_sd and tuple(model_sd[k].shape) != tuple(sd[k].shape)]
    print(f"[verify] missing={missing}")
    print(f"[verify] unexpected={unexpected}")
    print(f"[verify] shape-mismatch={mismatch}")
    if mismatch:
        raise SystemExit("state dict shape mismatch")
    policy.eval()

    actor_norm = getattr(policy, "actor_obs_normalizer", None)
    single_norm = getattr(policy, "single_obs_normalizer", None)
    print(f"[info] actor_obs_normalizer={type(actor_norm).__name__} single_obs_normalizer={type(single_norm).__name__}")

    os.makedirs(out_dir, exist_ok=True)
    export_cts_policy_as_jit(
        policy, actor_obs_normalizer=actor_norm, single_obs_normalizer=single_norm, path=out_dir, filename="policy.pt"
    )
    export_cts_policy_as_onnx(
        policy, actor_obs_normalizer=actor_norm, single_obs_normalizer=single_norm, path=out_dir, filename="policy.onnx"
    )
    print(f"[done] exported to {out_dir}")

    # quick self-check: run the exported TorchScript once
    exported = torch.jit.load(os.path.join(out_dir, "policy.pt"))
    single = torch.zeros(1, dims["num_single_obs"])
    for step in range(12):  # fill history
        action = exported(single)
    print(f"[check] TorchScript forward ok, action shape={tuple(action.shape)}")


if __name__ == "__main__":
    main()
