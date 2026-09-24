"""Small CPU exercise using Evo-1's real FlowmatchingActionHead.

Run from the repository root:
    conda run -n cs231n python learning/flow_matching_exercise.py --check
    conda run -n cs231n python learning/flow_matching_exercise.py

Fill in the two TODO functions yourself. No model weights or robot data are needed.
"""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Evo_1"))
from config import EvoConfig
from model.action_head.flow_matching import FlowmatchingActionHead


def training_step(model, fused_tokens, state, actions_gt, action_mask):
    """Return (pred_velocity, noise, target_velocity, loss).

    TODO 1:
    - Call model's training forward pass with the four tensors above.
    - The target velocity points from sampled noise toward the true action.
    - Flatten the action dimensions so target and prediction have the same shape.
    - Compute mean squared error. Here the mask is all ones.

    Hints: paper Eq. (3)-(4), flow_matching.py forward(), train.py near
    `target_velocity = ...`.
    """
    raise NotImplementedError("Fill in TODO 1: training_step")


def sample_actions(model, fused_tokens, state, action_mask):
    """Return a generated action sequence with shape [batch, horizon * per_action_dim].

    TODO 2: Call the model's sampling method with context, state, and mask.
    Hint: flow_matching.py get_action().
    """
    raise NotImplementedError("Fill in TODO 2: sample_actions")


def main():
    torch.manual_seed(0)
    config = EvoConfig(
        device="cpu",
        embed_dim=64,
        hidden_dim=128,
        state_hidden_dim=64,
        num_heads=4,
        num_layers=2,
        horizon=4,
        per_action_dim=3,
        state_dim=3,
        num_inference_timesteps=4,
    )
    model = FlowmatchingActionHead(config=config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch_size, token_count = 2, 5
    fused_tokens = torch.randn(batch_size, token_count, config.embed_dim)
    state = torch.randn(batch_size, config.state_dim)
    actions_gt = torch.randn(batch_size, config.horizon, config.per_action_dim)
    action_mask = torch.ones_like(actions_gt)

    print(f"fused_tokens: {tuple(fused_tokens.shape)}")
    print(f"state:        {tuple(state.shape)}")
    print(f"actions_gt:   {tuple(actions_gt.shape)}")
    print(f"action_mask:  {tuple(action_mask.shape)}")
    if "--check" in sys.argv:
        print("Environment and input shapes are ready.")
        return

    model.train()
    pred_velocity, noise, target_velocity, loss = training_step(
        model, fused_tokens, state, actions_gt, action_mask
    )
    assert pred_velocity.shape == target_velocity.shape == (batch_size, config.action_dim)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        generated = sample_actions(model, fused_tokens, state, action_mask)
    assert generated.shape == (batch_size, config.action_dim)

    print(f"noise:           {tuple(noise.shape)}")
    print(f"target_velocity: {tuple(target_velocity.shape)}")
    print(f"loss:            {loss.item():.4f}")
    print(f"generated:       {tuple(generated.shape)}")


if __name__ == "__main__":
    main()
