import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


STATE_FEATURES = [
    "goal_distance_m",
    "min_dynamic_obstacle_distance_m",
    "safety_scale",
    "raw_linear_mps",
    "safe_linear_mps",
    "global_path_length_m",
    "local_trajectory_length_m",
    "progress_mps",
    "risk",
]

ACTION_FEATURES = [
    "speed_scale",
    "lookahead_scale",
    "local_horizon_scale",
    "obstacle_caution",
]

ACTION_LOW = np.array([0.62, 0.95, 1.00, 1.00], dtype=np.float32)
ACTION_HIGH = np.array([1.00, 1.28, 1.45, 1.70], dtype=np.float32)


def mlp(in_dim, out_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.backbone = mlp(state_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.backbone[:-1](state)
        h = self.backbone[-1](h)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5.0, 2.0)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        mean, _ = self(state)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.q = mlp(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=-1))


def action_to_norm(action):
    return 2.0 * (action - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW) - 1.0


def norm_to_action(action_norm):
    return ACTION_LOW + 0.5 * (action_norm + 1.0) * (ACTION_HIGH - ACTION_LOW)


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)

    if "source_file" not in df.columns:
        df["source_file"] = Path(csv_path).name

    valid = df[
        (df["goal_distance_m"] >= 0.0)
        & (df["min_dynamic_obstacle_distance_m"] >= 0.0)
        & (df["navigation_status"] != "UNKNOWN")
    ].copy()

    valid = valid.sort_values(["source_file", "episode_id", "step"]).reset_index(drop=True)

    for col in STATE_FEATURES + ACTION_FEATURES + ["reward", "done"]:
        if col not in valid.columns:
            raise ValueError(f"Missing required column: {col}")

    next_state_cols = []
    group_cols = ["source_file", "episode_id"]

    for col in STATE_FEATURES:
        next_col = "next_" + col
        valid[next_col] = valid.groupby(group_cols)[col].shift(-1)
        valid[next_col] = valid[next_col].fillna(valid[col])
        next_state_cols.append(next_col)

    is_last = valid.groupby(group_cols)["step"].transform("max") == valid["step"]
    terminal = (valid["done"].astype(int) > 0) | is_last

    states = valid[STATE_FEATURES].astype(np.float32).to_numpy()
    next_states = valid[next_state_cols].astype(np.float32).to_numpy()
    actions = valid[ACTION_FEATURES].astype(np.float32).to_numpy()
    actions = np.clip(actions, ACTION_LOW, ACTION_HIGH)
    actions_norm = action_to_norm(actions)

    rewards = valid["reward"].astype(np.float32).to_numpy().reshape(-1, 1)
    dones = terminal.astype(np.float32).to_numpy().reshape(-1, 1)

    return valid, states, actions_norm.astype(np.float32), rewards, next_states, dones


def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_param.data * tau
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/root/ugv_nav_ws/results/c15_multi_map_rl_dataset/c15_all_valid_transitions.csv")
    parser.add_argument("--out-dir", default="/root/ugv_nav_ws/results/c2_offline_sac")
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid, states_np, actions_np, rewards_np, next_states_np, dones_np = load_dataset(args.csv)

    state_mean = states_np.mean(axis=0)
    state_std = states_np.std(axis=0)
    state_std[state_std < 1e-6] = 1.0

    states_np = (states_np - state_mean) / state_std
    next_states_np = (next_states_np - state_mean) / state_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    states = torch.tensor(states_np, dtype=torch.float32, device=device)
    actions = torch.tensor(actions_np, dtype=torch.float32, device=device)
    rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)
    dones = torch.tensor(dones_np, dtype=torch.float32, device=device)

    state_dim = states.shape[1]
    action_dim = actions.shape[1]

    actor = Actor(state_dim, action_dim, args.hidden_dim).to(device)
    critic1 = Critic(state_dim, action_dim, args.hidden_dim).to(device)
    critic2 = Critic(state_dim, action_dim, args.hidden_dim).to(device)
    target1 = Critic(state_dim, action_dim, args.hidden_dim).to(device)
    target2 = Critic(state_dim, action_dim, args.hidden_dim).to(device)

    target1.load_state_dict(critic1.state_dict())
    target2.load_state_dict(critic2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=args.lr)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=args.lr)

    log_alpha = torch.tensor(math.log(0.2), dtype=torch.float32, device=device, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr)
    target_entropy = -float(action_dim)

    log_path = out_dir / "sac_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["update", "critic_loss", "actor_loss", "alpha_loss", "alpha", "bc_loss"],
        )
        writer.writeheader()

        n = states.shape[0]
        for update in range(1, args.updates + 1):
            idx = torch.randint(0, n, (args.batch_size,), device=device)

            s = states[idx]
            a = actions[idx]
            r = rewards[idx]
            ns = next_states[idx]
            d = dones[idx]

            with torch.no_grad():
                next_a, next_logp = actor.sample(ns)
                alpha = log_alpha.exp()
                target_q = torch.min(target1(ns, next_a), target2(ns, next_a))
                y = r + args.gamma * (1.0 - d) * (target_q - alpha * next_logp)

            q1 = critic1(s, a)
            q2 = critic2(s, a)
            critic1_loss = F.mse_loss(q1, y)
            critic2_loss = F.mse_loss(q2, y)
            critic_loss = critic1_loss + critic2_loss

            critic1_opt.zero_grad()
            critic1_loss.backward()
            critic1_opt.step()

            critic2_opt.zero_grad()
            critic2_loss.backward()
            critic2_opt.step()

            pi, logp = actor.sample(s)
            q_pi = torch.min(critic1(s, pi), critic2(s, pi))
            bc_loss = F.mse_loss(pi, a)
            alpha = log_alpha.exp()
            actor_loss = (alpha * logp - q_pi).mean() + args.bc_weight * bc_loss

            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            soft_update(target1, critic1, args.tau)
            soft_update(target2, critic2, args.tau)

            if update % 100 == 0 or update == 1:
                row = {
                    "update": update,
                    "critic_loss": round(float(critic_loss.detach().cpu()), 6),
                    "actor_loss": round(float(actor_loss.detach().cpu()), 6),
                    "alpha_loss": round(float(alpha_loss.detach().cpu()), 6),
                    "alpha": round(float(log_alpha.exp().detach().cpu()), 6),
                    "bc_loss": round(float(bc_loss.detach().cpu()), 6),
                }
                writer.writerow(row)
                f.flush()
                print(row)

    actor.eval()
    with torch.no_grad():
        pred_norm = actor.deterministic(states).cpu().numpy()
    pred_actions = norm_to_action(pred_norm)
    behavior_actions = norm_to_action(actions.cpu().numpy())

    report_rows = []
    for i, name in enumerate(ACTION_FEATURES):
        mse = float(np.mean((pred_actions[:, i] - behavior_actions[:, i]) ** 2))
        report_rows.append({
            "action": name,
            "behavior_mean": round(float(behavior_actions[:, i].mean()), 4),
            "policy_mean": round(float(pred_actions[:, i].mean()), 4),
            "policy_min": round(float(pred_actions[:, i].min()), 4),
            "policy_max": round(float(pred_actions[:, i].max()), 4),
            "mse_to_behavior": round(mse, 6),
        })

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(out_dir / "sac_policy_report.csv", index=False)

    metadata = {
        "csv": args.csv,
        "rows": int(len(valid)),
        "device": str(device),
        "state_features": STATE_FEATURES,
        "action_features": ACTION_FEATURES,
        "action_low": ACTION_LOW.tolist(),
        "action_high": ACTION_HIGH.tolist(),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "gamma": args.gamma,
        "tau": args.tau,
        "bc_weight": args.bc_weight,
        "algorithm": "offline_sac_with_behavior_cloning_regularization",
    }

    checkpoint = {
        "actor_state_dict": actor.state_dict(),
        "critic1_state_dict": critic1.state_dict(),
        "critic2_state_dict": critic2.state_dict(),
        "metadata": metadata,
    }

    torch.save(checkpoint, out_dir / "sac_checkpoint.pt")
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "metadata": metadata,
        },
        out_dir / "sac_actor.pt",
    )

    with open(out_dir / "sac_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved: {out_dir / 'sac_actor.pt'}")
    print(f"Saved: {out_dir / 'sac_checkpoint.pt'}")
    print(f"Saved: {out_dir / 'sac_policy_report.csv'}")
    print(report_df.to_string(index=False))


if __name__ == "__main__":
    main()
