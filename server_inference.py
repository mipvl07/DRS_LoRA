"""
server_inference.py  —  run on SERVER
=======================================
Runs BSK env + HAPPO inference on the same machine.
Prints a metrics summary table at the end matching:
  Latency per step | RAM usage | CPU % | GPU mem | Power draw
"""

import argparse
import json
import os
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
from munch import Munch

OBS_DIM    = 39
ACTION_DIM = 11
N_AGENTS   = 4
HIDDEN     = [32, 32]
RNN_HIDDEN = 32
RNN_LAYERS = 1

BSK_ENV_ARGS = {
    "key":                  "het_cloud_AIS_cluster_D5-hard-random_rw",  #"het_cloud_cluster-hard-random_rw"
    "n_satellites":         4,
    "n_act_image":          4,
    "n_obs_image":          4,
    "battery_capacity":     400,
    "memory_size":          5000,
    "baud_rate":            100,
    "instr_baud_rate":      500,
    "init_battery_level":   100,
    "init_memory_percent":  0,
    "power_reward":         True,
    "battery_cost_scale":   10.0,
    "data_reward":          True,
    "data_cost_scale":      0.5,
    "orbit_num":            0.25,
    "uniform_targets":      2000,
    "randomize_enabled":    False,
    "failure_penalty":      -100,
    "use_render":           False,
}


class MLPBase(nn.Module):
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        self.feature_norm = nn.LayerNorm(obs_dim)
        layers, in_dim = [], obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(self.feature_norm(x))


class RNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.rnn  = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, hxs, masks):
        x   = x.unsqueeze(1)
        hxs = hxs * (1 - masks).unsqueeze(0)
        out, hxs = self.rnn(x, hxs.contiguous())
        return self.norm(out.squeeze(1)), hxs


class HARLActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = MLPBase(OBS_DIM, HIDDEN)
        self.rnn  = RNNLayer(HIDDEN[-1], RNN_HIDDEN, RNN_LAYERS)
        self.act  = nn.Linear(RNN_HIDDEN, ACTION_DIM)

    def forward(self, obs, rnn_state, mask, avail_actions=None):
        feat, rnn_state = self.rnn(self.base(obs), rnn_state, mask)
        logits = self.act(feat)
        if avail_actions is not None:
            logits[avail_actions == 0] = -1e10
        return logits.argmax(dim=-1), rnn_state


def load_actors(model_dir, device):
    actors = []
    for i in range(N_AGENTS):
        path = os.path.join(model_dir, f"actor_agent{i}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing: {path}\nFiles found: {os.listdir(model_dir)}"
            )
        sd = torch.load(path, map_location=device, weights_only=True)
        if i == 0:
            print("[server_inference] checkpoint keys (agent 0):")
            for k, v in sd.items():
                print(f"  {k:50s} {tuple(v.shape)}")
        remapped = {
            k.replace("act.action_out.", "act."): v for k, v in sd.items()
        }
        model = HARLActor().to(device)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing:    print(f"  WARNING missing keys    agent{i}: {missing}")
        if unexpected: print(f"  WARNING unexpected keys agent{i}: {unexpected}")
        model.eval()
        actors.append(model)
        print(f"[server_inference] actor_agent{i}.pt loaded  ✓")
    return actors


def make_env(env_key, seed=0):
    from harl.utils.envs_tools import make_train_env
    env_args = dict(BSK_ENV_ARGS)
    env_args["key"] = env_key
    print(f"[server_inference] loading BSK env  key={env_key}")
    env = make_train_env("bsk", seed, 1, Munch(env_args))
    print(f"[server_inference] env ready")
    return env


def parse_obs(obs):
    if isinstance(obs, (list, tuple)):
        return np.stack([np.array(obs[a])[0] for a in range(N_AGENTS)], axis=0)
    obs = np.array(obs)
    return obs[0] if obs.ndim == 3 else obs


def parse_avail(avail):
    if avail is None:
        return None
    if isinstance(avail, (list, tuple)):
        return np.stack([np.array(avail[a])[0] for a in range(N_AGENTS)], axis=0)
    avail = np.array(avail)
    return avail[0] if avail.ndim == 3 else avail


def parse_rewards(rewards):
    # VecEnv returns (n_threads=1, n_agents, 1) — squeeze to scalar mean
    return float(np.array(rewards).squeeze().mean())


def parse_dones(dones):
    return bool(np.any(np.array(dones)))


class MetricsLogger:
    def __init__(self, path):
        self.path    = path
        self.process = psutil.Process(os.getpid())

    def log(self, step, ep, latency_ms, env_step_ms):
        mem = self.process.memory_info()
        entry = {
            "ts":          time.time(),
            "episode":     ep,
            "step":        step,
            "latency_ms":  latency_ms,
            "env_step_ms": env_step_ms,
            "rss_mb":      mem.rss / 1e6,
            "cpu_pct":     self.process.cpu_percent(interval=None),
        }
        if torch.cuda.is_available():
            entry["gpu_alloc_mb"] = torch.cuda.memory_allocated() / 1e6
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry


@torch.no_grad()
def infer(actors, obs_np, rnn_np, masks_np, avail_np, device):
    actions_out = np.zeros(N_AGENTS, dtype=np.int32)
    rnn_out     = np.zeros_like(rnn_np)
    t0 = time.perf_counter()
    for i, actor in enumerate(actors):
        obs   = torch.tensor(obs_np[i:i+1],  dtype=torch.float32, device=device)
        rnn   = torch.tensor(rnn_np[i],       dtype=torch.float32, device=device).unsqueeze(1)
        mask  = torch.tensor(masks_np[i:i+1], dtype=torch.float32, device=device)
        avail = torch.tensor(avail_np[i:i+1], dtype=torch.float32, device=device)
        action, new_rnn = actor(obs, rnn, mask, avail)
        actions_out[i] = action.item()
        rnn_out[i]     = new_rnn.squeeze(1).cpu().numpy()
    return actions_out, rnn_out, (time.perf_counter() - t0) * 1000


def print_metrics_table(all_entries, episode_rewards):
    latencies  = [e["latency_ms"]  for e in all_entries]
    rss        = [e["rss_mb"]      for e in all_entries]
    cpu        = [e["cpu_pct"]     for e in all_entries]
    gpu        = [e.get("gpu_alloc_mb", None) for e in all_entries]
    gpu_vals   = [g for g in gpu if g is not None]

    gpu_str    = f"{np.mean(gpu_vals):.1f} MB" if gpu_vals else "—  (no GPU)"

    print("\n" + "═" * 44)
    print("  SERVER INFERENCE  —  METRICS SUMMARY")
    print("═" * 44)
    print(f"  {'Latency per step':<22} {np.mean(latencies):.2f} ms  (min {np.min(latencies):.2f}  max {np.max(latencies):.2f})")
    print(f"  {'RAM usage':<22} {np.mean(rss):.1f} MB  (peak {np.max(rss):.1f} MB)")
    print(f"  {'CPU %':<22} {np.mean(cpu):.1f} %")
    print(f"  {'GPU mem':<22} {gpu_str}")
    print(f"  {'Power draw':<22} —  (use jtop on Jetson)")
    print("─" * 44)
    print(f"  {'Episodes':<22} {len(episode_rewards)}")
    print(f"  {'Mean reward':<22} {np.mean(episode_rewards):.3f}")
    print(f"  {'Per-episode rewards':<22} {[round(r,2) for r in episode_rewards]}")
    print("═" * 44 + "\n")


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[server_inference] using device: {device}")

    actors  = load_actors(args.model_dir, device)
    logger  = MetricsLogger(args.metrics_out)
    env     = make_env(args.env_key, seed=args.seed)

    episode_rewards = []
    all_entries     = []   # collect every logged entry for summary

    for ep in range(args.n_episodes):

        reset_result = env.reset()
        # HARL reset → (obs, share_obs, available_actions)
        obs_arr   = parse_obs(reset_result[0])
        avail_arr = parse_avail(reset_result[2] if len(reset_result) > 2 else None)

        rnn_states = np.zeros((N_AGENTS, RNN_LAYERS, RNN_HIDDEN), dtype=np.float32)
        masks      = np.ones((N_AGENTS, 1), dtype=np.float32)

        ep_reward  = 0.0
        step       = 0
        done       = False

        print(f"\n[server_inference] ── Episode {ep+1}/{args.n_episodes} ──")

        while not done:
            avail_np = (avail_arr.astype(np.float32)
                        if avail_arr is not None
                        else np.ones((N_AGENTS, ACTION_DIM), dtype=np.float32))

            actions, rnn_states, latency_ms = infer(
                actors, obs_arr, rnn_states, masks, avail_np, device
            )

            # HARL VecEnv step → (obs, share_obs, rewards, dones, infos, available_actions)
            t_env = time.perf_counter()
            step_result = env.step(actions.reshape(1, N_AGENTS))
            env_step_ms = (time.perf_counter() - t_env) * 1000

            obs_arr   = parse_obs(step_result[0])
            # step_result[1] = share_obs — not needed for inference
            ep_reward += parse_rewards(step_result[2])
            done       = parse_dones(step_result[3])
            # step_result[4] = infos
            avail_arr  = parse_avail(step_result[5] if len(step_result) > 5 else None)

            masks = (np.zeros((N_AGENTS, 1), dtype=np.float32) if done
                     else np.ones((N_AGENTS, 1), dtype=np.float32))

            entry = logger.log(step=step, ep=ep,
                               latency_ms=latency_ms, env_step_ms=env_step_ms)
            all_entries.append(entry)

            print(f"  step={step:03d}  actions={actions}  "
                  f"inference={latency_ms:.2f}ms  env_step={env_step_ms:.1f}ms  "
                  f"ram={entry['rss_mb']:.0f}MB  cpu={entry['cpu_pct']:.0f}%")

            step += 1

        print(f"[server_inference] episode {ep+1} done  "
              f"reward={ep_reward:.3f}  steps={step}")
        episode_rewards.append(ep_reward)

    env.close()

    print_metrics_table(all_entries, episode_rewards)
    print(f"[server_inference] metrics saved → {args.metrics_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",   type=str, required=True)
    p.add_argument("--env_key",     type=str, default="het_cloud_cluster-hard-random_rw")
    p.add_argument("--n_episodes",  type=int, default=5)
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--metrics_out", type=str, default="metrics_server.jsonl")
    args = p.parse_args()
    run(args)
