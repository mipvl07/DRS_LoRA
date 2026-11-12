import argparse
import socket
import struct
import time
import numpy as np
from munch import Munch
import msgpack
import msgpack_numpy as m_np

m_np.patch()

HEADER     = struct.Struct(">I")
N_AGENTS   = 4
RNN_LAYERS = 1
RNN_HIDDEN = 32

BSK_ENV_ARGS = {
    "key":                  "het_cloud_cluster-hard-random_rw",
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

def send_msg(sock, payload):
    data = msgpack.packb(payload, default=m_np.encode, use_bin_type=True)
    sock.sendall(HEADER.pack(len(data)))
    sock.sendall(data)

def recv_msg(sock):
    raw = b""
    while len(raw) < HEADER.size:
        raw += sock.recv(HEADER.size - len(raw))
    length = HEADER.unpack(raw)[0]
    data = b""
    while len(data) < length:
        data += sock.recv(length - len(data))
    return msgpack.unpackb(data, object_hook=m_np.decode, raw=False)

def make_env(env_key, seed=0):
    from harl.utils.envs_tools import make_train_env
    env_args = dict(BSK_ENV_ARGS)
    env_args["key"] = env_key
    print(f"[server] loading BSK env  key={env_key}")
    env = make_train_env("bsk", seed, 1, Munch(env_args))
    print(f"[server] env ready")
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

def parse_dones(dones):
    return bool(np.any(np.array(dones)))

def run(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60)
    print(f"[server] connecting to Jetson {args.jetson_ip}:{args.port} ...")
    sock.connect((args.jetson_ip, args.port))
    print(f"[server] connected")

    env = make_env(args.env_key, seed=args.seed)
    episode_rewards = []

    for ep in range(args.n_episodes):
        reset_result = env.reset()
        obs_arr   = parse_obs(reset_result[0])
        avail_arr = parse_avail(reset_result[2] if len(reset_result) > 2 else None)
        rnn_states = np.zeros((N_AGENTS, RNN_LAYERS, RNN_HIDDEN), dtype=np.float32)
        masks      = np.ones((N_AGENTS, 1), dtype=np.float32)
        ep_reward  = 0.0
        step       = 0
        done       = False

        print(f"\n[server] -- Episode {ep+1}/{args.n_episodes} --")

        while not done:
            msg = {
                "step":          step,
                "episode":       ep,
                "obs":           obs_arr.astype(np.float32),
                "rnn_states":    rnn_states,
                "masks":         masks,
                "avail_actions": (avail_arr.astype(np.float32)
                                  if avail_arr is not None
                                  else np.ones((N_AGENTS, 11), dtype=np.float32)),
            }

            t0 = time.perf_counter()
            send_msg(sock, msg)
            reply = recv_msg(sock)
            rtt_ms = (time.perf_counter() - t0) * 1000

            actions    = reply["actions"]
            rnn_states = reply["rnn_states"]
            latency_ms = reply["latency_ms"]

            print(f"  step={step:03d}  actions={actions}  rtt={rtt_ms:.1f}ms  jetson={latency_ms:.1f}ms")

            step_result = env.step(actions.reshape(1, N_AGENTS))
            obs_arr   = parse_obs(step_result[0])
            rewards   = np.array(step_result[2])
            done      = parse_dones(step_result[3])
            avail_arr = parse_avail(step_result[5] if len(step_result) > 5 else None)
            masks     = (np.zeros((N_AGENTS, 1), dtype=np.float32) if done
                         else np.ones((N_AGENTS, 1), dtype=np.float32))
            ep_reward += float(rewards.mean())
            step += 1

        print(f"[server] episode {ep+1} done  reward={ep_reward:.3f}  steps={step}")
        episode_rewards.append(ep_reward)

    send_msg(sock, {"step": -1, "obs": np.array([])})
    sock.close()
    env.close()

    print(f"\n[server] -- Summary --")
    print(f"  mean reward : {np.mean(episode_rewards):.3f}")
    print(f"  rewards     : {[round(r,2) for r in episode_rewards]}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--jetson_ip",  type=str, default="10.209.35.126")
    p.add_argument("--port",       type=int, default=6000)
    p.add_argument("--env_key",    type=str, default="het_cloud_cluster-hard-random_rw")
    p.add_argument("--n_episodes", type=int, default=5)
    p.add_argument("--seed",       type=int, default=0)
    args = p.parse_args()
    run(args)
