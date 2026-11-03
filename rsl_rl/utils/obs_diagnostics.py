"""Rollout diagnostics: find extreme observations, actions and rewards.

The runner calls ``record_step`` after each env step and ``log`` one time per
iteration. All per-step operations stay on the GPU. A CPU sync occurs only if
an env becomes extreme and the event limit is not reached.

Scalars (per iteration):
  - Diag/obs_max/<group>/<term>: max abs raw value of the term (all envs, all
    rollout steps, all history frames). Groups: actor, critic, amp (env obs
    before the empirical normalization), actions (policy output before the
    env clip), rewards (reward manager step reward, weighted and not scaled
    by dt; "total" is the env reward divided by dt).
  - Diag/num_extreme_envs: number of envs that were extreme at one or more
    rollout steps. An env is extreme if a value is larger than the threshold
    of its group or is not finite.
  - Diag/max_joint_vel, Diag/max_ball_speed, Diag/max_contact_force: max abs
    robot joint velocity, max ball speed (world) and max contact force norm
    of all contact sensors (all envs, all rollout steps).
  - Diag/extreme_events_written: number of lines in extreme_states.jsonl.

extreme_states.jsonl: one JSON line when an env becomes extreme (the step
before it was not extreme). The env state is read after ``env.step`` returns.
If the env was reset in this step ("done": true), the sim state and the obs
are of the new episode, but the step rewards are of the old episode.
"""

from __future__ import annotations

import json
import math
import os

import torch


class ObsDiagnostics:
    def __init__(
        self,
        env,
        log_dir: str | None,
        obs_threshold: float,
        action_threshold: float,
        reward_threshold: float,
        max_events: int = 200,
    ):
        self.env = env.unwrapped
        self.device = self.env.device
        self.num_envs = self.env.num_envs
        self.max_events = max_events
        self.events_written = 0
        self.path = os.path.join(log_dir, "extreme_states.jsonl") if log_dir is not None else None

        # Column index of each term, for each group
        om = self.env.observation_manager
        self.layouts: dict[str, list[tuple[str, torch.Tensor]]] = {}
        for group in ("actor", "critic", "amp"):
            if group in om.active_terms:
                self.layouts[group] = self._obs_term_columns(om, group)
        am = self.env.action_manager
        self.layouts["actions"] = self._flat_term_columns(am.active_terms, am.action_term_dim)
        rm = self.env.reward_manager
        reward_names = list(rm.active_terms) + ["total"]
        self.layouts["rewards"] = self._flat_term_columns(reward_names, [1] * len(reward_names))
        self.thresholds = {group: obs_threshold for group in ("actor", "critic", "amp")}
        self.thresholds["actions"] = action_threshold
        self.thresholds["rewards"] = reward_threshold

        self.robot = self.env.scene["robot"]
        self.ball = self.env.scene.entities.get("ball")
        self.contact_sensors = {
            name: sensor
            for name, sensor in self.env.scene.sensors.items()
            if getattr(sensor.data, "found", None) is not None
        }
        self.prev_extreme = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # The last normalized actor inputs (newest last) and the raw actor input of
        # the current step. They show what the policy saw when it gave an extreme action.
        self.act_inputs: list[torch.Tensor] = []
        self.act_input_raw: torch.Tensor | None = None
        self.act_input_raw_used: torch.Tensor | None = None
        self.action_mean: torch.Tensor | None = None
        self._reset_iteration()

    def push_act_input(self, norm_obs: torch.Tensor):
        """Call before the policy acts, with the normalized actor input."""
        self.act_inputs.append(norm_obs.detach().clone())
        del self.act_inputs[:-3]

    @staticmethod
    def _flat_term_columns(names, dims) -> list[tuple[str, torch.Tensor]]:
        columns, start = [], 0
        for name, dim in zip(names, dims):
            columns.append((name, torch.arange(start, start + dim)))
            start += dim
        return columns

    @staticmethod
    def _obs_term_columns(om, group: str) -> list[tuple[str, torch.Tensor]]:
        """Column index of each term. A time-major history has one block per
        frame with all terms in it. A term-major history has all frames of a term
        in one block."""
        names = om.active_terms[group]
        dims = [tuple(int(d) for d in dim) for dim in om.group_obs_term_dim[group]]
        time_major = om.cfg[group].history_ordering == "time" and all(len(d) == 2 for d in dims)
        if not time_major:
            return ObsDiagnostics._flat_term_columns(names, [math.prod(d) for d in dims])
        history = dims[0][0]
        frame_dim = sum(d[1] for d in dims)
        columns, start = [], 0
        for name, (_, dim) in zip(names, dims):
            frame_cols = torch.arange(start, start + dim)
            columns.append((name, torch.cat([frame_cols + h * frame_dim for h in range(history)])))
            start += dim
        return columns

    def _reset_iteration(self):
        self.col_max: dict[str, torch.Tensor | None] = {group: None for group in self.layouts}
        self.extreme_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        zero = torch.zeros((), device=self.device)
        self.max_joint_vel = zero.clone()
        self.max_ball_speed = zero.clone()
        self.max_contact_force = zero.clone()

    def record_step(
        self,
        it: int,
        step: int,
        values: dict[str, torch.Tensor],
        dones: torch.Tensor,
        action_mean: torch.Tensor | None = None,
    ):
        """Record one rollout step. ``values`` has the raw tensors of the groups
        actor, critic, amp, actions (each [num_envs, dim]) and rewards ([num_envs]
        env reward). The function adds the reward manager terms."""
        values = dict(values)
        self.action_mean = action_mean
        # The raw actor input of this step's action is the actor obs of the step before.
        self.act_input_raw_used, self.act_input_raw = self.act_input_raw, values["actor"].detach().clone()
        rm = self.env.reward_manager
        values["rewards"] = torch.cat(
            [rm._step_reward, values["rewards"].reshape(-1, 1) / self.env.step_dt], dim=1
        )
        extreme = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for group, x in values.items():
            if group not in self.layouts:
                continue
            x_abs = x.detach().abs()
            col_max = x_abs.amax(dim=0)
            prev = self.col_max[group]
            self.col_max[group] = col_max if prev is None else torch.maximum(prev, col_max)
            env_max = x_abs.amax(dim=1)
            extreme |= (env_max > self.thresholds[group]) | ~torch.isfinite(env_max)
        self.extreme_any |= extreme

        # Earliest signs in the sim state
        self.max_joint_vel = torch.maximum(self.max_joint_vel, self.robot.data.joint_vel.abs().max())
        if self.ball is not None:
            ball_speed = torch.linalg.norm(self.ball.data.root_link_lin_vel_w, dim=-1).max()
            self.max_ball_speed = torch.maximum(self.max_ball_speed, ball_speed)
        for sensor in self.contact_sensors.values():
            force = sensor.data.force
            if force is not None:
                self.max_contact_force = torch.maximum(
                    self.max_contact_force, torch.linalg.norm(force, dim=-1).max()
                )

        # Write the env state when an env becomes extreme
        onset = extreme & ~self.prev_extreme
        self.prev_extreme = extreme
        if self.path is None or self.events_written >= self.max_events:
            return
        env_ids = onset.nonzero(as_tuple=False).flatten()[: self.max_events - self.events_written].tolist()
        if not env_ids:
            return
        with open(self.path, "a") as f:
            for env_id in env_ids:
                f.write(json.dumps(self._event(it, step, env_id, values, dones)) + "\n")
        self.events_written += len(env_ids)

    def _policy_input(self, env_id: int) -> dict:
        """Normalized actor input of the step that gave the action (and the two
        steps before), the 10 largest entries by term name, and the action mean."""
        out: dict = {}
        if not self.act_inputs:
            return out
        names = {}
        for name, cols in self.layouts["actor"]:
            for k, c in enumerate(cols.tolist()):
                names[c] = f"{name}[{k}]"
        rows = [x[env_id].float().cpu() for x in self.act_inputs]
        last = rows[-1]
        top = torch.topk(last.abs(), min(10, last.numel()))
        out["policy_input_top_norm"] = [
            {"term": names.get(int(i), str(int(i))), "norm_value": last[int(i)].item()}
            for i in top.indices
        ]
        out["policy_input_norm"] = [r.tolist() for r in rows]
        out["policy_input_term_names"] = [names.get(c, str(c)) for c in range(last.numel())]
        if self.act_input_raw_used is not None:
            out["policy_input_raw"] = self.act_input_raw_used[env_id].float().cpu().tolist()
        if self.action_mean is not None:
            out["action_mean"] = self.action_mean[env_id].detach().float().cpu().tolist()
        return out

    def _event(self, it: int, step: int, env_id: int, values: dict, dones: torch.Tensor) -> dict:
        offending = {}
        for group, x in values.items():
            if group not in self.layouts:
                continue
            row = x[env_id].detach().float().cpu()
            for name, cols in self.layouts[group]:
                term = row[cols]
                term_max = term.abs().max().item()
                if not math.isfinite(term_max) or term_max > self.thresholds[group]:
                    offending[f"{group}/{name}"] = {"max_abs": term_max, "values": term.tolist()}

        def to_list(t):
            return t[env_id].detach().float().cpu().tolist()

        data = self.robot.data
        rm = self.env.reward_manager
        event = {
            "iteration": it,
            "step": step,
            "env_id": env_id,
            "done": bool(dones.reshape(-1)[env_id].item()),
            "state_note": "read after env.step; post-reset state if done",
            "episode_length": int(self.env.episode_length_buf[env_id].item()),
            "offending": offending,
            "root_pos_w": to_list(data.root_link_pos_w),
            "root_lin_vel_w": to_list(data.root_link_lin_vel_w),
            "root_ang_vel_w": to_list(data.root_link_ang_vel_w),
            "joint_names": list(self.robot.joint_names),
            "joint_pos": to_list(data.joint_pos),
            "joint_vel": to_list(data.joint_vel),
            "actuator_force": to_list(data.actuator_force),
            "last_action": to_list(self.env.action_manager.action),
            "step_rewards": dict(zip(rm.active_terms, to_list(rm._step_reward))),
            "num_contacts": {
                name: int(sensor.data.found[env_id].sum().item()) for name, sensor in self.contact_sensors.items()
            },
            "max_contact_force": {
                name: torch.linalg.norm(sensor.data.force[env_id], dim=-1).max().item()
                for name, sensor in self.contact_sensors.items()
                if sensor.data.force is not None
            },
        }
        event.update(self._policy_input(env_id))
        if self.ball is not None:
            event["ball_pos_w"] = to_list(self.ball.data.root_link_pos_w)
            event["ball_lin_vel_w"] = to_list(self.ball.data.root_link_lin_vel_w)
        for attr in ("kick_timer", "kick_intensity", "kick_style", "kick_new_contact"):
            value = getattr(self.env, attr, None)
            if isinstance(value, torch.Tensor):
                event[attr] = value[env_id].item()
        return event

    def log(self, writer, it: int):
        """Write the scalars of this iteration, reset the accumulators and return
        the number of extreme envs."""
        for group, columns in self.layouts.items():
            col_max = self.col_max[group]
            if col_max is None:
                continue
            col_max = col_max.float().cpu()
            for name, cols in columns:
                writer.add_scalar(f"Diag/obs_max/{group}/{name}", col_max[cols].max().item(), it)
        num_extreme_envs = int(self.extreme_any.sum().item())
        writer.add_scalar("Diag/num_extreme_envs", num_extreme_envs, it)
        writer.add_scalar("Diag/max_joint_vel", self.max_joint_vel.item(), it)
        writer.add_scalar("Diag/max_ball_speed", self.max_ball_speed.item(), it)
        writer.add_scalar("Diag/max_contact_force", self.max_contact_force.item(), it)
        writer.add_scalar("Diag/extreme_events_written", self.events_written, it)
        self._reset_iteration()
        return num_extreme_envs
