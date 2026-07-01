from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco


HERE = Path(__file__).resolve().parent
L1 = 0.45
L2 = 0.35


@dataclass(frozen=True)
class Demo:
    xml_path: Path
    default_seconds: float


DEMOS = {
    "pendulum": Demo(HERE / "stage2_single_pendulum.xml", 3.0),
    "arm": Demo(HERE / "stage2_two_link_arm.xml", 3.0),
    "reach": Demo(HERE / "stage2_two_link_arm.xml", 4.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2: joint and actuator exercises.")
    parser.add_argument("demo", choices=DEMOS, help="pendulum, arm, or reach.")
    parser.add_argument("--seconds", type=float, default=None, help="Simulation duration.")
    parser.add_argument("--sample-dt", type=float, default=0.2, help="How often to print state.")
    parser.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer.")

    parser.add_argument("--pendulum-torque", type=float, default=0.0, help="Constant pendulum motor torque.")
    parser.add_argument("--pendulum-target", type=float, default=None, help="PD target angle in degrees.")

    parser.add_argument("--shoulder-torque", type=float, default=0.5, help="Manual shoulder torque for arm demo.")
    parser.add_argument("--elbow-torque", type=float, default=-0.2, help="Manual elbow torque for arm demo.")

    parser.add_argument("--target-x", type=float, default=0.55, help="Reach target x position.")
    parser.add_argument("--target-y", type=float, default=0.25, help="Reach target y position.")
    parser.add_argument("--kp", type=float, default=12.0, help="PD position gain for reach.")
    parser.add_argument("--kd", type=float, default=2.0, help="PD velocity gain for reach.")
    return parser


def joint_index(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint = model.joint(name)
    return joint.qposadr[0], joint.dofadr[0]


def clip_ctrl(model: mujoco.MjModel, actuator_id: int, value: float) -> float:
    low, high = model.actuator_ctrlrange[actuator_id]
    return max(float(low), min(float(high), value))


def ik_2link(x: float, y: float) -> tuple[float, float]:
    radius = math.hypot(x, y)
    max_radius = L1 + L2 - 1e-6
    if radius > max_radius:
        scale = max_radius / radius
        x *= scale
        y *= scale

    cos_elbow = (x * x + y * y - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    elbow = math.acos(cos_elbow)
    shoulder = math.atan2(y, x) - math.atan2(L2 * math.sin(elbow), L1 + L2 * math.cos(elbow))
    return shoulder, elbow


def make_model_and_data(demo_name: str, args: argparse.Namespace) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(DEMOS[demo_name].xml_path))
    data = mujoco.MjData(model)

    if demo_name == "pendulum":
        qpos, _ = joint_index(model, "hinge")
        data.qpos[qpos] = math.radians(25.0)
    else:
        target_id = model.site("target").id
        model.site_pos[target_id] = [args.target_x, args.target_y, 0.08]

    mujoco.mj_forward(model, data)
    return model, data


def set_controls(demo_name: str, model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace) -> None:
    if demo_name == "pendulum":
        qpos, dof = joint_index(model, "hinge")
        torque = args.pendulum_torque
        if args.pendulum_target is not None:
            target = math.radians(args.pendulum_target)
            gravity_comp = data.qfrc_bias[dof]
            torque += gravity_comp + 10.0 * (target - data.qpos[qpos]) - 1.5 * data.qvel[dof]
        data.ctrl[0] = clip_ctrl(model, 0, torque)
        return

    if demo_name == "arm":
        data.ctrl[0] = clip_ctrl(model, 0, args.shoulder_torque)
        data.ctrl[1] = clip_ctrl(model, 1, args.elbow_torque)
        return

    shoulder_qpos, shoulder_dof = joint_index(model, "shoulder")
    elbow_qpos, elbow_dof = joint_index(model, "elbow")
    shoulder_target, elbow_target = ik_2link(args.target_x, args.target_y)

    shoulder_torque = args.kp * (shoulder_target - data.qpos[shoulder_qpos]) - args.kd * data.qvel[shoulder_dof]
    elbow_torque = args.kp * (elbow_target - data.qpos[elbow_qpos]) - args.kd * data.qvel[elbow_dof]

    data.ctrl[0] = clip_ctrl(model, 0, shoulder_torque)
    data.ctrl[1] = clip_ctrl(model, 1, elbow_torque)


def print_header(demo_name: str, model: mujoco.MjModel) -> None:
    print(f"Demo: {demo_name}")
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"model.nq={model.nq}, model.nv={model.nv}, model.nu={model.nu}, timestep={model.opt.timestep:.4f}s")
    print("body + joint + geom define the mechanism; actuator maps data.ctrl to joint torque.")
    print()

    if demo_name == "pendulum":
        print("time_s angle_deg qvel_rad_s ctrl bob_x bob_z")
    else:
        print("time_s shoulder_deg elbow_deg ctrl0 ctrl1 ee_x ee_y target_x target_y error")


def print_sample(demo_name: str, model: mujoco.MjModel, data: mujoco.MjData, args: argparse.Namespace) -> None:
    if demo_name == "pendulum":
        qpos, dof = joint_index(model, "hinge")
        bob_id = model.body("bob").id
        bob_pos = data.xpos[bob_id]
        print(
            f"{data.time:6.3f} "
            f"{math.degrees(data.qpos[qpos]):9.3f} "
            f"{data.qvel[dof]:10.4f} "
            f"{data.ctrl[0]:6.3f} "
            f"{bob_pos[0]:6.3f} {bob_pos[2]:6.3f}"
        )
        return

    shoulder_qpos, _ = joint_index(model, "shoulder")
    elbow_qpos, _ = joint_index(model, "elbow")
    ee_id = model.site("end_effector").id
    ee_pos = data.site_xpos[ee_id]
    error = math.hypot(args.target_x - ee_pos[0], args.target_y - ee_pos[1])

    print(
        f"{data.time:6.3f} "
        f"{math.degrees(data.qpos[shoulder_qpos]):12.3f} "
        f"{math.degrees(data.qpos[elbow_qpos]):9.3f} "
        f"{data.ctrl[0]:5.2f} {data.ctrl[1]:5.2f} "
        f"{ee_pos[0]:5.3f} {ee_pos[1]:5.3f} "
        f"{args.target_x:8.3f} {args.target_y:8.3f} "
        f"{error:6.3f}"
    )


def run_text_demo(demo_name: str, args: argparse.Namespace) -> None:
    model, data = make_model_and_data(demo_name, args)
    seconds = args.seconds if args.seconds is not None else DEMOS[demo_name].default_seconds
    steps = max(1, int(seconds / model.opt.timestep))
    sample_every = max(1, int(args.sample_dt / model.opt.timestep))

    print_header(demo_name, model)
    for step in range(steps + 1):
        set_controls(demo_name, model, data, args)
        if step % sample_every == 0 or step == steps:
            print_sample(demo_name, model, data, args)
        mujoco.mj_step(model, data)


def run_viewer_demo(demo_name: str, args: argparse.Namespace) -> None:
    import mujoco.viewer

    model, data = make_model_and_data(demo_name, args)
    seconds = args.seconds if args.seconds is not None else DEMOS[demo_name].default_seconds

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < seconds:
            step_start = time.time()
            set_controls(demo_name, model, data, args)
            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main() -> None:
    args = build_parser().parse_args()
    if args.viewer:
        run_viewer_demo(args.demo, args)
    else:
        run_text_demo(args.demo, args)


if __name__ == "__main__":
    main()
