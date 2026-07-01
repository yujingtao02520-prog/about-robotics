from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Demo:
    xml_path: Path
    body_name: str
    joint_name: str
    default_seconds: float


DEMOS = {
    "box": Demo(HERE / "stage1_free_fall_box.xml", "box", "box_free", 1.0),
    "ball": Demo(HERE / "stage1_rolling_ball.xml", "ball", "ball_free", 2.0),
    "pendulum": Demo(HERE / "pendulum.xml", "bob", "hinge", 2.0),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 1: MuJoCo model/data/mj_step exercises.")
    parser.add_argument("demo", choices=DEMOS, help="Which small physics scene to run.")
    parser.add_argument("--seconds", type=float, default=None, help="Simulation duration.")
    parser.add_argument("--sample-dt", type=float, default=0.1, help="How often to print state.")
    parser.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer.")
    parser.add_argument("--speed", type=float, default=1.2, help="Initial x velocity for the ball demo.")
    parser.add_argument("--initial-angle", type=float, default=35.0, help="Initial pendulum angle in degrees.")
    return parser


def make_model_and_data(demo_name: str, args: argparse.Namespace) -> tuple[mujoco.MjModel, mujoco.MjData]:
    demo = DEMOS[demo_name]

    # model: immutable-ish description loaded from MJCF.
    model = mujoco.MjModel.from_xml_path(str(demo.xml_path))

    # data: mutable simulation state owned by one running simulation.
    data = mujoco.MjData(model)

    if demo_name == "ball":
        joint = model.joint(demo.joint_name)
        dof_start = joint.dofadr[0]
        data.qvel[dof_start + 0] = args.speed
    elif demo_name == "pendulum":
        joint = model.joint(demo.joint_name)
        qpos_start = joint.qposadr[0]
        data.qpos[qpos_start] = math.radians(args.initial_angle)

    mujoco.mj_forward(model, data)
    return model, data


def print_header(demo_name: str, model: mujoco.MjModel) -> None:
    print(f"Demo: {demo_name}")
    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"model.nq={model.nq}, model.nv={model.nv}, model.nu={model.nu}, timestep={model.opt.timestep:.4f}s")
    print()

    if demo_name in {"box", "ball"}:
        print("qpos for a free joint = [x, y, z, qw, qx, qy, qz]")
        print("qvel for a free joint = [vx, vy, vz, wx, wy, wz]")
        print("time_s  body_x  body_y  body_z      qpos_xyz            qvel_linear         qvel_angular")
    else:
        print("qpos for a hinge joint = [angle_rad]")
        print("qvel for a hinge joint = [angle_velocity_rad_s]")
        print("time_s  angle_deg  angle_vel_rad_s  bob_x  bob_y  bob_z")


def print_sample(demo_name: str, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    demo = DEMOS[demo_name]
    body_id = model.body(demo.body_name).id
    joint = model.joint(demo.joint_name)

    if demo_name in {"box", "ball"}:
        qpos_start = joint.qposadr[0]
        dof_start = joint.dofadr[0]
        qpos_xyz = data.qpos[qpos_start : qpos_start + 3]
        qvel_linear = data.qvel[dof_start : dof_start + 3]
        qvel_angular = data.qvel[dof_start + 3 : dof_start + 6]
        body_pos = data.xpos[body_id]

        print(
            f"{data.time:6.3f} "
            f"{body_pos[0]:7.3f} {body_pos[1]:7.3f} {body_pos[2]:7.3f}  "
            f"[{qpos_xyz[0]:6.3f} {qpos_xyz[1]:6.3f} {qpos_xyz[2]:6.3f}]  "
            f"[{qvel_linear[0]:6.3f} {qvel_linear[1]:6.3f} {qvel_linear[2]:6.3f}]  "
            f"[{qvel_angular[0]:6.3f} {qvel_angular[1]:6.3f} {qvel_angular[2]:6.3f}]"
        )
    else:
        qpos_start = joint.qposadr[0]
        dof_start = joint.dofadr[0]
        angle_deg = math.degrees(data.qpos[qpos_start])
        angle_vel = data.qvel[dof_start]
        bob_pos = data.xpos[body_id]
        print(
            f"{data.time:6.3f} "
            f"{angle_deg:10.3f} {angle_vel:16.4f}  "
            f"{bob_pos[0]:6.3f} {bob_pos[1]:6.3f} {bob_pos[2]:6.3f}"
        )


def run_text_demo(demo_name: str, args: argparse.Namespace) -> None:
    model, data = make_model_and_data(demo_name, args)
    seconds = args.seconds if args.seconds is not None else DEMOS[demo_name].default_seconds
    steps = max(1, int(seconds / model.opt.timestep))
    sample_every = max(1, int(args.sample_dt / model.opt.timestep))

    print_header(demo_name, model)
    for step in range(steps + 1):
        if step % sample_every == 0 or step == steps:
            print_sample(demo_name, model, data)

        # mj_step is the physics update: it mutates data according to model.
        mujoco.mj_step(model, data)


def run_viewer_demo(demo_name: str, args: argparse.Namespace) -> None:
    import mujoco.viewer

    model, data = make_model_and_data(demo_name, args)
    seconds = args.seconds if args.seconds is not None else DEMOS[demo_name].default_seconds

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < seconds:
            step_start = time.time()
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
