from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco


MODEL_PATH = Path(__file__).with_name("pendulum.xml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny MuJoCo pendulum simulation.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Simulation duration.")
    parser.add_argument("--sample-dt", type=float, default=0.2, help="Printed sample interval.")
    parser.add_argument("--initial-angle", type=float, default=35.0, help="Initial hinge angle in degrees.")
    parser.add_argument("--torque", type=float, default=0.0, help="Sinusoidal motor torque amplitude.")
    parser.add_argument("--frequency", type=float, default=1.0, help="Sinusoidal motor frequency in Hz.")
    parser.add_argument("--viewer", action="store_true", help="Open the interactive MuJoCo viewer.")
    return parser


def control_signal(t: float, torque: float, frequency: float) -> float:
    return torque * math.sin(2.0 * math.pi * frequency * t)


def make_sim(initial_angle_deg: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    hinge = model.joint("hinge")
    data.qpos[hinge.qposadr[0]] = math.radians(initial_angle_deg)
    mujoco.mj_forward(model, data)
    return model, data


def run_text_demo(args: argparse.Namespace) -> None:
    model, data = make_sim(args.initial_angle)
    bob_id = model.body("bob").id
    dt = model.opt.timestep
    steps = max(1, int(args.seconds / dt))
    sample_every = max(1, int(args.sample_dt / dt))

    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Model: nq={model.nq}, nv={model.nv}, nu={model.nu}, dt={dt:.4f}s")
    print("time_s angle_deg qvel_rad_s bob_z ctrl")

    for step in range(steps + 1):
        if model.nu:
            data.ctrl[0] = control_signal(data.time, args.torque, args.frequency)

        if step % sample_every == 0 or step == steps:
            angle_deg = math.degrees(data.qpos[0])
            qvel = data.qvel[0]
            bob_z = data.xpos[bob_id, 2]
            ctrl = data.ctrl[0] if model.nu else 0.0
            print(f"{data.time:6.3f} {angle_deg:9.3f} {qvel:10.4f} {bob_z:6.3f} {ctrl:7.4f}")

        mujoco.mj_step(model, data)


def run_viewer_demo(args: argparse.Namespace) -> None:
    import mujoco.viewer

    model, data = make_sim(args.initial_angle)
    deadline = args.seconds

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < deadline:
            step_start = time.time()
            if model.nu:
                data.ctrl[0] = control_signal(data.time, args.torque, args.frequency)
            mujoco.mj_step(model, data)
            viewer.sync()

            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


def main() -> None:
    args = build_parser().parse_args()
    if args.viewer:
        run_viewer_demo(args)
    else:
        run_text_demo(args)


if __name__ == "__main__":
    main()
