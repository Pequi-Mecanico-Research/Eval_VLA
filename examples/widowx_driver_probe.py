"""Read-only probe of the WidowX AI driver (`trossen_arm`).

Shows what the driver already exposes (joint limits, errors, external/compensation efforts, ...) so we can
decide how to compute collision, constraint-violation, unsafe-contact and reset-failure rates from it.

The script never changes the arm mode and never sends commands. Keep the arm supported/in a safe state:
the mode it is in after `configure()` is whatever the driver defaults to.

    python examples/widowx_driver_probe.py --ip 192.168.1.2 --duration 10 --hz 50
    # move the arm by hand (or push it lightly) while it samples to see how the efforts react.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trossen_arm


def to_plain(obj, depth: int = 0):
    """Turns driver objects (enums, structs, lists) into JSON-friendly values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_plain(x, depth) for x in obj]
    if hasattr(obj, "name") and hasattr(obj, "value"):  # pybind11 enum
        return str(obj)
    if depth > 3:
        return str(obj)
    attrs = [a for a in dir(obj) if not a.startswith("_") and not callable(getattr(obj, a, None))]
    if not attrs:
        return str(obj)
    return {a: to_plain(getattr(obj, a), depth + 1) for a in attrs}


def list_driver_api(driver) -> None:
    print("\n=== Public methods/attributes of the driver ===")
    for name in sorted(n for n in dir(driver) if not n.startswith("_")):
        print(" ", name)


def dump_getters(driver) -> dict:
    """Calls every zero-argument `get_*` method (read-only by convention) and prints what it returns."""
    print("\n=== get_* methods (zero args) ===")
    dump = {}
    for name in sorted(n for n in dir(driver) if n.startswith("get_")):
        try:
            value = to_plain(getattr(driver, name)())
        except TypeError:
            print(f"  {name}: needs arguments, skipped")
            continue
        except Exception as exc:
            print(f"  {name}: failed ({exc})")
            continue
        dump[name] = value
        print(f"  {name}: {json.dumps(value, default=str)[:300]}")
    return dump


def describe_robot_output(driver) -> None:
    out = driver.get_robot_output()
    print("\n=== RobotOutput structure ===")
    print(json.dumps(to_plain(out), indent=2, default=str)[:3000])


def sample(driver, log: list[dict]) -> None:
    out = driver.get_robot_output()
    j = out.joint.all
    log.append({
        "t": out.header.timestamp,        # µs since configuration
        "pos": list(j.positions),
        "vel": list(j.velocities),
        "eff": list(j.efforts),
        "ext": list(j.external_efforts),
        "comp": list(j.compensation_efforts),
        "cart": list(out.cartesian.positions),
        # Same quantities read through the driver's dedicated getters.
        "ext_all": list(driver.get_all_external_efforts()),
        "comp_all": list(driver.get_all_compensation_efforts()),
        "cart_ext": list(driver.get_cartesian_external_efforts()),
    })


def print_effort_getters(driver, label: str = "") -> None:
    print(f"{label}get_all_external_efforts()      :", np.round(list(driver.get_all_external_efforts()), 3))
    print(f"{label}get_all_compensation_efforts()  :", np.round(list(driver.get_all_compensation_efforts()), 3))
    print(f"{label}get_cartesian_external_efforts():", np.round(list(driver.get_cartesian_external_efforts()), 3))


def summarize(log: list[dict]) -> None:
    print(f"\n=== Summary of {len(log)} samples ===")
    for key in ("pos", "vel", "eff", "ext", "comp", "cart", "ext_all", "comp_all", "cart_ext"):
        data = np.array([s[key] for s in log], dtype=float)
        print(f"\n{key}  (columns = joints / cartesian axes)")
        print("  min :", np.round(data.min(axis=0), 3))
        print("  max :", np.round(data.max(axis=0), 3))
        print("  mean:", np.round(data.mean(axis=0), 3))
    t = np.array([s["t"] for s in log], dtype=float) * 1e-6
    if len(t) > 1:
        print(f"\nEffective sampling rate: {(len(t) - 1) / (t[-1] - t[0]):.1f} Hz")


def connect_driver(ip: str = "192.168.1.2") -> trossen_arm.TrossenArmDriver:
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_base, ip, False)
    return driver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="192.168.1.2")
    parser.add_argument("--duration", type=float, default=10.0, help="seconds of sampling")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--out", default="eval_results/driver_probe")
    args = parser.parse_args()

    driver = connect_driver(args.ip)

    try:
        list_driver_api(driver)
        dump = dump_getters(driver)
        describe_robot_output(driver)
        print("\n=== Effort getters (single reading) ===")
        print_effort_getters(driver)

        input(f"\nSampling for {args.duration}s at {args.hz} Hz. Move the arm by hand if you want, press Enter to start...")
        log: list[dict] = []
        period = 1 / args.hz
        end = time.perf_counter() + args.duration
        next_print = 0.0
        while time.perf_counter() < end:
            tick = time.perf_counter()
            sample(driver, log)
            if tick >= next_print:  # live readout ~2x per second
                print_effort_getters(driver, label=f"[{len(log)}] ")
                next_print = tick + 0.5
            time.sleep(max(period - (time.perf_counter() - tick), 0))

        summarize(log)

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "getters.json").write_text(json.dumps(dump, indent=2, default=str))
        (out_dir / "samples.json").write_text(json.dumps(log))
        print(f"\nSaved getters.json and samples.json to {out_dir}")
    finally:
        driver.cleanup() if hasattr(driver, "cleanup") else None


if __name__ == "__main__":
    main()
