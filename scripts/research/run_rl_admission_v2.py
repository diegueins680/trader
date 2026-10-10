"""Opt-in registered v2 development campaign; no data fetching or live client."""
import argparse
from pathlib import Path
import rl_admission_v2 as v2
from run_rl_composition import ROOT, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-registered-development-v2", action="store_true")
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--settlements", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run_registered_development_v2 or not all((args.panel, args.settlements, args.output)):
        parser.error("disabled: explicit v2 development opt-in and registered paths required")
    run(args.panel, args.settlements, args.output,
        registration=ROOT/"research-notes/registrations/rl-admission-continuation-v2.json",
        episode_type=v2.Episode, trainer=v2.train, version=v2.VERSION,
        observation=v2.OBSERVATION,
        extra_sources=("rl_admission_v2.py", "run_rl_admission_v2.py"))
