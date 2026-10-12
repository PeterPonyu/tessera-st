#!/usr/bin/env python3
"""Operational one-thread invocation of the unchanged pilot implementation."""
from datetime import datetime, timezone
from pathlib import Path
import torch

import run_pilot as pilot
from common import sha, dump


def main():
    torch.set_num_threads(1)
    pilot.OUT.mkdir(exist_ok=True)
    dump(pilot.OUT / 'execution.json', dict(
        started_at_utc=datetime.now(timezone.utc).isoformat(), device='cpu',
        torch_num_threads=torch.get_num_threads(),
        invocation_script_sha256=sha(Path(__file__)),
        implementation_sha256=sha(pilot.HERE / 'run_pilot.py'),
        runtime_decision_sha256=sha(pilot.HERE / 'runtime-decision.json'),
        fit_arm_seeds=pilot.SEEDS, fit_arms=['Tessera', 'NoGate'],
        partial_fourthread_results='runtime-fourthread-partial/; excluded from final results'))
    for section in pilot.SECTIONS:
        pilot.run(section)
    pilot.summarize()


if __name__ == '__main__':
    main()
