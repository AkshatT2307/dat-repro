"""Bounded overnight supervisor; decisions use validation, never generalization scores."""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
DATASETS = ('imagenet', 'marmousi2')
METHODS = ('idr', 'supervised')


def now():
    return datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(timespec='seconds')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def last_record(path):
    try:
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 16384))
            lines = f.read().splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except ValueError:
                continue
    except OSError:
        pass
    return {}


def run_dir(dataset, method, batch, epochs):
    return ROOT/'runs/epoch128'/dataset/method/f'b{batch}_e{epochs}'


def baseline(dataset):
    # Expected L1 of unmodified Gaussian-corrupted images, matching validation sigmas.
    sigma = (25 + 50) / (2 * 255) if dataset == 'imagenet' else (.1 + .2 + .4) / 3
    return math.sqrt(2 / math.pi) * sigma


def quality_reasons(dataset, summaries):
    """Conservative failure heuristics, not claims of paper-level quality."""
    reasons = []
    for method, summary in summaries.items():
        loss = summary.get('best_loss', math.inf)
        if not math.isfinite(loss) or loss >= .9 * baseline(dataset):
            reasons.append(f'{method}: best validation L1 {loss:.6g} fails to beat '
                           f'expected noisy-input L1 {baseline(dataset):.6g} by 10%')
    if all(m in summaries for m in METHODS):
        a, b = (summaries[m]['best_loss'] for m in METHODS)
        if math.isfinite(a) and math.isfinite(b) and a > 1.5 * b:
            reasons.append('Fast IDR validation L1 exceeds 1.5 times supervised L1')
    return reasons


def process_identity(pid):
    try:
        # Comm in /proc/PID/stat may contain spaces; fields after the closing ')'.
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, IndexError):
        return None


class Monitor:
    def __init__(self, session, interval=30):
        self.session, self.interval = session, interval
        self.events = ROOT/'logs/overnight_events.jsonl'
        self.configs = [(192, 20)]
        self.decisions = []
        self.selected = {}
        self.phase = 'Observing original four-experiment pipeline'

    def event(self, message):
        line = dict(time=now(), message=message)
        self.decisions.append(line)
        with self.events.open('a') as f:
            f.write(json.dumps(line) + '\n')
        print(f"{line['time']} | {message}", flush=True)

    def complete(self, dataset, batch, epochs):
        return all(read_json(run_dir(dataset, m, batch, epochs)/'training_summary.json')
                   and read_json(run_dir(dataset, m, batch, epochs)/'evaluation/summary.json')
                   for m in METHODS)

    def report(self):
        lines = ['# Overnight experiment status', '', f'Updated: {now()}', '', self.phase, '',
                 'Original plan: batch 192, up to 20 epochs, patience 5. Both methods use',
                 'the same ImageNet holdout or the separate seismic validation region.',
                 'Kodak, BSDS and seismic test scores never trigger adjustments.', '',
                 '| Dataset | Method | Batch / limit | Last step | Validated epochs | Best validation L1 | State |',
                 '|---|---|---|---|---|---|---|']
        for batch, epochs in self.configs:
            for dataset in DATASETS:
                for method in METHODS:
                    folder = run_dir(dataset, method, batch, epochs)
                    metric = last_record(folder/'metrics.jsonl')
                    val = last_record(folder/'validation.jsonl')
                    summary = read_json(folder/'training_summary.json')
                    evaluation = read_json(folder/'evaluation/summary.json')
                    state = ('training and evaluation complete' if summary and evaluation else
                             'training complete; evaluation pending' if summary else
                             'training started' if metric else 'not started')
                    if metric and not math.isfinite(metric.get('loss', math.inf)):
                        state += '; NONFINITE LOSS'
                    step = f"{metric.get('step', 0)}/{metric.get('total_steps', '?')}"
                    loss = val.get('best_loss')
                    loss = f'{loss:.6f}' if loss is not None else '-'
                    lines.append(f'| {dataset} | {method} | {batch} / {epochs} | {step} | '
                                 f"{val.get('epoch', 0)} | {loss} | {state} |")
        lines += ['', '## Decisions', '']
        lines += [f"- {d['time']}: {d['message']}" for d in self.decisions] or ['- No adjustments yet.']
        lines += ['', '## Review files', '',
                  '- Run folders: `runs/epoch128/DATASET/METHOD/bBATCH_eEPOCHS/`.',
                  '- `validation.jsonl`: validation curves; `training_summary.json`: best epoch and stopping state.',
                  '- `best.pt` / `best_weights.pth`: model selected using validation only.',
                  '- `evaluation/summary.json`: final generalization metrics.',
                  '- `runs/epoch128/DATASET/comparison_bBATCH_eEPOCHS/results.csv`: paired results.',
                  '- `logs/overnight_events.jsonl`: persistent decisions; `logs/tmux_latest.txt`: latest training pane.', '',
                  'Automatic recovery is bounded: at most one recovery run and one quality retry per dataset.',
                  'Quality retry triggers only for <10% L1 gain over expected noisy-input error, or Fast IDR',
                  'L1 >1.5x supervised. These are failure heuristics, not proof of paper-level performance.',
                  'A quality retry trains BOTH methods at batch 128 with up to 30 epochs in separate folders.',
                  'If original runs pass these checks, their batch and epoch limit stay unchanged.', '',
                  'Stop future automatic actions: `kill -INT $(cat repro/logs/overnight_monitor.pid)` (from DAT).',
                  'This does not stop an already running training pipeline.']
        if self.selected:
            lines += ['', '## Selected configurations (validation only)', '']
            lines += [f'- {dataset}: {value}' for dataset, value in self.selected.items()]
        tmp = ROOT/'OVERNIGHT_STATUS.tmp'
        tmp.write_text('\n'.join(lines) + '\n')
        tmp.replace(ROOT/'OVERNIGHT_STATUS.md')
        pane = subprocess.run(['tmux', 'capture-pane', '-p', '-t', f'{self.session}:0', '-S', '-120'],
                              capture_output=True, text=True, timeout=10)
        if pane.returncode == 0:
            (ROOT/'logs/tmux_latest.txt').write_text(pane.stdout)

    def wait(self, pid, identity):
        while identity is not None and process_identity(pid) == identity:
            self.report()
            time.sleep(self.interval)
        self.report()

    def launch(self, dataset, batch, epochs, reason):
        if (batch, epochs) not in self.configs:
            self.configs.append((batch, epochs))
        self.phase = f'Running {dataset}, batch {batch}, epoch limit {epochs}'
        self.event(f'{self.phase}: {reason}')
        log = ROOT/'logs'/f'overnight-{dataset}-b{batch}-e{epochs}-{time.time_ns()}.log'
        with log.open('a'):
            child = subprocess.Popen(['bash', '-o', 'pipefail', '-c',
                                      'bash run_epoch128.sh "$1" "$2" "$3" 2>&1 | tee "$4"',
                                      'monitor', dataset, str(batch), str(epochs), str(log)],
                                     cwd=ROOT, start_new_session=True)
            while child.poll() is None:
                self.report()
                time.sleep(self.interval)
        self.event(f'{dataset} batch {batch}, limit {epochs}: pipeline exit {child.returncode}; log {log.name}')
        return child.returncode == 0 and self.complete(dataset, batch, epochs)

    def finish_dataset(self, dataset):
        batch, epochs = 192, 20
        if not self.complete(dataset, batch, epochs):
            # Resume interrupted healthy runs; use a fresh smaller batch on numerical/memory failure.
            texts = []
            for method in METHODS:
                path = run_dir(dataset, method, batch, epochs)/'console.log'
                if path.exists():
                    texts.append(path.read_text()[-24000:].lower())
            failed = any(term in text for text in texts
                         for term in ('out of memory', 'nonfinite', 'loss nan', 'loss inf'))
            if failed:
                batch = 128
            if not self.launch(dataset, batch, epochs, 'Original pipeline incomplete; one recovery attempt'):
                self.event(f'{dataset}: recovery did not complete; preserving logs and moving to next dataset')
                self.selected[dataset] = 'INCOMPLETE: inspect logs'
                return
        summaries = {m: read_json(run_dir(dataset, m, batch, epochs)/'training_summary.json') for m in METHODS}
        reasons = quality_reasons(dataset, summaries)
        selected = f'b{batch}_e{epochs}'
        if reasons:
            self.event(f'{dataset}: validation quality check: ' + '; '.join(reasons))
            if self.launch(dataset, 128, 30, 'One paired quality retry: more updates and larger epoch ceiling'):
                newer = {m: read_json(run_dir(dataset, m, 128, 30)/'training_summary.json') for m in METHODS}
                # Keep paired budgets together; use the lower mean validation loss for selection.
                if sum(s['best_loss'] for s in newer.values()) < sum(s['best_loss'] for s in summaries.values()):
                    selected = 'b128_e30'
                self.event(f'{dataset}: paired configuration selected by mean validation L1: {selected}')
                remaining = quality_reasons(dataset, newer if selected == 'b128_e30' else summaries)
                if remaining:
                    self.event(f'{dataset}: quality concerns remain after bounded retry: ' + '; '.join(remaining))
            else:
                self.event(f'{dataset}: quality retry incomplete; retaining original completed results')
        else:
            self.event(f'{dataset}: validation checks passed; keeping {selected}')
        self.selected[dataset] = selected


def main():
    signal.signal(signal.SIGINT, signal.default_int_handler)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-pid', type=int, required=True)
    p.add_argument('--session', default='idr128')
    p.add_argument('--interval', type=int, default=30)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    os.chdir(ROOT)
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/overnight_monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (ROOT/'logs/overnight_monitor.pid').write_text(str(os.getpid()) + '\n')
        monitor = Monitor(a.session, max(5, a.interval))
        if a.once:
            monitor.report()
            return
        monitor.event(f'Monitor attached to original pipeline PID {a.pipeline_pid}; no initial configuration change')
        try:
            monitor.wait(a.pipeline_pid, process_identity(a.pipeline_pid))
            monitor.event('Original pipeline exited; checking completeness and validation quality')
            for dataset in DATASETS:
                monitor.finish_dataset(dataset)
            monitor.phase = 'Monitoring finished. Review completion states and selected configurations below.'
            monitor.event(monitor.phase)
        except KeyboardInterrupt:
            monitor.phase = 'Monitor stopped by user; no further automatic actions'
            monitor.event(monitor.phase)
        except Exception as error:
            monitor.phase = f'Monitor failed: {type(error).__name__}: {error}. Training processes are not stopped.'
            monitor.event(monitor.phase)
            raise
        finally:
            monitor.report()


if __name__ == '__main__':
    main()
