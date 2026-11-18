"""Notify locally when the four specifically requested continuation runs finish."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
RUNS = [('imagenet', method, 'b192_e20') for method in ('idr', 'supervised')] + [
    ('marmousi2', method, 'b128_e30') for method in ('idr', 'supervised')]
MARKER = ROOT/'logs/four_requested_completion.json'


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def alert(title, message):
    env = os.environ.copy()
    env.setdefault('DBUS_SESSION_BUS_ADDRESS', f'unix:path=/run/user/{os.getuid()}/bus')
    result = subprocess.run(['notify-send', '-a', 'IDR experiments', '-t', '0', title, message],
                            env=env, capture_output=True, text=True, timeout=15)
    subprocess.run(['tmux', 'display-message', '-t', 'idr128:0', '-d', '60000', message],
                   capture_output=True, timeout=10)
    pane = subprocess.run(['tmux', 'display-message', '-p', '-t', 'idr128:0', '#{pane_tty}'],
                          capture_output=True, text=True, timeout=10)
    if pane.returncode == 0:
        with open(pane.stdout.strip(), 'a') as f:
            f.write('\n\a' + title + ': ' + message + '\n')
    return dict(desktop_notification_exit=result.returncode, error=result.stderr.strip())


if __name__ == '__main__':
    if MARKER.exists():
        raise SystemExit('Completion alert already recorded; no duplicate notification.')
    print('Watching ImageNet batch-192 pair and Marmousi2 batch-128 pair.', flush=True)
    while True:
        complete = []
        for dataset, method, budget in RUNS:
            folder = ROOT/'runs/continuation5'/dataset/method/budget
            summary = read(folder/'training_summary.json')
            evaluation = read(folder/'evaluation/summary.json')
            if summary and evaluation:
                complete.append(dict(run=f'{dataset}/{method}/{budget}', **summary))
        status_text = (ROOT/'CONTINUATION_STATUS.md').read_text()
        status_complete = all(f'| {d}/{m}/{b} | complete;' in status_text for d,m,b in RUNS)
        if len(complete) == 4 and status_complete:
            plot_python = (ROOT/'logs/plot_python.txt').read_text().strip()
            try:
                subprocess.run([plot_python, str(ROOT/'plot_continuations.py')], cwd=ROOT, check=True)
            except Exception as error:
                alert('IDR results need attention', f'Training completed, but curve/report generation failed: {error}')
                raise
            message = 'All four requested continuations and evaluations finished: ImageNet Fast IDR + supervised (batch 192), Marmousi2 Fast IDR + supervised (batch 128). Loss curves and updated report are ready: repro/CONTINUATION_RESULTS.md.'
            outcome = alert('Four IDR experiments complete', message)
            MARKER.write_text(json.dumps(dict(time=datetime.now(ZoneInfo('Asia/Kolkata')).isoformat(),
                                             runs=complete, **outcome), indent=2)+'\n')
            print(message, outcome, flush=True)
            break
        report = (ROOT/'CONTINUATION_STATUS.md').read_text()
        failed = [f'{d}/{m}/{b}' for d,m,b in RUNS if f'| {d}/{m}/{b} | FAILED:' in report]
        if failed:
            outcome = alert('IDR experiment needs attention', 'A requested continuation failed: ' + ', '.join(failed))
            print('Failure notification:', failed, outcome, flush=True)
            break
        time.sleep(30)
