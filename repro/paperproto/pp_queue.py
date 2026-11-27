"""Tiny job scheduler. Re-reads paperproto/jobs.json every pass, so jobs can be appended live.

Job: {"id", "cmd": [...], "deps": [ids], "prio": int (lower first), "done_file": path, "env": {...}}
A job is done when its done_file exists. Failed jobs are retried (training resumes from
latest.pt) up to MAX_TRIES. State: runs/paperproto/queue_state.json; logs: runs/paperproto/logs/.
Concurrency: env PP_SLOTS (default 4), re-read every pass from runs/paperproto/SLOTS if present.
"""
import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'paperproto' / 'jobs.json'
BASE = ROOT / 'runs' / 'paperproto'
LOGS = BASE / 'logs'; LOGS.mkdir(parents=True, exist_ok=True)
STATE = BASE / 'queue_state.json'
MAX_TRIES = 3


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def main():
    running, tries, failed, started_at = {}, {}, set(), {}
    while True:
        jobs = json.loads(JOBS.read_text())
        slots = int((BASE / 'SLOTS').read_text()) if (BASE / 'SLOTS').exists() else int(os.environ.get('PP_SLOTS', 4))
        done = {j['id'] for j in jobs if (ROOT / j['done_file']).exists()}
        for jid, p in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[jid]
            if jid in done:
                print(now(), 'finished', jid, flush=True)
            else:
                tries[jid] = tries.get(jid, 0) + 1
                print(now(), f'FAILED rc={rc}', jid, f'try {tries[jid]}/{MAX_TRIES}', flush=True)
                if tries[jid] >= MAX_TRIES:
                    failed.add(jid)
        pending = [j for j in jobs if j['id'] not in done and j['id'] not in running and j['id'] not in failed
                   and all(d in done for d in j.get('deps', []))]
        pending.sort(key=lambda j: (j.get('prio', 50), jobs.index(j)))
        while pending and len(running) < slots:
            j = pending.pop(0)
            env = dict(os.environ, **j.get('env', {}))
            log = open(LOGS / f"{j['id']}.log", 'a')
            log.write(f'\n=== {now()} start {j["cmd"]}\n'); log.flush()
            running[j['id']] = subprocess.Popen(j['cmd'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            started_at[j['id']] = now()
            print(now(), 'started', j['id'], flush=True)
        blocked = [j['id'] for j in jobs if j['id'] not in done and j['id'] not in running and j['id'] not in failed
                   and not all(d in done for d in j.get('deps', []))]
        STATE.write_text(json.dumps(dict(updated=now(), slots=slots, running=sorted(running), started_at=started_at,
                                         done=sorted(done), failed=sorted(failed), tries=tries,
                                         pending=[j['id'] for j in pending], blocked=blocked,
                                         total=len(jobs)), indent=1))
        time.sleep(20)


if __name__ == '__main__':
    main()
