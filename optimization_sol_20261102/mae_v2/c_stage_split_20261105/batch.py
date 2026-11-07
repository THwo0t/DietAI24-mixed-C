"""Run only ASA portions, then score and verify all 843 frozen predictions."""
from pathlib import Path
import fcntl
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import run
from opt_provider import atomic_json, sha, stamp


def completed_worker(ctx):
    directory = HERE / 'run/ASA'
    path = directory / 'worker_done.json'
    if not path.exists():
        return False
    marker = run.load(path)
    run.require(marker['status'] == 'completed' and marker['profile'] == 'ASA'
                and marker['freeze_sha256'] == ctx['frozen']
                and marker['records_sha256'] == sha(directory/'records.json')
                and marker['usage_sha256'] == sha(directory/'usage.json'), 'worker_done_invalid')
    import score
    score.completed_records([r for r in ctx['rows'] if r['dataset'] == 'asa24'], 'ASA', ctx['frozen'])
    run.validate_identity_seal(ctx, directory)
    return True


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def main():
    codes = {}
    outcome = dict(status='stopped', started=stamp(), workers=32, scope='843_development',
                   online_scope='ASA_portions_only', N5_reused=314, ASA_identity_reused=529)
    child = None
    with (HERE/'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run.require(not (HERE/'batch_outcome.json').exists(), 'batch_already_attempted_do_not_restart_automatically')
            ctx = run.context()
            run.validate_identity_seal(ctx, HERE/'run/ASA')
            command = [sys.executable,'-B',str(HERE/'run.py'),'--profile','ASA','--allow-paid','--workers','32']
            child = subprocess.Popen(command,cwd=REPO,stdin=subprocess.DEVNULL)
            atomic_json(HERE/'worker_launch.json',dict(pid=child.pid,started=stamp(),workers=32,
                freeze_sha256=ctx['frozen'],command=command))
            deadline = time.monotonic()+40*60
            while not completed_worker(ctx):
                code = child.poll()
                run.require(code is None, 'ASA_worker_exited_before_valid_done_marker:'+str(code))
                run.require(time.monotonic() < deadline, 'ASA_worker_deadline_exceeded')
                time.sleep(2)
            try:
                codes['ASA_worker'] = child.wait(timeout=10)
                run.require(codes['ASA_worker'] == 0, 'ASA_worker_nonzero_after_done')
            except subprocess.TimeoutExpired:
                stop_child(child)
                codes['ASA_worker'] = child.returncode
                outcome['worker_exit_issue'] = 'completed_and_sealed_but_interpreter_needed_termination'
            for name, args in [('score',['run.py','--score']), ('verify',['verify_completed.py']),
                               ('compare',['compare_baseline.py'])]:
                codes[name] = subprocess.run([sys.executable,'-B',str(HERE/args[0]),*args[1:]],
                    cwd=REPO,stdin=subprocess.DEVNULL,timeout=180).returncode
                run.require(codes[name] == 0, name+'_failed')
            outcome['status'] = 'completed_offline_verified' if outcome.get('worker_exit_issue') else 'completed'
        except BaseException as exc:
            outcome['error_type'] = type(exc).__name__
            raise
        finally:
            if child is not None:
                stop_child(child)
            outcome.update(exit_codes=codes,finished=stamp())
            atomic_json(HERE/'batch_outcome.json',outcome)
            run.event(**outcome)
    return 0


if __name__ == '__main__':
    sys.exit(main())
