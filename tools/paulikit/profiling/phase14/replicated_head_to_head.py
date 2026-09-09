"""Replicated head-to-head on the REAL pipeline, per
phase13/MEASUREMENT_METHODOLOGY.md: interleaved, n>=5, first rep
discarded, cooldown between runs, temperature recorded."""
import json, os, statistics, subprocess, sys, time
SC = os.path.dirname(os.path.abspath(__file__))
PY_ = os.path.expanduser("~/.venvs/paulikit/bin/python")
CHILD = os.path.join(SC, "replicated_head_to_head_target.py")
TEMP = "/sys/class/thermal/thermal_zone7/temp"
def temp():
    try:
        with open(TEMP) as f: return int(f.read())/1000.0
    except OSError: return None
def cool(target=55.0, cap=180):
    t0 = time.perf_counter()
    while (temp() or 0) > target:
        if time.perf_counter()-t0 > cap: return False
        time.sleep(2)
    return True
def run(impl, N):
    p = subprocess.run([PY_, CHILD, impl, str(N)], capture_output=True,
        text=True, env=dict(os.environ, OPENBLAS_NUM_THREADS="1"))
    if p.returncode != 0:
        return dict(failed=True, error=p.stderr.strip()[-200:])
    return json.loads(p.stdout.strip().splitlines()[-1])
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
Ns = [int(a) for a in sys.argv[2:]] or [100, 150]
print(f"{reps} reps + 1 discarded, interleaved, cooldown to 55C\n", flush=True)
print(f"{'N':>4} {'q':>3} {'dim':>6} {'pauli_lcu':>20} {'paulikit':>20} {'ratio':>7} {'mem':>12}", flush=True)
allrows = []
for N in Ns:
    cool(); [run(i, N) for i in ("pauli_lcu", "paulikit")]   # warm-up
    per = {"pauli_lcu": [], "paulikit": []}; mem = {"pauli_lcu": [], "paulikit": []}
    meta = {}; bad = None
    for _ in range(reps):
        for impl in ("pauli_lcu", "paulikit"):
            cool(); r = run(impl, N)
            if r.get("failed"): bad = (impl, r["error"]); break
            per[impl].append(r["elapsed"]); mem[impl].append(r["peak_rss_mib"])
            meta = r; allrows.append(dict(r, temp=temp()))
        if bad: break
    if bad:
        print(f"{N:>4} {bad[0]} FAILED: {bad[1][:70]}", flush=True); continue
    ml, mp = statistics.mean(per["pauli_lcu"]), statistics.mean(per["paulikit"])
    sl = statistics.stdev(per["pauli_lcu"]); sp = statistics.stdev(per["paulikit"])
    Ml, Mp = statistics.mean(mem["pauli_lcu"]), statistics.mean(mem["paulikit"])
    print(f"{N:>4} {meta['qubits']:>3} {meta['dim']:>6} "
          f"{ml:>8.3f}s±{sl:<6.3f} {mp:>8.3f}s±{sp:<6.3f} "
          f"{ml/mp:>6.2f}x {Ml:>6.0f}/{Mp:<5.0f}M", flush=True)
out = os.path.join(SC, "replicated_head_to_head_results.jsonl")
with open(out, "a") as f:
    for r in allrows: f.write(json.dumps(r)+"\n")
print(f"\nratio >1 = paulikit faster. raw -> {out}", flush=True)
