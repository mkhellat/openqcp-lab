"""Head-to-head, cycle-measured, across the qubit boundary where
pauli_lcu's dense-input requirement becomes a hard ceiling."""
import json, os, statistics, subprocess, sys, time
SP=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.expanduser("~/.venvs/paulikit/bin/python")
CH=os.path.join(SP,"bench_child.py")
def temp():
    try:
        with open("/sys/class/thermal/thermal_zone7/temp") as f: return int(f.read())/1000
    except OSError: return None
def cool(t=55.0,cap=240):
    s=time.perf_counter()
    while (temp() or 0)>t:
        if time.perf_counter()-s>cap: return False
        time.sleep(2)
    return True
def run(impl,N,nw=4):
    csv=os.path.join(SP,"bperf.csv")
    p=subprocess.run(["perf","stat","-e","instructions:u,cycles:u","-x,","-o",csv,
                      PY_,CH,impl,str(N),str(nw)],
                     capture_output=True,text=True,
                     env=dict(os.environ,OPENBLAS_NUM_THREADS="1"))
    try: d=json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return dict(ok=False,error=(p.stderr or p.stdout)[-160:],impl=impl,N=N)
    try:
        for line in open(csv):
            f=line.split(",")
            if len(f)>2 and f[2].startswith("instructions"): d["ins"]=float(f[0])
            if len(f)>2 and f[2].startswith("cycles"):       d["cyc"]=float(f[0])
    except Exception: pass
    return d
Ns=[int(a) for a in sys.argv[1:]] or [150]
impls=["pauli_lcu","paulikit_default","paulikit_seq","paulikit_thread","paulikit_process"]
reps=3
print(f"{'N':>4} {'q':>3} {'impl':>17} {'wall':>9} {'cycles':>10} {'peak RSS':>10} {'terms':>14}")
for N in Ns:
    for impl in impls:
        vals=[]
        for r in range(reps):
            cool(); o=run(impl,N)
            if not o.get("ok"):
                print(f"{N:>4} {o.get('qubits','?'):>3} {impl:>17}   "
                      f"FAILED: {o.get('error','?')[:60]}",flush=True)
                vals=None; break
            vals.append(o)
        if not vals: continue
        w=statistics.median(v["elapsed"] for v in vals)
        c=statistics.median(v.get("cyc",0) for v in vals)/1e9
        m=statistics.median(v["peak_rss_mib"] for v in vals)
        print(f"{N:>4} {vals[0]['qubits']:>3} {impl:>17} {w:>8.3f}s "
              f"{c:>9.2f}B {m:>9.0f}M {vals[0]['n_terms']:>14,}",flush=True)
    print(flush=True)
