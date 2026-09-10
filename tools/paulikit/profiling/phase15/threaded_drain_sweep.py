"""Interleaved, thermally-controlled thread sweep per
phase13/MEASUREMENT_METHODOLOGY.md."""
import json, os, statistics, subprocess, sys, time
SP=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.expanduser("~/.venvs/paulikit/bin/python")
CH=os.path.join(SP,"thread_child.py")
def temp():
    try:
        with open("/sys/class/thermal/thermal_zone7/temp") as f: return int(f.read())/1000
    except OSError: return None
def cool(t=55.0,cap=180):
    s=time.perf_counter()
    while (temp() or 0)>t:
        if time.perf_counter()-s>cap: return False
        time.sleep(2)
    return True
def run(nt,N,cs):
    p=subprocess.run([PY_,CH,str(nt),str(N),str(cs)],capture_output=True,text=True,
                     env=dict(os.environ,OPENBLAS_NUM_THREADS="1"))
    if p.returncode!=0: return dict(failed=True,err=p.stderr[-300:])
    return json.loads(p.stdout.strip().splitlines()[-1])
N=int(sys.argv[1]); reps=int(sys.argv[2]); cs=int(sys.argv[3]) if len(sys.argv)>3 else 2
threads=[1,2,4,8]
print(f"N={N} chunk_size={cs}, {reps} reps + warm-up, INTERLEAVED, cooldown 55C\n",flush=True)
for nt in threads: cool(); run(nt,N,cs)          # warm-up, discarded
res={nt:[] for nt in threads}; mem={nt:[] for nt in threads}; terms=None
for r in range(reps):
    for nt in threads:
        cool()
        o=run(nt,N,cs)
        if o.get("failed"): print("FAIL",nt,o["err"]); sys.exit(1)
        res[nt].append(o["wall"]); mem[nt].append(o["peak_rss_mib"])
        terms=o["n_terms"]
        print(f"  rep{r+1} t={nt}: wall={o['wall']:6.3f}s cpu={o['cpu']:6.3f} "
              f"cores={o['cores']:4.2f} rss={o['peak_rss_mib']:5.0f}M",flush=True)
print()
base=statistics.mean(res[1])
print(f"{'threads':>8} {'wall mean':>10} {'sd':>7} {'speedup':>8} {'eff':>6} {'rss':>7}")
for nt in threads:
    m=statistics.mean(res[nt]); sd=statistics.stdev(res[nt]) if len(res[nt])>1 else 0
    print(f"{nt:>8} {m:>9.3f}s {sd:>7.3f} {base/m:>7.2f}x {base/m/nt:>5.0%} "
          f"{statistics.mean(mem[nt]):>6.0f}M")
print(f"\nterms={terms} (identical across all conditions)")
# Amdahl prediction from the measured serial fraction f=0.0336
f=0.0336
print("\nAmdahl prediction at f=0.0336 (measured GIL-held fraction):")
for nt in threads:
    print(f"  t={nt}: predicted {1/(f+(1-f)/nt):5.2f}x, measured {base/statistics.mean(res[nt]):5.2f}x")
