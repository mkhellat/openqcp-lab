"""Item 1, take 3. Wall clock is unusable on this machine: the core
alternates between ~4.3 GHz (turbo) and ~2.4 GHz (base) run to run
under intel_pstate/powersave, which produces a clean 2x in wall time
with IDENTICAL instruction counts. Verified with perf: fast runs
29.556B ins / 9.49B cyc, slow runs 29.559B ins / 10.8B cyc.

So this measures CYCLES and INSTRUCTIONS, which are frequency
invariant, and reports parallel speedup as the ratio of TOTAL CYCLES
consumed. For a fixed amount of work, a perfectly scaling parallel
run consumes the same total cycles as the serial one; overhead and
contention show up as extra cycles. Wall is still recorded, flagged
as advisory only.
"""
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
    csv=os.path.join(SP,f"perf_{nt}.csv")
    p=subprocess.run(["perf","stat","-e","instructions:u,cycles:u","-x,","-o",csv,
                      PY_,CH,str(nt),str(N),str(cs)],
                     capture_output=True,text=True,
                     env=dict(os.environ,OPENBLAS_NUM_THREADS="1"))
    if p.returncode!=0: return dict(failed=True,err=p.stderr[-300:])
    d=json.loads(p.stdout.strip().splitlines()[-1])
    for line in open(csv):
        f=line.split(",")
        if len(f)>2 and f[2].startswith("instructions"): d["ins"]=float(f[0])
        if len(f)>2 and f[2].startswith("cycles"):       d["cyc"]=float(f[0])
    return d
N=int(sys.argv[1]); reps=int(sys.argv[2]); cs=int(sys.argv[3]) if len(sys.argv)>3 else 2
threads=[1,2,4,8]
print(f"N={N} cs={cs}, {reps} reps, INTERLEAVED, cooldown 55C")
print("cycles/instructions are frequency-invariant; wall is advisory\n",flush=True)
for nt in threads: cool(); run(nt,N,cs)
res={nt:{"cyc":[],"ins":[],"wall":[]} for nt in threads}; terms=None
for r in range(reps):
    for nt in threads:
        cool(); o=run(nt,N,cs)
        if o.get("failed"): print("FAIL",nt,o["err"]); sys.exit(1)
        for k in ("cyc","ins","wall"): res[nt][k].append(o[k])
        terms=o["n_terms"]
        print(f"  rep{r+1} t={nt}: cyc={o['cyc']/1e9:6.3f}B ins={o['ins']/1e9:6.3f}B "
              f"wall={o['wall']:6.3f}s",flush=True)
print()
b_cyc=statistics.median(res[1]["cyc"]); b_ins=statistics.median(res[1]["ins"])
print(f"{'thr':>4} {'cycles(B)':>10} {'sd':>6} {'ins(B)':>8} {'cyc vs t=1':>11} {'wall(s)':>9}")
for nt in threads:
    c=statistics.median(res[nt]["cyc"]); sd=statistics.stdev(res[nt]["cyc"])/1e9 if reps>1 else 0
    i=statistics.median(res[nt]["ins"]); w=statistics.median(res[nt]["wall"])
    print(f"{nt:>4} {c/1e9:>10.3f} {sd:>6.3f} {i/1e9:>8.3f} {c/b_cyc:>10.2f}x {w:>9.3f}")
print(f"\nterms={terms}")
print("cyc vs t=1 near 1.00 = perfect scaling (no extra cycles spent);")
print("above 1.00 = coordination + contention overhead.")
