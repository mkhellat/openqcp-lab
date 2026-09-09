import json, os, statistics, subprocess, sys, time
SP=os.path.dirname(os.path.abspath(__file__))
PY_=os.path.expanduser("~/.venvs/paulikit/bin/python")
CH=os.path.join(SP,"drain_ab_child.py")
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
def run(v,N,cs):
    p=subprocess.run([PY_,CH,v,str(N),str(cs)],capture_output=True,text=True,
                     env=dict(os.environ,OPENBLAS_NUM_THREADS="1"))
    if p.returncode!=0: return dict(failed=True,err=p.stderr[-200:])
    return json.loads(p.stdout.strip().splitlines()[-1])
N=int(sys.argv[1]); reps=int(sys.argv[2]); cs=2
print(f"N={N} chunk_size={cs}, {reps} reps + warm-up, interleaved, cooldown 55C\n",flush=True)
for v in ("old","new"): cool(); run(v,N,cs)
res={"old":[],"new":[]}; mem={"old":[],"new":[]}
for r in range(reps):
    for v in ("old","new"):
        cool()
        o=run(v,N,cs)
        if o.get("failed"): print("FAIL",v,o["err"]); sys.exit(1)
        res[v].append(o["elapsed"]); mem[v].append(o["peak_rss_mib"])
        print(f"  rep{r+1} {v:>3}: {o['elapsed']:6.3f}s  {o['peak_rss_mib']:5.0f}MiB "
              f"terms={o['n_terms']}",flush=True)
print()
for v in ("old","new"):
    print(f"{v:>3}: mean={statistics.mean(res[v]):6.3f}s "
          f"sd={statistics.stdev(res[v]):5.3f}  "
          f"rss={statistics.mean(mem[v]):5.0f}MiB")
a,b=statistics.mean(res["old"]),statistics.mean(res["new"])
print(f"\nnew/old = {b/a:.3f}  ({'new faster' if b<a else 'old faster'} "
      f"by {abs(1-b/a)*100:.1f}%)")
# Welch's t-test
import math
def welch(x,y):
    mx,my=statistics.mean(x),statistics.mean(y)
    vx,vy=statistics.variance(x),statistics.variance(y)
    nx,ny=len(x),len(y)
    se=math.sqrt(vx/nx+vy/ny)
    if se==0: return 0.0,float('inf')
    t=(mx-my)/se
    df=(vx/nx+vy/ny)**2/((vx/nx)**2/(nx-1)+(vy/ny)**2/(ny-1))
    return t,df
t,df=welch(res["old"],res["new"])
print(f"Welch t={t:.3f} df={df:.1f}  (|t|>~2.3 at df~8 is p<0.05)")
