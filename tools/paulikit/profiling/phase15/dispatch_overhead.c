/* Per-task dispatch overhead: OpenMP vs pthreads, at OUR granularity.
 * A chunk is ~390us of work and there are 5595 of them, so what
 * matters is cost per task, not peak throughput. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+1e-9*t.tv_nsec;}
#define NTASK 5595
static volatile double sink=0;
static void work(int i){ /* ~cheap stand-in; we measure DISPATCH */
    double s=0; for(int k=0;k<200;k++) s+=k*i; sink+=s; }

typedef struct{int lo,hi;} range_t;
static void*pt_worker(void*a){ range_t*r=a; for(int i=r->lo;i<r->hi;i++) work(i); return NULL;}

int main(void){
    int threads[]={1,2,4,8};
    printf("dispatching %d tasks (work() is deliberately tiny - this\n",NTASK);
    printf("isolates DISPATCH cost, not compute)\n\n");
    printf("%8s %14s %14s\n","threads","OpenMP us/task","pthread us/task");
    for(unsigned t=0;t<sizeof(threads)/sizeof(*threads);t++){
        int nt=threads[t];
        /* OpenMP: dynamic schedule, one task per chunk index */
        omp_set_num_threads(nt);
        double t0=now();
        #pragma omp parallel for schedule(dynamic,1)
        for(int i=0;i<NTASK;i++) work(i);
        double omp_t=(now()-t0)/NTASK*1e6;
        /* pthreads: static split, threads created once */
        pthread_t th[8]; range_t rg[8];
        t0=now();
        for(int k=0;k<nt;k++){
            rg[k].lo=(int)((long)NTASK*k/nt); rg[k].hi=(int)((long)NTASK*(k+1)/nt);
            pthread_create(&th[k],NULL,pt_worker,&rg[k]);
        }
        for(int k=0;k<nt;k++) pthread_join(th[k],NULL);
        double pt_t=(now()-t0)/NTASK*1e6;
        printf("%8d %14.4f %14.4f\n",nt,omp_t,pt_t);
    }
    printf("\n(for scale: one real chunk is ~390 us of actual work)\n");
    return 0;
}
