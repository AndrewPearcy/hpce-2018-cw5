__constant int D = 3;

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
double __attribute__((overloadable)) atomic_add(__global double *valq,double delta) {
   union {
     double f;
     unsigned long  i;
   } old;
   union {
     double f;
     unsigned long  i;
   } new1;
  do {
     old.f = *valq;
     new1.f = old.f + delta;
   } while (atom_cmpxchg((volatile __global unsigned long *)valq, old.i, new1.i) != old.i);
   return old.f;
} 

__constant float updf(float x)
{
    return exp(-x*x/2) / sqrt(2*3.1415926535897932384626433832795);
}
__constant float mpdf(int r, float range, float *x, 
                        __global const float *M, 
                        __global const float *C, 
                        __global const float *bounds)
{
    float dx=range/r;

    float acc=1.0f;
    for(unsigned i=0; i<D; i++){
        float xt=C[i];
        for(unsigned j=0; j<D; j++){
          xt += M[i*D+j] * x[j];
        }
        acc *= updf(xt) * dx;
      }

      for(unsigned i=0; i<D;i++){
        if(x[i] > bounds[i]){
          acc=0;
        }
      }

      return acc;
}

__kernel void kernel_i1_i2_i3(unsigned r,
                        const float range,
                        __global const float *M,
                        __global const float *C,
                        __global const float *bounds,
                        __global double *acc) {

    unsigned i1 = get_global_id(0);
    unsigned i2 = get_global_id(1);
    unsigned i3 = get_global_id(2);

    float x1= -range/2 + range * (i1/(float)r);
    float x2= -range/2 + range * (i2/(float)r);
    float x3= -range/2 + range * (i3/(float)r);

    float x[3]={x1,x2,x3};
    atomic_add(acc, mpdf(r, range, x, M, C, bounds));
}


