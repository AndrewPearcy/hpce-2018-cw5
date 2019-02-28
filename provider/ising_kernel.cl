enum {
  rng_group_bond_lr=1,
  rng_group_bond_ud=2,
  rng_group_flip=3,
  rng_group_init=4,
};


__constant uint hfinal(uint acc)
{
  acc ^= acc >> 15;
  acc *= 2246822519U;
  acc ^= acc >> 13;
  acc *= 3266489917U;
  acc ^= acc >> 16;
  return acc;
}


__constant uint hround(uint acc, uint data)
{ acc += data * 2246822519U; acc  = (acc<<13) | (acc>>(32-13)); return acc * 2654435761U;
}

__constant uint hrng(uint seed, uint group, uint iter, uint pos)
{
  uint acc=0;
  acc=hround(acc,seed);
  acc=hround(acc,group);
  acc=hround(acc,iter);
  acc=hround(acc,pos);
  return hfinal(acc);
}


__constant __kernel void kernel_create_bonds(unsigned n, uint seed,
                                  unsigned step, uint prob,
                                  __global const int *spins,
                                  __global int *up_down,
                                  __global int *left_right)
{
  unsigned y = get_global_id(0);
  unsigned x = get_global_id(1);

  unsigned y_index = y * n;
  bool sC=spins[y_index+x];
  
  bool sU=spins[ ((y+1)%n)*n + x ];
  if(sC!=sU){
    up_down[y_index+x]=0;
  }else{
    up_down[y_index+x]=hrng(seed, rng_group_bond_ud, step, y*n+x) < prob;
  }

  bool sR=spins[ y_index + (x+1)%n ];
  if(sC!=sR){
    left_right[y_index+x]=0;
  }else{
    left_right[y_index+x]=hrng(seed, rng_group_bond_lr, step, y*n+x) < prob;
  }
}

__constant __kernel void kernel_create_clusters(unsigned n,
                                            __global int *up_down,
                                            __global int *left_right,
                                            __global unsigned *cluster,
                                            __global bool *finished) {

  unsigned y = get_global_id(0);
  unsigned x = get_global_id(1);

  unsigned y_index = y * n;

  unsigned prev=cluster[y_index+x];
  unsigned curr=prev;
  if(left_right[y_index+x]){
     curr=min(curr, cluster[y_index+(x+1)%n]);
   }
   if(left_right[y*n+(x+n-1)%n]){
     curr=min(curr, cluster[y_index+(x+n-1)%n]);
   }
   if(up_down[y*n+x]){
     curr=min(curr, cluster[ ((y+1)%n)*n+x]);
   }
   if(up_down[((y+n-1)%n)*n+x]){
     curr=min(curr, cluster[ ((y+n-1)%n)*n+x]);
   }
   if(curr!=prev){
     cluster[y_index+x]=curr;
     *finished=false;
   }
}

__constant __kernel void kernel_flip_clusters(uint seed, unsigned step,
                                              __global unsigned *clusters,
                                              __global int *spins)
{
  unsigned i = get_global_id(0);

  unsigned cluster=clusters[i];
  if(hrng(seed, rng_group_flip, step, cluster) >> 31){
    spins[i] ^= 1;
  }
}

__constant __kernel void kernel_nClusters(__global unsigned *clusters,
                                          __global unsigned *counts,
                                          __global unsigned *nClusters)
{
  unsigned i = get_global_id(0);
  if(!atomic_add(&counts[clusters[i]],(unsigned) 1)){
    atomic_add(nClusters, (unsigned) 1);
  }
}
