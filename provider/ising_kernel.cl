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
{
  acc += data * 2246822519U;
  acc  = (acc<<13) | (acc>>(32-13));
  return acc * 2654435761U;
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
