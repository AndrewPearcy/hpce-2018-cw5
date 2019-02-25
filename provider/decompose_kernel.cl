
__kernel void  kernel_makeBit(
	__global uint *matrix,
	uint seed,
	uint P) 
    {
      uint input=get_global_id(0);
      const uint PRIME32_1  = 2654435761U;
      const uint PRIME32_2 = 2246822519U;
      seed += input * PRIME32_2;
      seed  = (seed<<13) | (seed>>(32-13));
      seed *= PRIME32_1;
      matrix[input] = seed % P;
    };

