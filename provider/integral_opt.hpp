#ifndef user_integral_opt_hpp
#define user_integral_opt_hpp

#include "puzzler/puzzles/integral.hpp"
#include "tbb/parallel_for.h"
#include <chrono>

namespace puzzler {
class IntegralOptProvider
  : public puzzler::IntegralPuzzle
{
public:
  IntegralOptProvider()
  {}

	virtual void Execute(
			   puzzler::ILog *log,
			   const puzzler::IntegralInput *input,
				puzzler::IntegralOutput *output
			   ) const override
	{
      auto begin = std::chrono::high_resolution_clock::now();
      unsigned r=input->resolution;

      const float range=12;

      std::vector<double> accumulators(r*r*r, 0.0f);
      tbb::parallel_for((unsigned) 0, r, [&](unsigned i1){
	for(unsigned i2=0; i2<r; i2++){
	  for(unsigned i3=0; i3<r; i3++){
	    float x1= -range/2 + range * (i1/(float)r);
	    float x2= -range/2 + range * (i2/(float)r);
	    float x3= -range/2 + range * (i3/(float)r);

	    float x[3]={x1,x2,x3};
	    accumulators[i1*r*r+i2*r+i3] += mpdf(r, range, x, &input->M[0], &input->C[0], &input->bounds[0]);
	  }
	}
      });
      double acc=0;
      for(unsigned i =0; i < r*r*r; i++){
	acc += accumulators[i];	
      }
      log->LogInfo("Integral = %g", acc);
      output->value=acc;
      auto end = std::chrono::high_resolution_clock::now();
std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << "ms" << std::endl;
    }

/*
	float mpdf(int r, float range, float x[D], const float M[D*D], const float C[D], const float bounds[D]) const
    {
      float dx=range/r;

      tbb::atomic<float> acc=1.0f;
      tbb::parallel_for((unsigned) 0, (unsigned) D, [&](unsigned i){
        float xt=C[i];
        for(unsigned j=0; j<D; j++){
          xt += M[i*D+j] * x[j];
        }
        acc = acc * (updf(xt) * dx);
      });

      tbb::parallel_for((unsigned) 0, (unsigned)D, [&](unsigned i){
        if(x[i] > bounds[i]){
          acc=0;
        }
      });

      return acc;
    }
*/

 float mpdf(int r, float range, float x[D], const float M[D*D], const float C[D], const float bounds[D]) const
    {
      float dx=range/r;

      float acc=1.0f;
      for(unsigned i=0; i<D; i++){
        float xt=C[i];
        for(unsigned j= 0; j<D; j++){
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

};
}; //end namespace puzzler
#endif