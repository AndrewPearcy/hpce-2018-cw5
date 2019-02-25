#ifndef user_integral_tbb_hpp
#define user_integral_tbb_hpp

#include "puzzler/puzzles/integral.hpp"
#include "tbb/parallel_for.h"
namespace puzzler {
class IntegralTbbProvider
  : public puzzler::IntegralPuzzle
{
public:
  IntegralTbbProvider()
  {}

	virtual void Execute(
			   puzzler::ILog *log,
			   const puzzler::IntegralInput *input,
				puzzler::IntegralOutput *output
			   ) const override
	{

      unsigned r=input->resolution;

      const float range=12;

      std::vector<double> accumulators(r*r*r, 0.0f);
//      double acc=0;
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
    }


};
}; //end namespace puzzler
#endif
