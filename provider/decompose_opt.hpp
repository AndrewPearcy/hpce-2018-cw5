#ifndef user_decompose_opt_hpp
#define user_decompose_opt_hpp

#include "puzzler/puzzles/decompose.hpp"
#include "tbb/parallel_for.h"
#include <chrono>
namespace puzzler {
class DecomposeOptProvider
  : public puzzler::DecomposePuzzle
{
public:
  DecomposeOptProvider()
  {}

	virtual void Execute(
			  puzzler::ILog *log,
			  const puzzler::DecomposeInput *pInput,
				puzzler::DecomposeOutput *pOutput
			   ) const override
	{
    auto begin = std::chrono::high_resolution_clock::now();
    unsigned n=pInput->n;
    unsigned rr=n;
    unsigned cc=n;
    unsigned p=7;

    log->LogInfo("Building random matrix");
    std::vector<uint32_t> matrix(rr*cc);
    tbb::parallel_for((unsigned long) 0, matrix.size(), [&](unsigned i){
      matrix[i]=make_bit(pInput->seed, i);
    });
    dump(log, Log_Verbose, rr, cc, &matrix[0]);

    log->LogInfo("Doing the decomposition");
    decompose(log, rr, cc, p, &matrix[0]);

    log->LogInfo("Collecting decomposed hash.");
    dump(log, Log_Verbose, rr, cc, &matrix[0]);
    tbb::atomic<uint64_t> hash=0;
    tbb::parallel_for((unsigned long) 0, matrix.size(), [&](unsigned i){
      hash += uint64_t(matrix[i])*i;
    });
    pOutput->hash=hash;

    log->LogInfo("Finished");
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << "us" << std::endl;
	}



void decompose(ILog *log, unsigned rr, unsigned cc, unsigned p, uint32_t *matrix) const
    {
      auto at = [=](unsigned r, unsigned c) -> uint32_t &{
        assert(r<rr && c<cc);
        return matrix[rr*c+r];
      };

      dump(log, Log_Debug, rr, cc, matrix);

      unsigned rank=0;
      for(unsigned c1=0; c1<cc; c1++){
        unsigned r1=rank;
        while(r1<rr && at(r1,c1)==0){
          ++r1;
        }

        if(r1!=rr){
          unsigned pivot=at(r1,c1);
          tbb::parallel_for((unsigned) 0, cc, [&](unsigned c2){
            std::swap( at(r1,c2), at(rank,c2) );
            at(rank,c2)=div( at(rank,c2) , pivot );
          });

          tbb::parallel_for((unsigned) rank+1, rr, [&](unsigned r2){
            unsigned count=at(r2, c1);
            for(unsigned c2=0; c2<cc; c2++){
              at(r2,c2) = sub( at(r2,c2) , mul( count, at(rank,c2)) );
            }
          });

          ++rank;
        }

        dump(log, Log_Debug, rr, cc, matrix);
      }
    }



};
};
#endif
