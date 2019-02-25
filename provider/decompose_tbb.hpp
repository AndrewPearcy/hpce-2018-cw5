#ifndef user_decompose_tbb_hpp
#define user_decompose_tbb_hpp

#include "puzzler/puzzles/decompose.hpp"
#include "tbb/parallel_for.h"

namespace puzzler {
class DecomposeTbbProvider
  : public puzzler::DecomposePuzzle
{
public:
  DecomposeTbbProvider()
  {}

	virtual void Execute(
			  puzzler::ILog *log,
			  const puzzler::DecomposeInput *pInput,
				puzzler::DecomposeOutput *pOutput
			   ) const override
	{
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
	}

};
};
#endif
