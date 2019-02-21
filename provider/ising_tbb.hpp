#ifndef user_ising_tbb_hpp
#define user_ising_tbb_hpp

#include "puzzler/puzzles/ising.hpp"

#include "tbb/parallel_for.h"
namespace puzzler {
class IsingTbbProvider
  : public puzzler::IsingPuzzle
{
public:
  IsingTbbProvider()
  {}

	virtual void Execute(
			   puzzler::ILog *log,
			   const puzzler::IsingInput *pInput,
				 puzzler::IsingOutput *pOutput
			   ) const override
	{
      log->LogInfo("Building world");
      unsigned n=pInput->n;
      uint32_t prob=pInput->prob;
      uint32_t seed=pInput->seed;
      std::vector<int> spins(n*n);
      std::vector<int> left_right(n*n);
      std::vector<int> up_down(n*n);
      std::vector<unsigned> clusters(n*n);
      std::vector<unsigned> counts(n*n);
      tbb::parallel_for((unsigned) 0, n*n, [&](unsigned i){
        spins[i]=hrng(seed, rng_group_init, 0, i) & 1;
      });

      log->LogInfo("Doing iterations");
      std::vector<uint32_t> stats(n);

      tbb::parallel_for((unsigned) 0, n, [&](unsigned i){
        log->LogVerbose("  Iteration %u", i);
        create_bonds(   log, n, seed, i, prob, &spins[0], 
                        &up_down[0], &left_right[0]);
        create_clusters(log,  n, seed, i,
                        &up_down[0], &left_right[0], &clusters[0]);
        flip_clusters(  log,  n, seed, i, &clusters[0], &spins[0]);
        count_clusters( log,  n, seed, i, &clusters[0], &counts[0], stats[i]);
        log->LogVerbose("  clusters count is %u", stats[i]);

        log->Log( puzzler::Log_Debug, [&](std::ostream &dst){
          dst<<"\n";
          tbb::parallel_for((unsigned) 0, n, [&](unsigned y){
              tbb::parallel_for((unsigned) 0, n, [&](unsigned  x){
              dst<<(spins[y*n+x]?"+":" ");
            });
            dst<<"\n";
          });
        });
      });
      
      pOutput->history=stats;
      log->LogInfo("Finished");
    }

};
};
#endif
