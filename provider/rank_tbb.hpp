#ifndef user_rank_tbb_hpp
#define user_rank_tbb_hpp

#include "puzzler/puzzles/rank.hpp"
#include <chrono>
#include "tbb/parallel_for.h"

namespace puzzler{
class RankTbbProvider
  : public puzzler::RankPuzzle
{
public:
  RankTbbProvider()
  {}

	virtual void Execute(
			   puzzler::ILog *log,
			   const puzzler::RankInput *pInput,
				puzzler::RankOutput *pOutput
			   ) const override
	{
      auto begin = std::chrono::high_resolution_clock::now();
      const std::vector<std::vector<uint32_t> > &edges=pInput->edges;
      float tol=pInput->tol;
      unsigned n=edges.size();

      log->LogInfo("Starting iterations.");
      std::vector<float> curr(n, 0.0f);
      curr[0]=1.0;
      std::vector<float> next(n, 0.0f);
      float dist = norm(curr, next);
      while(tol < dist) {
        log->LogVerbose("dist=%g", dist);
        iteration(log, n, edges, &curr[0], &next[0]);
        std::swap(curr, next);
        dist=norm(curr, next);
      }
      
      pOutput->ranks= (std::vector<float>) curr;
      
      log->LogInfo("Finished");
      auto end = std::chrono::high_resolution_clock::now();
      std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << "ns" << std::endl;
	}


  void iteration(ILog *log, unsigned n, 
                  const std::vector<std::vector<uint32_t> > &edges, 
                  const float *current, 
                  float *next) const
    {
      tbb::parallel_for((unsigned) 0, n, [&](unsigned i){
        next[i]=0;
      });
      std::vector<tbb::atomic<float>> atomic_next(n, 0.0f);
      //for(unsigned i=0; i<n; i++){
      tbb::parallel_for((unsigned) 0, n, [&](unsigned i){  
	for(unsigned j=0; j<edges[i].size(); j++){
          int dst=edges[i][j];
          atomic_next[dst] = atomic_next[dst] + (current[i] / edges[i].size());
        }
      });

      tbb::parallel_for((unsigned) 0, n, [&](unsigned i) {
        next[i] = atomic_next[i].load(); 
      });

      tbb::atomic<double> total=0;
      tbb::parallel_for((unsigned) 0, n, [&](unsigned i){
        next[i] = (current[i] * 0.3  + next[i] * 0.7 );
        total = total + next[i];
      });
      log->LogVerbose("  total=%g", total);
      tbb::parallel_for((unsigned) 0, n ,[&](unsigned i){
        next[i] = next[i] / total;
        log->LogVerbose("    c[%u] = %g", i, next[i]);
      });
    }

};
};

#endif
