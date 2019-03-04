#ifndef user_ising_tbb_hpp
#define user_ising_tbb_hpp

#include "puzzler/puzzles/ising.hpp"
#include <chrono>
#include "tbb/parallel_for.h"
namespace puzzler {
class IsingTbbProvider
  : public puzzler::IsingPuzzle
{
public:
  IsingTbbProvider()
  {}

  void create_bonds(ILog *log, unsigned n, uint32_t seed, unsigned step,  
                    uint32_t prob, const int *spins, int *up_down, int *left_right) const
  {
    log->LogVerbose("  create_bonds %u", step);
    tbb::parallel_for((unsigned) 0, n, [&](unsigned y){
      tbb::parallel_for((unsigned) 0, n, [&](unsigned x){
        bool sC=spins[y*n+x];
        
        bool sU=spins[ ((y+1)%n)*n + x ];
        if(sC!=sU){
          up_down[y*n+x]=0;
        }else{
          up_down[y*n+x]=hrng(seed, rng_group_bond_ud, step, y*n+x) < prob;
        }

        bool sR=spins[ y*n + (x+1)%n ];
        if(sC!=sR){
          left_right[y*n+x]=0;
        }else{
          left_right[y*n+x]=hrng(seed, rng_group_bond_lr, step, y*n+x) < prob;
        }
      });
    });
  }


  void create_clusters(ILog *log, unsigned n, uint32_t /*seed*/, unsigned step, 
                        const int *up_down, const int *left_right, unsigned *cluster) const
  {
    log->LogVerbose("  create_clusters %u", step);

    tbb::parallel_for((unsigned) 0, n*n, [&](unsigned i){
      cluster[i]=i;
    });

    bool finished=false;
    unsigned diameter=0;
    while(!finished){
      diameter++;
      finished=true;
      tbb::parallel_for((unsigned) 0, n, [&](unsigned y){
        tbb::parallel_for((unsigned) 0, n, [&](unsigned x){
          unsigned prev=cluster[y*n+x];
          unsigned curr=prev;
          if(left_right[y*n+x]){
            curr=std::min(curr, cluster[y*n+(x+1)%n]);
          }
          if(left_right[y*n+(x+n-1)%n]){
            curr=std::min(curr, cluster[y*n+(x+n-1)%n]);
          }
          if(up_down[y*n+x]){
            curr=std::min(curr, cluster[ ((y+1)%n)*n+x]);
          }
          if(up_down[((y+n-1)%n)*n+x]){
            curr=std::min(curr, cluster[ ((y+n-1)%n)*n+x]);
          }
          if(curr!=prev){
            cluster[y*n+x]=curr;
            finished=false;
          }
        });
      });
    }
    log->LogVerbose("    diameter %u", diameter);
  }

  void flip_clusters(ILog *log, unsigned n, uint32_t seed, unsigned step, unsigned *clusters, int *spins) const
  {
    log->LogVerbose("  flip_clusters %u", step);

    tbb::parallel_for((unsigned) 0, n*n, [&](unsigned i){
      unsigned cluster=clusters[i];
      if(hrng(seed, rng_group_flip, step, cluster) >> 31){
        spins[i] ^= 1;
      }
    });
  }

  void count_clusters(ILog *log, unsigned n, uint32_t /*seed*/, unsigned step, const unsigned *clusters, unsigned *counts, unsigned &nClusters) const
  {
    log->LogVerbose("  count_clusters %u", step);

    tbb::parallel_for((unsigned) 0, n*n, [&](unsigned i){
      counts[i]=0;
    });

    nClusters=0;
    for(unsigned i=0; i<n*n; i++){
      nClusters += !counts[clusters[i]]++ ? 1 : 0;
    }

  }

	virtual void Execute(
			   puzzler::ILog *log,
			   const puzzler::IsingInput *pInput,
				 puzzler::IsingOutput *pOutput
			   ) const override
	{
      auto begin = std::chrono::high_resolution_clock::now();
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

      for(unsigned i = 0; i < n; i++){
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
          for(unsigned y=0; y<n; y++){
            for(unsigned x=0; x<n; x++){
              dst<<(spins[y*n+x]?"+":" ");
            }
            dst<<"\n";
          }
        });
      }
      
      pOutput->history=stats;
      log->LogInfo("Finished");
      auto end = std::chrono::high_resolution_clock::now();
std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count() << "ns" << std::endl;
    }

};
};
#endif
