#ifndef user_ising_opencl_hpp
#define user_ising_opencl_hpp

#include "puzzler/puzzles/ising.hpp"
#include "tbb/parallel_for.h"

namespace puzzler{
class IsingOpenCLProvider
  : public puzzler::IsingPuzzle
{
public:
  std::string LoadSource(const char *fileName) const
  {
	  std::string baseDir="provider";
	  if(getenv("HPCE_CL_SRC_DIR")){
	    baseDir=getenv("HPCE_CL_SRC_DIR");
    }
	
	  std::string fullName=baseDir+"/"+fileName;
	
	  // Open a read-only binary stream over the file
	  std::ifstream src(fullName, std::ios::in | std::ios::binary);
	  if(!src.is_open())
		  throw std::runtime_error("LoadSource : Couldn't load cl file from '"+fullName+"'.");
	
	  // Read all characters of the file into a string
	  return std::string(
		  (std::istreambuf_iterator<char>(src)), // Node the extra brackets.
           std::istreambuf_iterator<char>()
	  );
  }

  mutable size_t vector_size;
  mutable size_t buff_size;

  cl::Program program;
  cl::Context context;
  mutable cl::CommandQueue queue;

  mutable cl::Buffer buffSpins;
  mutable cl::Buffer buffUp_Down;
  mutable cl::Buffer buffLeft_Right;
  mutable cl::Buffer buffClusters;
  mutable cl::Buffer buffCounts;

  IsingOpenCLProvider()
  {
    //Open CL Device selection
    std::vector<cl::Platform> platforms;

    cl::Platform::get(&platforms);
    if(platforms.size()==0)
	    throw std::runtime_error("No OpenCL platforms found.");

    std::cerr<<"Found "<<platforms.size()<<" platforms\n";
    for(unsigned i=0;i<platforms.size();i++){
	    std::string vendor=platforms[i].getInfo<CL_PLATFORM_VENDOR>();
	    std::cerr<<"  Platform "<<i<<" : "<<vendor<<"\n";
    }

    int selectedPlatform=0;
    if(getenv("HPCE_SELECT_PLATFORM")){
	    selectedPlatform=atoi(getenv("HPCE_SELECT_PLATFORM"));
    }
    std::cerr<<"Choosing platform "<<selectedPlatform<<"\n";
    cl::Platform platform=platforms.at(selectedPlatform);

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(devices.size()==0){
	    throw std::runtime_error("No opencl devices found.\n");
    }

    std::cerr<<"Found "<<devices.size()<<" devices\n";
    for(unsigned i=0;i<devices.size();i++){
	    std::string name=devices[i].getInfo<CL_DEVICE_NAME>();
	    std::cerr<<"  Device "<<i<<" : "<<name<<"\n";
    }

    int selectedDevice=0;
    if(getenv("HPCE_SELECT_DEVICE")){
	    selectedDevice=atoi(getenv("HPCE_SELECT_DEVICE"));
    }
    std::cerr<<"Choosing device "<<selectedDevice<<"\n";
    cl::Device device=devices.at(selectedDevice);

    context = cl::Context(devices);

    std::string kernelSource=LoadSource("ising_kernel.cl");

    cl::Program::Sources sources;	// A vector of (data,length) pairs
    sources.push_back(std::make_pair(kernelSource.c_str(), kernelSource.size()+1));	// push on our single string

    program = cl::Program(context, sources);
    try{
        program.build(devices);
    } catch(...) {
       for(unsigned i=0;i<devices.size();i++){
		    std::cerr<<"Log for device "<<devices[i].getInfo<CL_DEVICE_NAME>()<<":\n\n";
		    std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i])<<"\n\n";
	    }
	    throw; 
    } 

      queue = cl::CommandQueue(context, device);
  }

  void create_bonds(ILog *log, unsigned n, uint32_t seed, 
                    unsigned step,  uint32_t prob) const 
  {
    log->LogVerbose("  create_bonds %u", step);

    cl::Kernel create_bonds_kernel(program, "kernel_create_bonds");
    
    unsigned vector_size=n*n;

    create_bonds_kernel.setArg(0, n);
    create_bonds_kernel.setArg(1, seed);
    create_bonds_kernel.setArg(2, step);
    create_bonds_kernel.setArg(3, prob);
    create_bonds_kernel.setArg(4, buffSpins);
    create_bonds_kernel.setArg(5, buffUp_Down);
    create_bonds_kernel.setArg(6, buffLeft_Right);

    
    cl::NDRange offset(0,0);
    cl::NDRange globalSize(n,n);
    cl::NDRange localSize=cl::NullRange;
    

    queue.enqueueNDRangeKernel(create_bonds_kernel, offset, globalSize, localSize);
    queue.enqueueBarrierWithWaitList();

    /*
    for(unsigned y=0; y<n; y++){
      for(unsigned x=0; x<n; x++){
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
      }
    }
    */
  }

  void create_clusters(ILog *log, unsigned n, uint32_t /*seed*/, unsigned step) const
  {
    log->LogVerbose("  create_clusters %u", step);

    cl::Kernel populateKernel(program, "kernel_populateClusters");

    cl::NDRange populateOffset(0);
    cl::NDRange populateGlobalSize(vector_size);
    cl::NDRange populateLocalSize;

    populateKernel.setArg(0, buffClusters);
    queue.enqueueNDRangeKernel(populateKernel, populateOffset, populateGlobalSize, populateLocalSize);
    queue.enqueueBarrierWithWaitList();

    cl::Kernel kernel(program, "kernel_create_clusters");
    size_t buffFinished_size = sizeof(bool);
    cl::Buffer buffFinished(context, CL_MEM_WRITE_ONLY, buffFinished_size);

    cl::NDRange offset(0,0);
    cl::NDRange globalSize(n,n);
    cl::NDRange localSize=cl::NullRange;


    kernel.setArg(0, n);
    kernel.setArg(1, buffUp_Down);
    kernel.setArg(2, buffLeft_Right);
    kernel.setArg(3, buffClusters);
    kernel.setArg(4, buffFinished);

    bool finished=false;
    unsigned diameter=0;
    while(!finished){
      diameter++;
      finished=true;


      queue.enqueueWriteBuffer(buffFinished, CL_TRUE, 0, buffFinished_size, &finished);

      queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);
      queue.enqueueBarrierWithWaitList();
      queue.enqueueReadBuffer(buffFinished, CL_TRUE, 0, buffFinished_size, &finished);


      /*
      for(unsigned y=0; y<n; y++){
        for(unsigned x=0; x<n; x++){
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
        }
      }
      */
    }
    //queue.enqueueReadBuffer(buffClusters, CL_TRUE, 0, buff_size, cluster);
    log->LogVerbose("    diameter %u", diameter);
  }

  void flip_clusters(ILog *log, unsigned n, uint32_t seed, unsigned step) const
  {
    log->LogVerbose("  flip_clusters %u", step);

    cl::Kernel kernel(program, "kernel_flip_clusters");

    unsigned vector_size=n*n;

    cl::NDRange offset(0);
    cl::NDRange globalSize(vector_size);
    cl::NDRange localSize=cl::NullRange;

    kernel.setArg(0, seed);
    kernel.setArg(1, step);
    kernel.setArg(2, buffClusters);
    kernel.setArg(3, buffSpins);

    queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);
    queue.enqueueBarrierWithWaitList();

    /*
    for(unsigned i=0; i<n*n; i++){
      unsigned cluster=clusters[i];
      if(hrng(seed, rng_group_flip, step, cluster) >> 31){
        spins[i] ^= 1;
      }
    }
    */
  }

  
  void count_clusters(ILog *log, unsigned n, uint32_t /*seed*/, unsigned step, 
                        unsigned *counts, unsigned &nClusters) const
  {
    log->LogVerbose("  count_clusters %u", step);

    /*
    tbb::parallel_for((unsigned) 0, n*n, [&](unsigned i){
      counts[i]=0; 
    });
    */

    cl::Kernel kernel(program, "kernel_nClusters");

    nClusters=0;
    size_t buff_size = sizeof(unsigned) * n*n;
    cl::Buffer buffNClusters(context, CL_MEM_WRITE_ONLY, sizeof(unsigned));

    kernel.setArg(0, buffClusters);
    kernel.setArg(1, buffCounts);
    kernel.setArg(2, buffNClusters);

    queue.enqueueWriteBuffer(buffCounts, CL_TRUE, 0, buff_size, counts);
    queue.enqueueWriteBuffer(buffNClusters, CL_TRUE, 0, sizeof(unsigned), &nClusters);
    
    cl::NDRange offset(0);
    cl::NDRange globalSize(n*n);
    cl::NDRange localSize=cl::NullRange;

    queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);
    queue.enqueueBarrierWithWaitList();
    queue.enqueueReadBuffer(buffNClusters, CL_TRUE, 0, sizeof(unsigned), &nClusters);

    /*
    for(unsigned i=0; i<n*n; i++){
      nClusters += !counts[clusters[i]]++ ? 1 : 0;
    }
    
    for(unsigned i=0; i<n*n; i++){
      if(counts[i]){
        nClusters++;
      }
    }
    */
  }


	virtual void Execute(
			   puzzler::ILog *log,
			   const puzzler::IsingInput *pInput,
				 puzzler::IsingOutput *pOutput
			   ) const override
	{
      log->LogInfo("Building world");
      unsigned n=pInput->n;
      vector_size=n*n;
      uint32_t prob=pInput->prob;
      uint32_t seed=pInput->seed;
      std::vector<int> spins(n*n);
      std::vector<int> left_right(n*n);
      std::vector<int> up_down(n*n);
      std::vector<unsigned> clusters(n*n);
      std::vector<unsigned> counts(n*n, 0);
      vector_size=n*n;
      tbb::parallel_for((unsigned long) 0, vector_size, [&](unsigned i){
        spins[i]=hrng(seed, rng_group_init, 0, i) & 1;
      });

      log->LogInfo("Doing iterations");
      std::vector<uint32_t> stats(n);

      //Create Buffers
      buff_size = sizeof(int) * vector_size;
      buffSpins = cl::Buffer(context, CL_MEM_READ_WRITE, buff_size);
      buffLeft_Right = cl::Buffer(context, CL_MEM_READ_WRITE, buff_size);
      buffUp_Down = cl::Buffer(context, CL_MEM_READ_WRITE, buff_size);
      buffClusters = cl::Buffer(context, CL_MEM_READ_WRITE, buff_size);
      buffCounts = cl::Buffer(context, CL_MEM_READ_ONLY, buff_size);

      queue.enqueueWriteBuffer(buffSpins, CL_TRUE, 0, buff_size, &spins[0]);
      queue.enqueueWriteBuffer(buffUp_Down, CL_TRUE, 0, buff_size, &up_down[0]);
      queue.enqueueWriteBuffer(buffLeft_Right, CL_TRUE, 0, buff_size, &left_right[0]);
      queue.enqueueWriteBuffer(buffClusters, CL_TRUE, 0, buff_size, &clusters[0]);

      for(unsigned i=0; i<n; i++){
        log->LogVerbose("  Iteration %u", i);
        create_bonds(log, n, seed, i, prob);
        create_clusters(log,  n, seed, i);
        flip_clusters(log,  n, seed, i);
        count_clusters(log,  n, seed, i, &counts[0], stats[i]);
        log->LogVerbose("  clusters count is %u", stats[i]);

        log->Log( Log_Debug, [&](std::ostream &dst){
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
	}

};
};

#endif