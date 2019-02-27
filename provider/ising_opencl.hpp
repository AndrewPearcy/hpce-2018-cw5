#ifndef user_ising_opencl_hpp
#define user_ising_opencl_hpp

#include "puzzler/puzzles/ising.hpp"

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


  cl::Program program;
  cl::Context context;
  cl::CommandQueue queue;

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
                    unsigned step,  uint32_t prob, 
                    const int *spins, int *up_down, int *left_right) const 
    {
      log->LogVerbose("  create_bonds %u", step);

      cl::Kernel create_bonds_kernel(program, "kernel_create_bonds");
    
      unsigned vector_size = n*n;
      size_t buffSpins_size = sizeof(int) * vector_size;
      cl::Buffer buffSpins(context, CL_MEM_READ_ONLY, buffSpins_size);
      size_t buffUp_Down_size = sizeof(int) * vector_size;
      cl::Buffer buffUp_Down(context, CL_MEM_WRITE_ONLY, buffUp_Down_size);
      size_t buffLeft_Right_size = sizeof(int) * vector_size;
      cl::Buffer buffLeft_Right(context, CL_MEM_WRITE_ONLY, buffLeft_Right_size);

      create_bonds_kernel.setArg(0, n);
      create_bonds_kernel.setArg(1, seed);
      create_bonds_kernel.setArg(2, step);
      create_bonds_kernel.setArg(3, prob);
      create_bonds_kernel.setArg(4, buffSpins);
      create_bonds_kernel.setArg(5, buffUp_Down);
      create_bonds_kernel.setArg(6, buffLeft_Right);

      cl::Event evCopiedSpins;
      queue.enqueueWriteBuffer(buffSpins, CL_FALSE, 0, buffSpins_size, spins, NULL, &evCopiedSpins);
      cl::Event evCopiedUp_Down;
      queue.enqueueWriteBuffer(buffUp_Down, CL_FALSE, 0, buffUp_Down_size, up_down, NULL, &evCopiedUp_Down);
      cl::Event evCopiedLeft_Right;
      queue.enqueueWriteBuffer(buffLeft_Right, CL_FALSE, 0, buffLeft_Right_size, 
                                left_right, NULL, &evCopiedLeft_Right);

      cl::NDRange offset(0,0);
      cl::NDRange globalSize(n,n);
      cl::NDRange localSize=cl::NullRange;
      

      std::vector<cl::Event> kernelDependencies{evCopiedSpins, evCopiedUp_Down, evCopiedLeft_Right};
      cl::Event evExecutedKernel;
      queue.enqueueNDRangeKernel(create_bonds_kernel, offset, globalSize, localSize,
                                  &kernelDependencies, &evExecutedKernel);
      std::vector<cl::Event> copyBackDependencies(1, evExecutedKernel);
      queue.enqueueReadBuffer(buffUp_Down, CL_TRUE, 0, buffUp_Down_size, up_down, &copyBackDependencies);
      queue.enqueueReadBuffer(buffLeft_Right, CL_TRUE, 0, buffLeft_Right_size, left_right); 

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
      for(unsigned i=0; i<n*n; i++){
        spins[i]=hrng(seed, rng_group_init, 0, i) & 1;
      }

      log->LogInfo("Doing iterations");
      std::vector<uint32_t> stats(n);

      for(unsigned i=0; i<n; i++){
        log->LogVerbose("  Iteration %u", i);
        create_bonds(   log, n, seed, i, prob, &spins[0], &up_down[0], &left_right[0]);
        create_clusters(log,  n, seed, i, &up_down[0], &left_right[0], &clusters[0]);
        flip_clusters(  log,  n, seed, i, &clusters[0], &spins[0]);
        count_clusters( log,  n, seed, i, &clusters[0], &counts[0], stats[i]);
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
