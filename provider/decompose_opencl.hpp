#ifndef user_decompose_opencl_hpp
#define user_decompose_opencl_hpp

#include "puzzler/puzzles/decompose.hpp"
#include "tbb/parallel_for.h"
#include "CL/cl.hpp"
#include <fstream>    

namespace puzzler {
class DecomposeOpenCLProvider
  : public puzzler::DecomposePuzzle
{
public:
  
std::string LoadSource(const char *fileName) const
{
	std::string baseDir="provider/";
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

  DecomposeOpenCLProvider()
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
    size_t matrix_size = rr*cc;
 
 	// OpenCl Code
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
	
	cl::Context context(devices);
	
	const char *filename = "decompose_kernel.cl";	

	std::string kernelSource=LoadSource(filename);

	cl::Program::Sources sources;	// A vector of (data,length) pairs
	sources.push_back(std::make_pair(kernelSource.c_str(), kernelSource.size()+1));	// push on our single string

	cl::Program program(context, sources);
	try{
		program.build(devices);
	}catch(...){
		for(unsigned i=0;i<devices.size();i++){
			std::cerr<<"Log for device "<<devices[i].getInfo<CL_DEVICE_NAME>()<<":\n\n";
			std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i])<<"\n\n";
		}	
		throw;
	} 

	
    log->LogInfo("Building random matrix");
    std::vector<uint32_t> matrix(matrix_size);
    
    cl::Buffer buffMatrix(context, CL_MEM_WRITE_ONLY, matrix_size);
    cl::Kernel kernel(program, "kernel_make_bit");
    kernel.setArg(0, buffMatrix);
    kernel.setArg(1, pInput->seed);
    kernel.setArg(2, p);
    cl::CommandQueue queue(context, device); 
    queue.enqueueWriteBuffer(buffMatrix, CL_TRUE, 0, matrix_size, &matrix);
    cl::NDRange offset(0);
    cl::NDRange globalSize(matrix.size());
    cl::NDRange localSize=cl::NullRange;
    
    queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);	
    queue.enqueueBarrierWithWaitList();
    queue.enqueueReadBuffer(buffMatrix, CL_TRUE, 0, matrix_size, &matrix); 


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

    uint32_t kernel_make_bit(uint32_t seed, uint32_t input) const
    {
      const unsigned P = 7;
      const uint32_t PRIME32_1  = 2654435761U;
      const uint32_t PRIME32_2 = 2246822519U;
      seed += input * PRIME32_2;
      seed  = (seed<<13) | (seed>>(32-13));
      seed *= PRIME32_1;
      return seed % P;
    }

  void setupOpenCL() {  }


};
};
#endif
