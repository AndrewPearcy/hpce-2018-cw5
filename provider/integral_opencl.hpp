#ifndef user_integral_opencl_hpp
#define user_integral_opencl_hpp

#define CL_USE_DEPRICATE_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 120

#include "puzzler/puzzles/integral.hpp"
#include <fstream>
#include <streambuf>
#include "CL/cl.hpp"


namespace puzzler{
class IntegralOpenCLProvider
  : public puzzler::IntegralPuzzle
{
public:
  IntegralOpenCLProvider()
  {}

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

    void kernel_i1_i2_i3(unsigned i1, unsigned i2, unsigned i3, 
                            unsigned r, 
                            const float range,
                            const float *M,
                            const float *C,
                            const float *bounds,
                            double *acc) const {
        float x1= -range/2 + range * (i1/(float)r);
        float x2= -range/2 + range * (i2/(float)r);
        float x3= -range/2 + range * (i3/(float)r);

        float x[3]={x1,x2,x3};
        *acc += mpdf(r, range, x, M, C, bounds);
    }


	virtual void Execute(
			   puzzler::ILog *log,
			   const puzzler::IntegralInput *pInput,
				puzzler::IntegralOutput *pOutput
			   ) const override
	{
        unsigned r=pInput->resolution;

        const float range=12;

        double acc=0;

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

        cl::Context context(devices);

        std::string kernelSource=LoadSource("integral_kernel.cl");

        cl::Program::Sources sources;	// A vector of (data,length) pairs
        sources.push_back(std::make_pair(kernelSource.c_str(), kernelSource.size()+1));	// push on our single string

        cl::Program program(context, sources);
        try{
            program.build(devices);
        } catch(...) {
           for(unsigned i=0;i<devices.size();i++){
		        std::cerr<<"Log for device "<<devices[i].getInfo<CL_DEVICE_NAME>()<<":\n\n";
		        std::cerr<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i])<<"\n\n";
	        }
	        throw; 
        }
        
        size_t buffM_size = sizeof(float)*(*&pInput->M).size();
        cl::Buffer buffM(context, CL_MEM_WRITE_ONLY, buffM_size);  
        size_t buffC_size = sizeof(float)*(*&pInput->C).size();
        cl::Buffer buffC(context, CL_MEM_WRITE_ONLY, buffC_size);
        size_t buffBounds_size = sizeof(float)*(*&pInput->bounds).size();
        cl::Buffer buffBounds(context, CL_MEM_WRITE_ONLY, 
                                buffBounds_size);
        cl::Buffer buffAcc(context, CL_MEM_READ_WRITE, sizeof(double));

        cl::Kernel kernel(program, "kernel_i1_i2_i3");

        kernel.setArg(0, r);
        kernel.setArg(1, range);
        kernel.setArg(2, buffM);
        kernel.setArg(3, buffC);
        kernel.setArg(4, buffBounds);
        kernel.setArg(5, buffAcc);
        
        cl::CommandQueue queue(context, device);

        queue.enqueueWriteBuffer(buffM, CL_TRUE, 0, buffM_size, &pInput->M[0]);
        queue.enqueueWriteBuffer(buffC, CL_TRUE, 0, buffC_size, &pInput->C[0]);
        queue.enqueueWriteBuffer(buffBounds, CL_TRUE, 0, buffBounds_size, &pInput->bounds[0]);
        queue.enqueueWriteBuffer(buffAcc, CL_TRUE, 0, sizeof(double), &acc);


        cl::NDRange offset(0,0,0);          // Start iterations from 0,0,0
        cl::NDRange globalSize(r,r,r);      
        cl::NDRange localSize=cl::NullRange;

        queue.enqueueNDRangeKernel(kernel, offset, globalSize, localSize);	
        queue.enqueueBarrierWithWaitList();
        queue.enqueueReadBuffer(buffAcc, CL_TRUE, 0, sizeof(double), &acc); 
        

        /*
        for(unsigned i1=0; i1<r; i1++){
            for(unsigned i2=0; i2<r; i2++){
                for(unsigned i3=0; i3<r; i3++){
                    kernel_i1_i2_i3(i1, i2, i3, r, range,
                                    &pInput->M[0], 
                                    &pInput->C[0], 
                                    &pInput->bounds[0], 
                                    &acc);
                }
            }
        }
        */

        log->LogInfo("Integral = %g", acc);
        pOutput->value=acc;
	}

};
};

#endif
