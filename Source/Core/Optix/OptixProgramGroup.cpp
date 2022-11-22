#include "OptixProgramGroup.h"
#include "Core/Cuda/CudaAssert.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{

    OptixProgramGroupObject::OptixProgramGroupObject( OptixProgramGroupDesc         aProgramGroupDescription,
                                                      OptixProgramGroupOptions      aProgramGroupOptions,
                                                      Ref<OptixDeviceContextObject> aRayTracingContext )
        : mRayTracingContext{ aRayTracingContext }
    {
        OPTIX_CHECK( optixProgramGroupCreate( mRayTracingContext->mOptixObject, &aProgramGroupDescription, 1, &aProgramGroupOptions,
                                              NULL, NULL, &mOptixObject ) );
    }

    OptixProgramGroupObject::~OptixProgramGroupObject() { OPTIX_CHECK( optixProgramGroupDestroy( mOptixObject ) ); }

} // namespace SE::Graphics