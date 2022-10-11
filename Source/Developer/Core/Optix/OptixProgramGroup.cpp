#include "OptixProgramGroup.h"
#include "Core/Logging.h"
#include "Core/Memory.h"
#include "Cuda/CudaAssert.h"


namespace LTSE::Graphics
{

    OptixProgramGroupObject::OptixProgramGroupObject( OptixProgramGroupDesc a_ProgramGroupDescription, OptixProgramGroupOptions a_ProgramGroupOptions,
                                                      Ref<OptixDeviceContextObject> a_RTContext )
        : m_RTContext{ a_RTContext }
    {
        OPTIX_CHECK( optixProgramGroupCreate( m_RTContext->RTObject, &a_ProgramGroupDescription, 1, &a_ProgramGroupOptions, NULL, NULL, &RTObject ) );
    }

    OptixProgramGroupObject::~OptixProgramGroupObject() { OPTIX_CHECK( optixProgramGroupDestroy( RTObject ) ); }

} // namespace LTSE::Graphics