#include "OptixModule.h"
#include "Core/Cuda/CudaAssert.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{

    OptixModuleObject::OptixModuleObject( const std::string aLaunchParameterVariableName, const char *aPtxCode,
                                          Ref<OptixDeviceContextObject> aRayTracingContext )
        : mRayTracingContext{ aRayTracingContext }
        , mLaunchParameterVariableName{ aLaunchParameterVariableName }
    {
        OptixModuleCompileOptions lModuleCompileOptions{};

        lModuleCompileOptions.maxRegisterCount = 50;
        lModuleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        lModuleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        mPipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        mPipelineCompileOptions.usesMotionBlur                   = false;
        mPipelineCompileOptions.numPayloadValues                 = 2;
        mPipelineCompileOptions.numAttributeValues               = 2;
        mPipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        mPipelineCompileOptions.pipelineLaunchParamsVariableName = mLaunchParameterVariableName.c_str();

        mPipelineLinkOptions.maxTraceDepth = 2;

        const std::string ptxCode = aPtxCode;

        OPTIX_CHECK( optixModuleCreateFromPTX( mRayTracingContext->mOptixObject, &lModuleCompileOptions, &mPipelineCompileOptions,
                                               ptxCode.c_str(), ptxCode.size(), NULL, NULL, &mOptixObject ) );
    }

    OptixModuleObject::~OptixModuleObject() { OPTIX_CHECK( optixModuleDestroy( mOptixObject ) ); }

    void OptixModuleObject::CreateMissGroup( std::string aEntryName )
    {
        OptixProgramGroupOptions lProgramGroupOptions{};
        OptixProgramGroupDesc    lProgramGroupDescription{};
        lProgramGroupDescription.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        lProgramGroupDescription.miss.module            = mOptixObject;
        lProgramGroupDescription.miss.entryFunctionName = aEntryName.c_str();

        mMissProgramGroups.push_back(
            SE::Core::New<OptixProgramGroupObject>( lProgramGroupDescription, lProgramGroupOptions, mRayTracingContext ) );
    }

    void OptixModuleObject::CreateRayGenGroup( std::string aEntryName )
    {
        OptixProgramGroupOptions lProgramGroupOptions{};
        OptixProgramGroupDesc    lProgramGroupDescription{};
        lProgramGroupDescription.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        lProgramGroupDescription.raygen.module            = mOptixObject;
        lProgramGroupDescription.raygen.entryFunctionName = aEntryName.c_str();

        mRayGenProgramGroups.push_back(
            SE::Core::New<OptixProgramGroupObject>( lProgramGroupDescription, lProgramGroupOptions, mRayTracingContext ) );
    }

    void OptixModuleObject::CreateHitGroup( std::string aClosestHitEntryName, std::string aAnyHitHitEntryName )
    {
        OptixProgramGroupOptions lProgramGroupOptions{};
        OptixProgramGroupDesc    lProgramGroupDescription{};
        lProgramGroupDescription.kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
        lProgramGroupDescription.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        lProgramGroupDescription.hitgroup.moduleCH            = mOptixObject;
        lProgramGroupDescription.hitgroup.entryFunctionNameCH = aClosestHitEntryName.c_str();
        lProgramGroupDescription.hitgroup.moduleAH            = mOptixObject;
        lProgramGroupDescription.hitgroup.entryFunctionNameAH = aAnyHitHitEntryName.c_str();

        mHitProgramGroups.push_back(
            SE::Core::New<OptixProgramGroupObject>( lProgramGroupDescription, lProgramGroupOptions, mRayTracingContext ) );
    }

    Ref<OptixPipelineObject> OptixModuleObject::CreatePipeline()
    {
        std::vector<Ref<OptixProgramGroupObject>> lProgramGroups;
        for( auto pg : mRayGenProgramGroups ) lProgramGroups.push_back( pg );
        for( auto pg : mHitProgramGroups ) lProgramGroups.push_back( pg );
        for( auto pg : mMissProgramGroups ) lProgramGroups.push_back( pg );
        return SE::Core::New<OptixPipelineObject>( mPipelineLinkOptions, mPipelineCompileOptions, lProgramGroups, mRayTracingContext );
    }

} // namespace SE::Graphics