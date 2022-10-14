#include "OptixModule.h"
#include "Core/Logging.h"
#include "Core/Memory.h"
#include "Core/Cuda/CudaAssert.h"


namespace LTSE::Graphics
{

    OptixModuleObject::OptixModuleObject( const std::string a_LaunchParameterVariableName, const char *a_PtxCode, Ref<OptixDeviceContextObject> a_RTContext )
        : m_RTContext{ a_RTContext }
        , m_LaunchParameterVariableName{ a_LaunchParameterVariableName }
    {
        OptixModuleCompileOptions moduleCompileOptions{};

        moduleCompileOptions.maxRegisterCount = 50;
        moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        // pipelineCompileOptions = {};
        m_PipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        m_PipelineCompileOptions.usesMotionBlur                   = false;
        m_PipelineCompileOptions.numPayloadValues                 = 2;
        m_PipelineCompileOptions.numAttributeValues               = 2;
        m_PipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        m_PipelineCompileOptions.pipelineLaunchParamsVariableName = m_LaunchParameterVariableName.c_str();

        m_PipelineLinkOptions.maxTraceDepth = 2;

        const std::string ptxCode = a_PtxCode;

        OPTIX_CHECK( optixModuleCreateFromPTX( m_RTContext->RTObject, &moduleCompileOptions, &m_PipelineCompileOptions, ptxCode.c_str(), ptxCode.size(), NULL, NULL, &RTObject ) );
    }

    OptixModuleObject::~OptixModuleObject() { OPTIX_CHECK( optixModuleDestroy( RTObject ) ); }

    void OptixModuleObject::CreateMissGroup( std::string a_EntryName )
    {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc       = {};
        pgDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module                 = RTObject;
        pgDesc.miss.entryFunctionName      = a_EntryName.c_str();

        m_MissProgramGroups.push_back( LTSE::Core::New<OptixProgramGroupObject>( pgDesc, pgOptions, m_RTContext ) );
    }

    void OptixModuleObject::CreateRayGenGroup( std::string a_EntryName )
    {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc       = {};
        pgDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module               = RTObject;
        pgDesc.raygen.entryFunctionName    = a_EntryName.c_str();

        m_RayGenProgramGroups.push_back( LTSE::Core::New<OptixProgramGroupObject>( pgDesc, pgOptions, m_RTContext ) );
    }

    void OptixModuleObject::CreateHitGroup( std::string a_ClosestHitEntryName, std::string a_AnyHitHitEntryName )
    {
        OptixProgramGroupOptions pgOptions  = {};
        OptixProgramGroupDesc pgDesc        = {};
        pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH            = RTObject;
        pgDesc.hitgroup.entryFunctionNameCH = a_ClosestHitEntryName.c_str();
        pgDesc.hitgroup.moduleAH            = RTObject;
        pgDesc.hitgroup.entryFunctionNameAH = a_AnyHitHitEntryName.c_str();

        m_HitProgramGroups.push_back( LTSE::Core::New<OptixProgramGroupObject>( pgDesc, pgOptions, m_RTContext ) );
    }

    Ref<OptixPipelineObject> OptixModuleObject::CreatePipeline()
    {
        std::vector<Ref<OptixProgramGroupObject>> l_ProgramGroups;
        for( auto pg : m_RayGenProgramGroups )
            l_ProgramGroups.push_back( pg );
        for( auto pg : m_HitProgramGroups )
            l_ProgramGroups.push_back( pg );
        for( auto pg : m_MissProgramGroups )
            l_ProgramGroups.push_back( pg );
        return LTSE::Core::New<OptixPipelineObject>( m_PipelineLinkOptions, m_PipelineCompileOptions, l_ProgramGroups, m_RTContext );
    }

} // namespace LTSE::Graphics