#pragma once

#include <string>
#include <vector>
#include "Core/Memory.h"

#include "OptixContext.h"
#include "OptixPipeline.h"
#include "Optix7.h"


namespace LTSE::Graphics
{
    using namespace LTSE::Core;

    struct OptixModuleObject
    {
        OptixModule RTObject = nullptr;

        OptixModuleObject() = default;
        OptixModuleObject( const std::string a_LaunchParameterVariableName, const char *a_PtxCode, Ref<OptixDeviceContextObject> a_RTContext );

        ~OptixModuleObject();

        OptixPipelineCompileOptions GetPipelineCompileOptions() { return m_PipelineCompileOptions; }

        OptixPipelineLinkOptions GetPipelineLinkOptions() { return m_PipelineLinkOptions; };

        void CreateMissGroup( std::string a_EntryName );
        void CreateRayGenGroup( std::string a_EntryName );
        void CreateHitGroup( std::string a_ClosestHitEntryName, std::string a_AnyHitHitEntryName );

        Ref<OptixPipelineObject> CreatePipeline();

        std::vector<Ref<OptixProgramGroupObject>> m_RayGenProgramGroups = {};
        std::vector<Ref<OptixProgramGroupObject>> m_HitProgramGroups    = {};
        std::vector<Ref<OptixProgramGroupObject>> m_MissProgramGroups   = {};

      private:
        Ref<OptixDeviceContextObject> m_RTContext = nullptr;
        OptixPipelineLinkOptions m_PipelineLinkOptions{};
        OptixPipelineCompileOptions m_PipelineCompileOptions{};
        const std::string m_LaunchParameterVariableName;
    };

} // namespace LTSE::Graphics