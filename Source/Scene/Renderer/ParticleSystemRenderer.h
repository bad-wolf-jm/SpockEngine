#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

// #include "Graphics/Vulkan/VkGpuBuffer.h"
// #include "Graphics/Vulkan/VkRenderContext.h"
// #include "Graphics/Vulkan/DescriptorSet.h"
// #include "Graphics/Vulkan/VkGraphicsPipeline.h"
// #include "Graphics/Vulkan/IGraphicContext.h"
// #include "Graphics/Vulkan/VkRenderPass.h"
#include "Graphics/API.h"

#include "Scene/ParticleData.h"
#include "Scene/VertexData.h"

#include "SceneRenderPipeline.h"

namespace SE::Graphics
{

    using namespace math;
    namespace fs = std::filesystem;

    struct CameraViewUniforms
    {
        mat4  Model;
        mat4  View;
        mat4  Projection;
        float ParticleSize;
    };

    struct ParticleRendererCreateInfo
    {
        float    LineWidth      = 1.0f;
        fs::path VertexShader   = "";
        fs::path FragmentShader = "";

        Ref<VkRenderPass> RenderPass = nullptr;

        bool operator==( const ParticleRendererCreateInfo &p ) const
        {
            return ( VertexShader == p.VertexShader ) && ( FragmentShader == p.FragmentShader );
        }
    };

    struct ParticleSystemRendererCreateInfoHash
    {
        size_t operator()( const ParticleRendererCreateInfo &node ) const
        {
            std::size_t h2 = std::hash<float>()( node.LineWidth );
            std::size_t h3 = std::hash<std::string>()( node.VertexShader.string() );
            std::size_t h4 = std::hash<std::string>()( node.FragmentShader.string() );

            return h2 ^ h3 ^ h4;
        }
    };

    class ParticleSystemRenderer : public SE::Core::SceneRenderPipeline<PositionData>
    {
      public:
        struct ParticleData
        {
            math::mat4 Model         = math::mat4( 1.0f );
            uint32_t   ParticleCount = 0;
            float      ParticleSize  = 0.0f;

            Ref<VkGpuBuffer> Particles = nullptr;

            ParticleData()                       = default;
            ParticleData( const ParticleData & ) = default;
        };

        ParticleRendererCreateInfo Spec;
        Ref<DescriptorSetLayout>   PipelineLayout = nullptr;

        ParticleSystemRenderer() = default;

        ParticleSystemRenderer( Ref<IGraphicContext> aGraphicContext, VkRenderContext &aRenderContext,
                                ParticleRendererCreateInfo aCreateInfo );

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

        ~ParticleSystemRenderer() = default;

        void Render( math::mat4 aProjection, math::mat4 aView, VkRenderContext &aRenderContext, ParticleData &aParticleData );

      protected:
        Ref<VkGpuBuffer>   mParticleVertices  = nullptr;
        Ref<VkGpuBuffer>   mParticleIndices   = nullptr;
        Ref<VkGpuBuffer>   mCameraBuffer      = nullptr;
        Ref<DescriptorSet> mCameraDescriptors = nullptr;
    };

} // namespace SE::Graphics
