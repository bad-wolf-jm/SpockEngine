#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "Core/GraphicContext//Buffer.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext//RenderContext.h"

#include "Scene/ParticleData.h"
#include "Scene/VertexData.h"

#include "Graphics/Implementation/Vulkan/VkRenderPass.h"
#include "SceneRenderPipeline.h"

namespace LTSE::Graphics
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

        Ref<Internal::sVkRenderPassObject> RenderPass = nullptr;

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

    class ParticleSystemRenderer : public LTSE::Core::SceneRenderPipeline<PositionData>
    {
      public:
        struct ParticleData
        {
            math::mat4 Model         = math::mat4( 1.0f );
            uint32_t   ParticleCount = 0;
            float      ParticleSize  = 0.0f;

            Ref<Buffer> Particles = nullptr;

            ParticleData()                       = default;
            ParticleData( const ParticleData & ) = default;
        };

        ParticleRendererCreateInfo Spec;
        Ref<DescriptorSetLayout>   PipelineLayout = nullptr;

        ParticleSystemRenderer() = default;
        ParticleSystemRenderer(
            GraphicContext &a_GraphicContext, RenderContext &a_RenderContext, ParticleRendererCreateInfo a_CreateInfo );

        std::vector<Ref<DescriptorSetLayout>> GetDescriptorSetLayout();
        std::vector<sPushConstantRange>       GetPushConstantLayout();

        ~ParticleSystemRenderer() = default;

        void Render( math::mat4 a_Projection, math::mat4 a_View, RenderContext &aRenderContext, ParticleData &a_ParticleData );

      protected:
        Ref<Buffer>        mParticleVertices  = nullptr;
        Ref<Buffer>        mParticleIndices   = nullptr;
        Ref<Buffer>        mCameraBuffer      = nullptr;
        Ref<DescriptorSet> mCameraDescriptors = nullptr;
    };

} // namespace LTSE::Graphics
