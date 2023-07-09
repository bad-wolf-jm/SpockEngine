#pragma once

#include <filesystem>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "Graphics/API.h"

#include "Scene/ParticleData.h"
#include "Scene/VertexData.h"

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
        float               LineWidth      = 1.0f;
        Ref<IShaderProgram> VertexShader   = nullptr;
        Ref<IShaderProgram> FragmentShader = nullptr;

        Ref<IRenderContext> RenderPass = nullptr;

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
            std::size_t h3 = std::hash<std::size_t>()( (size_t)node.VertexShader.get() );
            std::size_t h4 = std::hash<std::size_t>()( (size_t)node.FragmentShader.get() );

            return h2 ^ h3 ^ h4;
        }
    };

    class ParticleSystemRenderer
    {
      public:
        struct ParticleData
        {
            math::mat4 Model         = math::mat4( 1.0f );
            uint32_t   ParticleCount = 0;
            float      ParticleSize  = 0.0f;

            Ref<IGraphicBuffer> Particles = nullptr;

            ParticleData()                       = default;
            ParticleData( const ParticleData & ) = default;
        };

        ParticleRendererCreateInfo Spec;
        Ref<IDescriptorSetLayout>  PipelineLayout = nullptr;

        ParticleSystemRenderer() = default;

        ParticleSystemRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext,
                                ParticleRendererCreateInfo aCreateInfo );

        ~ParticleSystemRenderer() = default;

        void Render( math::mat4 aProjection, math::mat4 aView, Ref<IRenderContext> aRenderContext, ParticleData &aParticleData );

      protected:
        Ref<IGraphicContext>   mGraphicContext    = nullptr;
        Ref<IGraphicsPipeline> mPipeline          = nullptr;
        Ref<IGraphicBuffer>    mParticleVertices  = nullptr;
        Ref<IGraphicBuffer>    mParticleIndices   = nullptr;
        Ref<IGraphicBuffer>    mCameraBuffer      = nullptr;
        Ref<IDescriptorSet>    mCameraDescriptors = nullptr;
    };

} // namespace SE::Graphics
