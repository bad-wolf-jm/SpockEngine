#include "ParticleSystemRenderer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

using namespace SE::Core;

namespace SE::Graphics
{

    std::vector<Ref<DescriptorSetLayout>> ParticleSystemRenderer::GetDescriptorSetLayout() { return { PipelineLayout }; }

    std::vector<sPushConstantRange> ParticleSystemRenderer::GetPushConstantLayout() { return {}; };

    ParticleSystemRenderer::ParticleSystemRenderer( Ref<IGraphicContext> aGraphicContext, VkRenderContext &aRenderContext,
                                                    ParticleRendererCreateInfo aCreateInfo )
        : SceneRenderPipeline<PositionData>( aGraphicContext )
        , Spec{ aCreateInfo }
    {
        mCameraBuffer =
            New<VkGpuBuffer>( mGraphicContext, eBufferType::UNIFORM_BUFFER, true, false, true, true, sizeof( CameraViewUniforms ) );

        SceneRenderPipelineCreateInfo lCreateInfo{};
        lCreateInfo.IsTwoSided           = true;
        lCreateInfo.LineWidth            = aCreateInfo.LineWidth;
        lCreateInfo.VertexShader         = aCreateInfo.VertexShader;
        lCreateInfo.FragmentShader       = aCreateInfo.FragmentShader;
        lCreateInfo.RenderPass           = aRenderContext.GetRenderPass();
        lCreateInfo.InstanceBufferLayout = Particle::GetDefaultLayout();

        DescriptorSetLayoutCreateInfo lPipelineLayoutCI{};
        lPipelineLayoutCI.Bindings = {
            DescriptorBindingInfo{ 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::VERTEX } } };
        PipelineLayout = New<DescriptorSetLayout>( mGraphicContext, lPipelineLayoutCI );

        Initialize( lCreateInfo );

        mCameraDescriptors = New<DescriptorSet>( mGraphicContext, PipelineLayout );
        mCameraDescriptors->Write( mCameraBuffer, false, 0, sizeof( CameraViewUniforms ), 0 );

        std::vector<math::vec3> lVertexBufferData = {
            { -.5f, -.5f, 0.0f }, { -.5f, .5f, 0.0f }, { .5f, .5f, 0.0f }, { .5f, -.5f, 0.0f } };
        std::vector<uint32_t> lIndexBufferData = { 0, 2, 1, 0, 3, 2 };

        mParticleVertices =
            New<VkGpuBuffer>( mGraphicContext, lVertexBufferData, eBufferType::VERTEX_BUFFER, false, false, false, true );
        mParticleIndices =
            New<VkGpuBuffer>( mGraphicContext, lIndexBufferData, eBufferType::INDEX_BUFFER, false, false, false, true );
    }

    void ParticleSystemRenderer::Render( math::mat4 aProjection, math::mat4 aView, VkRenderContext &aRenderContext,
                                         ParticleData &aParticleData )
    {
        if( aParticleData.Particles == nullptr ) return;

        CameraViewUniforms l_View{ aParticleData.Model, aView, aProjection, aParticleData.ParticleSize };

        mCameraBuffer->Write( l_View );
        aRenderContext.Bind( Pipeline );
        aRenderContext.Bind( mCameraDescriptors, 0, -1 );
        aRenderContext.Bind( mParticleVertices, mParticleIndices, 0 );
        aRenderContext.Bind( aParticleData.Particles, 1 );
        aRenderContext.Draw( 6, 0, 0, aParticleData.ParticleCount, 0 );
    }

} // namespace SE::Graphics
