#include "ParticleSystemRenderer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

using namespace SE::Core;

namespace SE::Graphics
{

    std::vector<Ref<DescriptorSetLayout>> ParticleSystemRenderer::GetDescriptorSetLayout() { return { PipelineLayout }; }

    std::vector<sPushConstantRange> ParticleSystemRenderer::GetPushConstantLayout() { return {}; };

    ParticleSystemRenderer::ParticleSystemRenderer( Ref<VkGraphicContext> a_GraphicContext, ARenderContext &aRenderContext,
                                                    ParticleRendererCreateInfo aCreateInfo )
        : SceneRenderPipeline<PositionData>( a_GraphicContext )
        , Spec{ aCreateInfo }
    {
        mCameraBuffer = New<VkGpuBuffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true,
                                          sizeof( CameraViewUniforms ) );

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
        std::vector<uint32_t> l_IndexBufferData = { 0, 2, 1, 0, 3, 2 };

        mParticleVertices =
            New<VkGpuBuffer>( mGraphicContext, lVertexBufferData, eBufferBindType::VERTEX_BUFFER, false, false, false, true );
        mParticleIndices =
            New<VkGpuBuffer>( mGraphicContext, l_IndexBufferData, eBufferBindType::INDEX_BUFFER, false, false, false, true );
    }

    void ParticleSystemRenderer::Render( math::mat4 a_Projection, math::mat4 a_View, ARenderContext &aRenderContext,
                                         ParticleData &a_ParticleData )
    {
        if( a_ParticleData.Particles == nullptr ) return;

        CameraViewUniforms l_View{ a_ParticleData.Model, a_View, a_Projection, a_ParticleData.ParticleSize };

        mCameraBuffer->Write( l_View );
        aRenderContext.Bind( Pipeline );
        aRenderContext.Bind( mCameraDescriptors, 0, -1 );
        aRenderContext.Bind( mParticleVertices, mParticleIndices, 0 );
        aRenderContext.Bind( a_ParticleData.Particles, 1 );
        aRenderContext.Draw( 6, 0, 0, a_ParticleData.ParticleCount, 0 );
    }

} // namespace SE::Graphics
