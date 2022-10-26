#include "ParticleSystemRenderer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

using namespace LTSE::Core;

namespace LTSE::Graphics
{

    std::vector<Ref<DescriptorSetLayout>> ParticleSystemRenderer::GetDescriptorSetLayout() { return { PipelineLayout }; }

    std::vector<sPushConstantRange> ParticleSystemRenderer::GetPushConstantLayout() { return {}; };

    ParticleSystemRenderer::ParticleSystemRenderer(
        GraphicContext &a_GraphicContext, RenderContext &a_RenderContext, ParticleRendererCreateInfo a_CreateInfo )
        : SceneRenderPipeline<PositionData>( a_GraphicContext )
        , Spec{ a_CreateInfo }
    {
        m_CameraBuffer =
            New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( CameraViewUniforms ) );

        SceneRenderPipelineCreateInfo l_CreateInfo{};
        l_CreateInfo.IsTwoSided           = true;
        l_CreateInfo.LineWidth            = a_CreateInfo.LineWidth;
        l_CreateInfo.VertexShader         = a_CreateInfo.VertexShader;
        l_CreateInfo.FragmentShader       = a_CreateInfo.FragmentShader;
        l_CreateInfo.RenderPass           = a_CreateInfo.RenderPass;
        l_CreateInfo.InstanceBufferLayout = Particle::GetDefaultLayout();

        DescriptorSetLayoutCreateInfo l_PipelineLayoutCI{};
        l_PipelineLayoutCI.Bindings = {
            DescriptorBindingInfo{ 0, Internal::eDescriptorType::UNIFORM_BUFFER, { Internal::eShaderStageTypeFlags::VERTEX } } };
        PipelineLayout = New<DescriptorSetLayout>( mGraphicContext, l_PipelineLayoutCI );

        Initialize( l_CreateInfo );

        m_CameraDescriptors = New<DescriptorSet>( mGraphicContext, PipelineLayout );
        m_CameraDescriptors->Write( m_CameraBuffer, false, 0, sizeof( CameraViewUniforms ), 0 );

        std::vector<math::vec3> g_vertex_buffer_data = {
            { -.5f, -.5f, 0.0f }, { -.5f, .5f, 0.0f }, { .5f, .5f, 0.0f }, { .5f, -.5f, 0.0f } };
        std::vector<uint32_t> l_IndexBufferData = { 0, 2, 1, 0, 3, 2 };

        m_ParticleVertices =
            New<Buffer>( mGraphicContext, g_vertex_buffer_data, eBufferBindType::VERTEX_BUFFER, false, false, false, true );
        m_ParticleIndices =
            New<Buffer>( mGraphicContext, l_IndexBufferData, eBufferBindType::INDEX_BUFFER, false, false, false, true );
    }

    void ParticleSystemRenderer::Render(
        math::mat4 a_Projection, math::mat4 a_View, RenderContext &aRenderContext, ParticleData &a_ParticleData )
    {
        if( a_ParticleData.Particles == nullptr ) return;

        CameraViewUniforms l_View{ a_ParticleData.Model, a_View, a_Projection, a_ParticleData.ParticleSize };

        m_CameraBuffer->Write( l_View );
        aRenderContext.Bind( Pipeline );
        aRenderContext.Bind( m_CameraDescriptors, 0, -1 );
        aRenderContext.Bind( m_ParticleVertices, m_ParticleIndices, 0 );
        aRenderContext.Bind( a_ParticleData.Particles, 1 );
        aRenderContext.Draw( 6, 0, 0, a_ParticleData.ParticleCount, 0 );
    }

} // namespace LTSE::Graphics
