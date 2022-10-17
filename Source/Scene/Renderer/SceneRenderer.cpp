#include "SceneRenderer.h"

#include <chrono>
#include <gli/gli.hpp>

#include "Core/Vulkan/VkPipeline.h"

#include "Scene/Components/VisualHelpers.h"
#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Profiling/BlockTimer.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

#include "MeshRenderer.h"
#include "ParticleSystemRenderer.h"

namespace LTSE::Core
{

    using namespace math;
    using namespace LTSE::Core::EntityComponentSystem::Components;
    using namespace LTSE::Core::Primitives;

    DirectionalLightData::DirectionalLightData( const DirectionalLightComponent &a_Spec, math::mat4 a_Transform )
    {
        float l_Azimuth   = math::radians( a_Spec.Azimuth );
        float l_Elevation = math::radians( a_Spec.Elevation );

        Direction = math::vec3{ math::sin( l_Elevation ) * math::cos( l_Azimuth ), math::cos( l_Elevation ), math::sin( l_Elevation ) * math::sin( l_Azimuth ) };
        Color     = a_Spec.Color;
        Intensity = a_Spec.Intensity;
    }

    PointLightData::PointLightData( const PointLightComponent &a_Spec, math::mat4 a_Transform )
    {
        WorldPosition = a_Transform * math::vec4( a_Spec.Position, 1.0f );
        Color         = a_Spec.Color;
        Intensity     = a_Spec.Intensity;
    }

    SpotlightData::SpotlightData( const SpotlightComponent &a_Spec, math::mat4 a_Transform )
    {
        float l_Azimuth   = math::radians( a_Spec.Azimuth );
        float l_Elevation = math::radians( a_Spec.Elevation );

        WorldPosition   = a_Transform * math::vec4( a_Spec.Position, 1.0f );
        LookAtDirection = math::vec3{ math::sin( l_Elevation ) * math::cos( l_Azimuth ), math::cos( l_Elevation ), math::sin( l_Elevation ) * math::sin( l_Azimuth ) };
        Color           = a_Spec.Color;
        Intensity       = a_Spec.Intensity;
        Cone            = math::cos( math::radians( a_Spec.Cone / 2 ) );
    }

    SceneRenderer::SceneRenderer( Ref<Scene> a_World, RenderContext &a_RenderContext, Ref<LTSE::Graphics::Internal::sVkRenderPassObject> a_RenderPass )
        : mGraphicContext{ a_World->GetGraphicContext() }
        , m_World{ a_World }
        , m_RenderPass{ a_RenderPass }
    {
        m_SceneDescriptors = New<DescriptorSet>( mGraphicContext, MeshRenderer::GetCameraSetLayout( mGraphicContext ) );

        m_CameraUniformBuffer    = New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( WorldMatrices ) );
        m_ShaderParametersBuffer = New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( CameraSettings ) );
        m_SceneDescriptors->Write( m_CameraUniformBuffer, false, 0, sizeof( WorldMatrices ), 0 );
        m_SceneDescriptors->Write( m_ShaderParametersBuffer, false, 0, sizeof( CameraSettings ), 1 );

        TextureDescription l_EmptyTextureDescription{};
        l_EmptyTextureDescription.IsHostVisible       = false;
        l_EmptyTextureDescription.Usage               = { TextureUsageFlags::SAMPLED, TextureUsageFlags::TRANSFER_DESTINATION };
        l_EmptyTextureDescription.MinificationFilter  = SamplerFilter::LINEAR;
        l_EmptyTextureDescription.MagnificationFilter = SamplerFilter::LINEAR;
        l_EmptyTextureDescription.MipmapMode          = SamplerMipmap::LINEAR;
        l_EmptyTextureDescription.WrappingMode        = SamplerWrapping::REPEAT;
        l_EmptyTextureDescription.Sampled             = true;
        gli::texture2d l_EmptyTextureImage( gli::load_ktx( GetResourcePath( "textures\\empty.ktx" ).string() ) );
        m_EmptyTexture = New<Graphics::Texture2D>( mGraphicContext, l_EmptyTextureDescription, l_EmptyTextureImage );

        CoordinateGridRendererCreateInfo l_CoordinateGridRendererCreateInfo{};
        l_CoordinateGridRendererCreateInfo.RenderPass = m_RenderPass;
        m_CoordinateGridRenderer                      = New<CoordinateGridRenderer>( mGraphicContext, a_RenderContext, l_CoordinateGridRendererCreateInfo );
        m_VisualHelperRenderer                        = New<VisualHelperRenderer>( mGraphicContext, m_RenderPass );
    }

    MeshRendererCreateInfo SceneRenderer::GetRenderPipelineCreateInfo( RenderContext &aRenderContext, MaterialShaderComponent &a_PipelineSpecification )
    {
        MeshRendererCreateInfo l_CreateInfo;

        l_CreateInfo.Opaque         = ( a_PipelineSpecification.Type == MaterialType::Opaque );
        l_CreateInfo.IsTwoSided     = a_PipelineSpecification.IsTwoSided;
        l_CreateInfo.LineWidth      = a_PipelineSpecification.LineWidth;
        l_CreateInfo.VertexShader   = "Shaders\\PBRMeshShader.vert.spv";
        l_CreateInfo.FragmentShader = "Shaders\\PBRMeshShader.frag.spv";
        l_CreateInfo.RenderPass     = m_RenderPass;

        return l_CreateInfo;
    }

    MeshRenderer &SceneRenderer::GetRenderPipeline( RenderContext &aRenderContext, MeshRendererCreateInfo const &a_PipelineSpecification )
    {
        if( m_MeshRenderers.find( a_PipelineSpecification ) == m_MeshRenderers.end() )
            m_MeshRenderers[a_PipelineSpecification] = MeshRenderer( mGraphicContext, a_PipelineSpecification );

        return m_MeshRenderers[a_PipelineSpecification];
    }

    MeshRenderer &SceneRenderer::GetRenderPipeline( RenderContext &aRenderContext, MaterialShaderComponent &a_PipelineSpecification )
    {
        MeshRendererCreateInfo l_CreateInfo = GetRenderPipelineCreateInfo( aRenderContext, a_PipelineSpecification );

        return GetRenderPipeline( aRenderContext, l_CreateInfo );
    }

    ParticleSystemRenderer &SceneRenderer::GetRenderPipeline( RenderContext &aRenderContext, ParticleShaderComponent &a_PipelineSpecification )
    {
        ParticleRendererCreateInfo l_CreateInfo = GetRenderPipelineCreateInfo( aRenderContext, a_PipelineSpecification );

        if( m_ParticleRenderers.find( l_CreateInfo ) == m_ParticleRenderers.end() )
            m_ParticleRenderers[l_CreateInfo] = ParticleSystemRenderer( mGraphicContext, aRenderContext, l_CreateInfo );

        return m_ParticleRenderers[l_CreateInfo];
    }

    ParticleRendererCreateInfo SceneRenderer::GetRenderPipelineCreateInfo( RenderContext &aRenderContext, ParticleShaderComponent &a_PipelineSpecification )
    {
        ParticleRendererCreateInfo l_CreateInfo;
        l_CreateInfo.LineWidth      = a_PipelineSpecification.LineWidth;
        l_CreateInfo.VertexShader   = "Shaders\\ParticleSystem.vert.spv";
        l_CreateInfo.FragmentShader = "Shaders\\ParticleSystem.frag.spv";
        l_CreateInfo.RenderPass     = m_RenderPass;

        return l_CreateInfo;
    }

    void SceneRenderer::Render( RenderContext &aRenderContext )
    {
        LTSE_PROFILE_FUNCTION();

        UpdateDescriptorSets( aRenderContext );
        m_World->GetMaterialSystem()->UpdateDescriptors();

        int l_DirectionalLightCount = 0;
        int l_SpotlightCount        = 0;
        int l_PointLightCount       = 0;

        m_World->ForEach<DirectionalLightComponent>(
            [&]( auto a_Entity, auto &a_Component )
            {
                View.DirectionalLights[l_DirectionalLightCount] = DirectionalLightData( a_Component, math::mat4( 1.0f ) );
                l_DirectionalLightCount++;
            } );

        m_World->ForEach<PointLightComponent>(
            [&]( auto a_Entity, auto &a_Component )
            {
                math::mat4 l_TransformMatrix = math::mat4( 1.0f );
                if( a_Entity.Has<TransformMatrixComponent>() )
                    l_TransformMatrix = a_Entity.Get<TransformMatrixComponent>().Matrix;

                View.PointLights[l_PointLightCount] = PointLightData( a_Component, l_TransformMatrix );
                l_PointLightCount++;
            } );

        m_World->ForEach<SpotlightComponent>(
            [&]( auto a_Entity, auto &a_Component )
            {
                math::mat4 l_TransformMatrix = math::mat4( 1.0f );
                if( a_Entity.Has<TransformMatrixComponent>() )
                    l_TransformMatrix = a_Entity.Get<TransformMatrixComponent>().Matrix;

                View.Spotlights[l_SpotlightCount] = SpotlightData( a_Component, l_TransformMatrix );
                l_SpotlightCount++;
            } );

        View.PointLightCount       = l_PointLightCount;
        View.DirectionalLightCount = l_DirectionalLightCount;
        View.SpotlightCount        = l_SpotlightCount;

        if( m_World->Environment.Has<AmbientLightingComponent>() )
        {
            auto &l_Component = m_World->Environment.Get<AmbientLightingComponent>();

            Settings.AmbientLightIntensity = l_Component.Intensity;
            Settings.AmbientLightColor     = math::vec4( l_Component.Color, 0.0 );
        }

        m_CameraUniformBuffer->Write( View );
        m_ShaderParametersBuffer->Write( Settings );

        std::unordered_map<MeshRendererCreateInfo, std::vector<Entity>, MeshRendererCreateInfoHash> lOpaqueMeshQueue{};
        m_World->ForEach<StaticMeshComponent, MaterialShaderComponent>(
            [&]( auto a_Entity, auto &a_StaticMeshComponent, auto &a_MaterialData )
            {
                auto &l_PipelineCreateInfo = GetRenderPipelineCreateInfo( aRenderContext, a_MaterialData );
                if( lOpaqueMeshQueue.find( l_PipelineCreateInfo ) == lOpaqueMeshQueue.end() )
                    lOpaqueMeshQueue[l_PipelineCreateInfo] = std::vector<Entity>{};
                lOpaqueMeshQueue[l_PipelineCreateInfo].push_back( a_Entity );
            } );

        if( m_World->mVertexBuffer && m_World->mIndexBuffer )
        {
            aRenderContext.Bind( m_World->mTransformedVertexBuffer, m_World->mIndexBuffer );
            for( auto &lPipelineData : lOpaqueMeshQueue )
            {
                auto &a_Pipeline = GetRenderPipeline( aRenderContext, lPipelineData.first );
                if( a_Pipeline.Pipeline )
                    aRenderContext.Bind( a_Pipeline.Pipeline );
                else
                    continue;

                aRenderContext.Bind( m_SceneDescriptors, 0, -1 );
                aRenderContext.Bind( m_World->GetMaterialSystem()->GetDescriptorSet(), 1, -1 );

                for( auto &lMeshInformation : lPipelineData.second )
                {
                    if( lMeshInformation.Has<NodeDescriptorComponent>() )
                        aRenderContext.Bind( lMeshInformation.Get<NodeDescriptorComponent>().Descriptors, 2, -1 );

                    MeshRenderer::MaterialPushConstants l_MaterialPushConstants{};
                    l_MaterialPushConstants.mMaterialID = lMeshInformation.Get<MaterialComponent>().mMaterialID;

                    aRenderContext.PushConstants( { Graphics::Internal::eShaderStageTypeFlags::FRAGMENT }, 0, l_MaterialPushConstants );

                    auto &l_StaticMeshComponent = lMeshInformation.Get<StaticMeshComponent>();
                    aRenderContext.Draw( l_StaticMeshComponent.mIndexCount, l_StaticMeshComponent.mIndexOffset, l_StaticMeshComponent.mVertexOffset, 1, 0 );
                }
            }
        }

        // std::unordered_map<ParticleRendererCreateInfo, std::vector<ParticleSystemComponent>, ParticleSystemRendererCreateInfoHash> lParticleSystemQueue{};
        // m_World->ForEach<ParticleSystemComponent>(
        //     [&]( auto a_Entity, auto &a_ParticleSystemComponent )
        //     {
        //         if( !a_Entity.Has<RendererComponent>() )
        //             return;

        //         auto &l_RendererComponent = a_Entity.Get<RendererComponent>();
        //         if( !l_RendererComponent.Material )
        //             return;

        //         if( l_RendererComponent.Material.Has<ParticleShaderComponent>() )
        //         {
        //             auto &l_ParticleShaderComponent = l_RendererComponent.Material.Get<ParticleShaderComponent>();
        //             auto &a_Pipeline                = GetRenderPipeline( aRenderContext, l_ParticleShaderComponent );

        //             ParticleSystemRenderer::ParticleData l_ParticleData{};
        //             l_ParticleData.Model         = math::mat4( 1.0f );
        //             l_ParticleData.ParticleCount = a_ParticleSystemComponent.ParticleCount;
        //             l_ParticleData.ParticleSize  = a_ParticleSystemComponent.ParticleSize;
        //             l_ParticleData.Particles     = a_ParticleSystemComponent.Particles;
        //             a_Pipeline.Render( View.Projection, View.View, aRenderContext, l_ParticleData );
        //         }
        //     } );

        if( RenderGizmos )
        {
            m_VisualHelperRenderer->View       = View.View;
            m_VisualHelperRenderer->Projection = View.Projection;
            m_World->ForEach<DirectionalLightHelperComponent>(
                [&]( auto a_Entity, auto &a_DirectionalLightHelperComponent )
                {
                    math::mat4 l_Transform = math::mat4( 1.0f );
                    if( a_Entity.Has<TransformMatrixComponent>() )
                        l_Transform = a_Entity.Get<TransformMatrixComponent>().Matrix;
                    m_VisualHelperRenderer->Render( l_Transform, a_DirectionalLightHelperComponent, aRenderContext );
                } );

            m_World->ForEach<SpotlightHelperComponent>(
                [&]( auto a_Entity, auto &a_SpotlightHelperComponent )
                {
                    math::mat4 l_Transform = math::mat4( 1.0f );
                    if( a_Entity.Has<TransformMatrixComponent>() )
                        l_Transform = a_Entity.Get<TransformMatrixComponent>().Matrix;
                    m_VisualHelperRenderer->Render( l_Transform, a_SpotlightHelperComponent, aRenderContext );
                } );

            m_World->ForEach<PointLightHelperComponent>(
                [&]( auto a_Entity, auto &a_PointLightHelperComponent )
                {
                    math::mat4 l_Transform = math::mat4( 1.0f );
                    if( a_Entity.Has<TransformMatrixComponent>() )
                        l_Transform = a_Entity.Get<TransformMatrixComponent>().Matrix;
                    m_VisualHelperRenderer->Render( l_Transform, a_PointLightHelperComponent, aRenderContext );
                } );
        }

        if( RenderCoordinateGrid )
            m_CoordinateGridRenderer->Render( View.Projection, View.View, aRenderContext );
    }

    void SceneRenderer::UpdateDescriptorSets( RenderContext &aRenderContext )
    {
        LTSE_PROFILE_FUNCTION();

        m_World->ForEach<TransformMatrixComponent>(
            [&]( auto a_Entity, auto &a_Component )
            {
                if( !( a_Entity.Has<NodeDescriptorComponent>() ) )
                {
                    auto &l_NodeDescriptor         = a_Entity.Add<NodeDescriptorComponent>();
                    l_NodeDescriptor.Descriptors   = New<DescriptorSet>( mGraphicContext, MeshRenderer::GetNodeSetLayout( mGraphicContext ) );
                    l_NodeDescriptor.UniformBuffer = New<Buffer>( mGraphicContext, eBufferBindType::UNIFORM_BUFFER, true, false, true, true, sizeof( NodeMatrixDataComponent ) );

                    l_NodeDescriptor.Descriptors->Write( l_NodeDescriptor.UniformBuffer, false, 0, sizeof( NodeMatrixDataComponent ), 0 );
                }

                auto &l_NodeDescriptor = a_Entity.Get<NodeDescriptorComponent>();
                NodeMatrixDataComponent l_NodeTransform{};
                l_NodeTransform.Transform = a_Component.Matrix;
                a_Entity.IfExists<SkeletonComponent>(
                    [&]( auto &l_SkeletonComponent )
                    {
                        l_NodeTransform.JointCount = l_SkeletonComponent.BoneCount;
                        for( uint32_t i = 0; i < l_SkeletonComponent.BoneCount; i++ )
                            l_NodeTransform.Joints[i] = l_SkeletonComponent.JointMatrices[i];
                    } );

                l_NodeDescriptor.UniformBuffer->Write( l_NodeTransform );
            } );
    }

} // namespace LTSE::Core