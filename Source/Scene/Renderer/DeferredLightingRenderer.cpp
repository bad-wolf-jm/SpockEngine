#include "DeferredLightingRenderer.h"

#include <chrono>

#include "Scene/Primitives/Primitives.h"
#include "Scene/VertexData.h"

#include "Core/Logging.h"
#include "Core/Resource.h"

#include "Shaders/Embedded/gToneMap.h"
#include "Shaders/Embedded/gPBRFunctions.h"
#include "Shaders/Embedded/gDeferredLightingFragmentShaderPreamble.h"
#include "Shaders/Embedded/gDeferredLightingFragmentShaderCalculation.h"
#include "Shaders/Embedded/gDeferredLightingVertexShader.h"

namespace SE::Core
{
    using namespace math;
    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetCameraSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );
        lNewLayout->AddBinding( 0, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 1, eDescriptorType::UNIFORM_BUFFER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetTextureSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext );

        DescriptorSetLayoutCreateInfo lTextureBindLayout{};
        lNewLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 1, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 2, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->AddBinding( 3, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetDirectionalShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext, true );
        lNewLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetSpotlightShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext, true );
        lNewLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        lNewLayout->Build();

        return lNewLayout;
    }

    Ref<IDescriptorSetLayout> DeferredLightingRenderer::GetPointLightShadowSetLayout( Ref<IGraphicContext> aGraphicContext )
    {
        auto lNewLayout = CreateDescriptorSetLayout( aGraphicContext, true );
        lNewLayout->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );

        lNewLayout->Build();

        return lNewLayout;
    }

    DeferredLightingRenderer::DeferredLightingRenderer( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext )
        : mGraphicContext( aGraphicContext )
    {
        mPipeline = CreateGraphicsPipeline( mGraphicContext, aRenderContext, ePrimitiveTopology::TRIANGLES );

        mPipeline->SetCulling( eFaceCulling::NONE );
        mPipeline->SetDepthParameters( false, false, eDepthCompareOperation::ALWAYS );

        fs::path lShaderPath = "E:\\Work\\Git\\SpockEngine\\Resources\\Shaders\\Cache";
        auto     lVertexShader =
            CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::VERTEX, 450, "defered_lighting_vertex_shader", lShaderPath );
        lVertexShader->AddCode( SE::Private::Shaders::gDeferredLightingVertexShader_data );
        lVertexShader->Compile();
        mPipeline->SetShader( eShaderStageTypeFlags::VERTEX, lVertexShader, "main" );

        auto lFragmentShader = CreateShaderProgram( mGraphicContext, eShaderStageTypeFlags::FRAGMENT, 450,
                                                    "defered_lighting_fragment_shader", lShaderPath );
        lFragmentShader->AddCode( SE::Private::Shaders::gDeferredLightingFragmentShaderPreamble_data );
        lFragmentShader->AddCode( SE::Private::Shaders::gToneMap_data );
        lFragmentShader->AddCode( SE::Private::Shaders::gPBRFunctions_data );
        lFragmentShader->AddCode( SE::Private::Shaders::gDeferredLightingFragmentShaderCalculation_data );
        lFragmentShader->Compile();
        mPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, lFragmentShader, "main" );

        CameraSetLayout            = GetCameraSetLayout( mGraphicContext );
        TextureSetLayout           = GetTextureSetLayout( mGraphicContext );
        DirectionalShadowSetLayout = GetDirectionalShadowSetLayout( mGraphicContext );
        SpotlightShadowSetLayout   = GetSpotlightShadowSetLayout( mGraphicContext );
        PointLightShadowSetLayout  = GetPointLightShadowSetLayout( mGraphicContext );

        mPipeline->AddDescriptorSet( CameraSetLayout );
        mPipeline->AddDescriptorSet( TextureSetLayout );
        mPipeline->AddDescriptorSet( DirectionalShadowSetLayout );
        mPipeline->AddDescriptorSet( SpotlightShadowSetLayout );
        mPipeline->AddDescriptorSet( PointLightShadowSetLayout );

        mPipeline->Build();
    }

} // namespace SE::Core