#include "UIWindow.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/Logging.h"

#include "Graphics/Vulkan/VkPipeline.h"
#include "Graphics/Vulkan/VkRenderContext.h"
#include "Graphics/Vulkan/vkSwapChainRenderContext.h"

#include "Core/Profiling/BlockTimer.h"
#include "Graphics/Interface/IWindow.h"

#include <stdexcept>

#include "Core/Resource.h"
#include "UI/UI.h"
#include "UIContext.h"

namespace SE::Core
{
    UIWindow::UIWindow( Ref<IGraphicContext> aGraphicContext, ImGuiViewport *aViewport )
        : mViewport{ aViewport }
        , mGraphicContext{ aGraphicContext }
    {
        mWindow        = SE::Core::New<IWindow>( (GLFWwindow *)aViewport->PlatformHandle );
        mSwapChain     = CreateSwapChain( mGraphicContext, mWindow );
        mRenderContext = CreateRenderContext( mGraphicContext, mSwapChain );

        CreatePipeline();

        mVertexBuffer = CreateBuffer( mGraphicContext, eBufferType::VERTEX_BUFFER, true, true, true, true, 1 );
        mIndexBuffer  = CreateBuffer( mGraphicContext, eBufferType::INDEX_BUFFER, true, true, true, true, 1 );
    }

    UIWindow::UIWindow( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext )
        : mWindow{ nullptr }
        , mGraphicContext{ aGraphicContext }
        , mRenderContext{ aRenderContext }
        , mSwapChain{ nullptr }
        , mViewport{ nullptr }
    {
        CreatePipeline();

        mVertexBuffer = CreateBuffer( mGraphicContext, eBufferType::VERTEX_BUFFER, true, true, true, true, 1 );
        mIndexBuffer  = CreateBuffer( mGraphicContext, eBufferType::INDEX_BUFFER, true, true, true, true, 1 );
    }

    void UIWindow::CreatePipeline()
    {
        mUIRenderPipeline = CreateGraphicsPipeline( mGraphicContext, mRenderContext, ePrimitiveTopology::TRIANGLES );

        mUIRenderPipeline->SetCulling( eFaceCulling::NONE );

        mUIRenderPipeline->SetShader( eShaderStageTypeFlags::VERTEX, GetResourcePath( "Shaders\\ui_shader.vert.spv" ), "main" );
        mUIRenderPipeline->SetShader( eShaderStageTypeFlags::FRAGMENT, GetResourcePath( "Shaders\\ui_shader.frag.spv" ), "main" );

        mUIRenderPipeline->AddInput( "Position", eBufferDataType::VEC2, 0, 0 );
        mUIRenderPipeline->AddInput( "TextureCoords", eBufferDataType::VEC2, 0, 1 );
        mUIRenderPipeline->AddInput( "Color", eBufferDataType::COLOR, 0, 2 );

        auto lDescriptorSet = CreateDescriptorSetLayout( mGraphicContext );
        lDescriptorSet->AddBinding( 0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { eShaderStageTypeFlags::FRAGMENT } );
        lDescriptorSet->Build();
        mUIRenderPipeline->AddDescriptorSet( lDescriptorSet );

        mUIRenderPipeline->AddPushConstantRange( { eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 );

        mUIRenderPipeline->Build();
    }

    void UIWindow::SetupRenderState( Ref<IRenderContext> aRenderContext, ImDrawData *aDrawData )
    {
        SE_PROFILE_FUNCTION();

        int lFramebufferWidth  = (int)( aDrawData->DisplaySize.x * aDrawData->FramebufferScale.x );
        int lFramebufferHeight = (int)( aDrawData->DisplaySize.y * aDrawData->FramebufferScale.y );

        float lS[2];
        lS[0] = 2.0f / aDrawData->DisplaySize.x;
        lS[1] = 2.0f / aDrawData->DisplaySize.y;
        float lT[2];
        lT[0] = -1.0f - aDrawData->DisplayPos.x * lS[0];
        lT[1] = -1.0f - aDrawData->DisplayPos.y * lS[1];

        aRenderContext->Bind( mUIRenderPipeline );
        aRenderContext->Bind( Cast<VkGpuBuffer>( mVertexBuffer ), Cast<VkGpuBuffer>( mIndexBuffer ) );
        aRenderContext->SetViewport( { 0, 0 }, { lFramebufferWidth, lFramebufferHeight } );
        aRenderContext->PushConstants( { eShaderStageTypeFlags::VERTEX }, 0, lS );
        aRenderContext->PushConstants( { eShaderStageTypeFlags::VERTEX }, sizeof( float ) * 2, lT );
    }

    void UIWindow::Render( Ref<IRenderContext> aRenderContext, ImDrawData *aDrawData )
    {
        SE_PROFILE_FUNCTION();

        int lFramebufferWidth  = (int)( aDrawData->DisplaySize.x * aDrawData->FramebufferScale.x );
        int lFramebufferHeight = (int)( aDrawData->DisplaySize.y * aDrawData->FramebufferScale.y );
        if( ( lFramebufferWidth <= 0 ) || ( lFramebufferHeight <= 0 ) ) return;

        if( aDrawData->TotalVtxCount > 0 )
        {
            size_t lVertexSize = aDrawData->TotalVtxCount * sizeof( ImDrawVert );
            size_t lIndexSize  = aDrawData->TotalIdxCount * sizeof( ImDrawIdx );

            mVertexBuffer->Resize( lVertexSize );
            mIndexBuffer->Resize( lIndexSize );

            int lVertexOffset = 0;
            int lIndexOffset  = 0;
            for( int n = 0; n < aDrawData->CmdListsCount; n++ )
            {
                const ImDrawList *lImGuiDrawCommands = aDrawData->CmdLists[n];
                mVertexBuffer->Upload( lImGuiDrawCommands->VtxBuffer.Data, lImGuiDrawCommands->VtxBuffer.Size, lVertexOffset );
                mIndexBuffer->Upload( lImGuiDrawCommands->IdxBuffer.Data, lImGuiDrawCommands->IdxBuffer.Size, lIndexOffset );

                lVertexOffset += lImGuiDrawCommands->VtxBuffer.Size;
                lIndexOffset += lImGuiDrawCommands->IdxBuffer.Size;
            }
        }

        SetupRenderState( aRenderContext, aDrawData );

        int lGlobalVtxOffset = 0;
        int lGlobalIdxOffset = 0;
        for( int n = 0; n < aDrawData->CmdListsCount; n++ )
        {
            const ImDrawList *lImGuiDrawCommands = aDrawData->CmdLists[n];

            Render( aRenderContext, lImGuiDrawCommands, lGlobalVtxOffset, lGlobalIdxOffset, lFramebufferWidth, lFramebufferHeight,
                    aDrawData->DisplayPos, aDrawData->FramebufferScale );

            lGlobalIdxOffset += lImGuiDrawCommands->IdxBuffer.Size;
            lGlobalVtxOffset += lImGuiDrawCommands->VtxBuffer.Size;
        }
    }

    void UIWindow::Render( Ref<IRenderContext> aRenderContext, ImDrawList const *aDrawList, int aVertexOffset, int aIndexOffset,
                           int aFbWidth, int aFbHeight, ImVec2 aPosition, ImVec2 aScale )
    {
        ImVec4 lOffset = ImVec4{ aPosition.x, aPosition.y, aPosition.x, aPosition.y };
        ImVec4 lScale  = ImVec4{ aScale.x, aScale.y, aScale.x, aScale.y };

        for( int i = 0; i < aDrawList->CmdBuffer.Size; i++ )
        {
            const ImDrawCmd *lPcmd = &aDrawList->CmdBuffer[i];

            if( !( (VkDescriptorSet)lPcmd->TextureId ) ) continue;

            ImVec4 lClipRect = ( lPcmd->ClipRect - lOffset ) * lScale;
            if( !( ( lClipRect.x < aFbWidth ) && ( lClipRect.y < aFbHeight ) && ( lClipRect.z >= 0.0f ) && ( lClipRect.w >= 0.0f ) ) )
                continue;

            if( lClipRect.x < 0.0f ) lClipRect.x = 0.0f;
            if( lClipRect.y < 0.0f ) lClipRect.y = 0.0f;

            aRenderContext->SetScissor( { (int32_t)( lClipRect.x ), (int32_t)( lClipRect.y ) },
                                        { (uint32_t)( lClipRect.z - lClipRect.x ), (uint32_t)( lClipRect.w - lClipRect.y ) } );
            aRenderContext->Bind( (void *)lPcmd->TextureId, 0, -1 );
            aRenderContext->Draw( lPcmd->ElemCount, lPcmd->IdxOffset + aIndexOffset, lPcmd->VtxOffset + aVertexOffset, 1, 0 );
        }
    }

    void UIWindow::Render( ImDrawData *aDrawData )
    {
        mRenderContext->BeginRender();
        Render( mRenderContext, aDrawData );
    }

    void UIWindow::EndRender( ImDrawData *aDrawData )
    {
        mRenderContext->EndRender();
        mRenderContext->Present();
    }

    UIWindow::~UIWindow() {}

} // namespace SE::Core
