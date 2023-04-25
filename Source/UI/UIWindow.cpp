#include "UIWindow.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/Logging.h"

#include "Graphics/Vulkan/VkPipeline.h"

#include "Core/Profiling/BlockTimer.h"
#include "Graphics/Interface/IWindow.h"

#include <stdexcept>

#include "Core/Resource.h"
#include "UI/UI.h"
#include "UIContext.h"

namespace SE::Core
{
    UIWindow::UIWindow( Ref<VkGraphicContext> aGraphicContext, ImGuiViewport *aViewport )
        : mViewport{ aViewport }
        , mGraphicContext{ aGraphicContext }
    {
        mWindow        = SE::Core::New<IWindow>( (GLFWwindow *)aViewport->PlatformHandle );
        mSwapChain     = SE::Core::New<SwapChain>( mGraphicContext, mWindow );
        mRenderContext = SE::Graphics::ARenderContext( mGraphicContext, mSwapChain );

        DescriptorBindingInfo lDescriptorBinding = {
            0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { Graphics::eShaderStageTypeFlags::FRAGMENT } };
        DescriptorSetLayoutCreateInfo lBindingLayout = { { lDescriptorBinding } };
        mUIDescriptorSetLayout                       = New<DescriptorSetLayout>( mGraphicContext, lBindingLayout );

        CreatePipeline();

        mVertexBuffer = New<VkGpuBuffer>( mGraphicContext, eBufferType::VERTEX_BUFFER, true, true, true, true, 1 );
        mIndexBuffer  = New<VkGpuBuffer>( mGraphicContext, eBufferType::INDEX_BUFFER, true, true, true, true, 1 );
    }

    UIWindow::UIWindow( Ref<VkGraphicContext> aGraphicContext, ARenderContext &aRenderContext )
        : mWindow{ nullptr }
        , mGraphicContext{ aGraphicContext }
        , mRenderContext{ aRenderContext }
        , mSwapChain{ nullptr }
        , mViewport{ nullptr }
    {

        DescriptorBindingInfo lDescriptorBinding = {
            0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { Graphics::eShaderStageTypeFlags::FRAGMENT } };
        DescriptorSetLayoutCreateInfo lBindingLayout = { { lDescriptorBinding } };
        mUIDescriptorSetLayout                       = New<DescriptorSetLayout>( mGraphicContext, lBindingLayout );

        CreatePipeline();

        mVertexBuffer = New<VkGpuBuffer>( mGraphicContext, eBufferType::VERTEX_BUFFER, true, true, true, true, 1 );
        mIndexBuffer  = New<VkGpuBuffer>( mGraphicContext, eBufferType::INDEX_BUFFER, true, true, true, true, 1 );
    }

    void UIWindow::CreatePipeline()
    {
        std::string lUIVertexShaderFiles = GetResourcePath( "Shaders\\ui_shader.vert.spv" ).string();
        mUIVertexShader =
            New<Graphics::ShaderModule>( mGraphicContext, lUIVertexShaderFiles, Graphics::eShaderStageTypeFlags::VERTEX );

        std::string lUIFragmentShaderFiles = GetResourcePath( "Shaders\\ui_shader.frag.spv" ).string();
        mUIFragmentShader =
            New<Graphics::ShaderModule>( mGraphicContext, lUIFragmentShaderFiles, Graphics::eShaderStageTypeFlags::FRAGMENT );
        GraphicsPipelineCreateInfo lUIPipelineCreateInfo = {};
        lUIPipelineCreateInfo.mShaderStages              = { { mUIVertexShader, "main" }, { mUIFragmentShader, "main" } };
        lUIPipelineCreateInfo.InputBufferLayout          = {
            { "Position", eBufferDataType::VEC2, 0, 0 },
            { "TextureCoords", eBufferDataType::VEC2, 0, 1 },
            { "Color", eBufferDataType::COLOR, 0, 2 },
        };
        lUIPipelineCreateInfo.Topology      = ePrimitiveTopology::TRIANGLES;
        lUIPipelineCreateInfo.Culling       = eFaceCulling::NONE;
        lUIPipelineCreateInfo.SampleCount   = mRenderContext.GetRenderTarget()->mSpec.mSampleCount;
        lUIPipelineCreateInfo.LineWidth     = 1.0f;
        lUIPipelineCreateInfo.RenderPass    = mRenderContext.GetRenderPass();
        lUIPipelineCreateInfo.PushConstants = {
            { { Graphics::eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 },
        };
        lUIPipelineCreateInfo.SetLayouts = { mUIDescriptorSetLayout };

        mUIRenderPipeline = New<GraphicsPipeline>( mGraphicContext, lUIPipelineCreateInfo );
    }

    void UIWindow::SetupRenderState( ARenderContext &aRenderContext, ImDrawData *aDrawData )
    {
        SE_PROFILE_FUNCTION();

        aRenderContext.Bind( mUIRenderPipeline );

        if( aDrawData->TotalVtxCount > 0 ) aRenderContext.Bind( mVertexBuffer, mIndexBuffer );

        int lFramebufferWidth  = (int)( aDrawData->DisplaySize.x * aDrawData->FramebufferScale.x );
        int lFramebufferHeight = (int)( aDrawData->DisplaySize.y * aDrawData->FramebufferScale.y );

        aRenderContext.GetCurrentCommandBuffer()->SetViewport( { 0, 0 }, { lFramebufferWidth, lFramebufferHeight } );

        float lS[2];
        lS[0] = 2.0f / aDrawData->DisplaySize.x;
        lS[1] = 2.0f / aDrawData->DisplaySize.y;
        float lT[2];
        lT[0] = -1.0f - aDrawData->DisplayPos.x * lS[0];
        lT[1] = -1.0f - aDrawData->DisplayPos.y * lS[1];
        aRenderContext.PushConstants( { Graphics::eShaderStageTypeFlags::VERTEX }, 0, lS );
        aRenderContext.PushConstants( { Graphics::eShaderStageTypeFlags::VERTEX }, sizeof( float ) * 2, lT );
    }

    // Render function
    void UIWindow::Render( ARenderContext &aRenderContext, ImDrawData *aDrawData )
    {
        SE_PROFILE_FUNCTION();

        // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
        int lFramebufferWidth  = (int)( aDrawData->DisplaySize.x * aDrawData->FramebufferScale.x );
        int lFramebufferHeight = (int)( aDrawData->DisplaySize.y * aDrawData->FramebufferScale.y );

        if( ( lFramebufferWidth <= 0 ) || ( lFramebufferHeight <= 0 ) ) return;

        if( aDrawData->TotalVtxCount > 0 )
        {
            // Create or resize the vertex/index buffers
            size_t vertex_size = aDrawData->TotalVtxCount * sizeof( ImDrawVert );
            size_t index_size  = aDrawData->TotalIdxCount * sizeof( ImDrawIdx );

            mVertexBuffer->Resize( vertex_size );
            mIndexBuffer->Resize( index_size );

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

        // Setup desired Vulkan state
        SetupRenderState( aRenderContext, aDrawData );

        // Will project scissor/clipping rectangles into framebuffer space
        // clang-format off
        ImVec4 lOffset = ImVec4{ aDrawData->DisplayPos.x, aDrawData->DisplayPos.y, aDrawData->DisplayPos.x, aDrawData->DisplayPos.y };
        ImVec4 lScale  = ImVec4{ aDrawData->FramebufferScale.x, aDrawData->FramebufferScale.y, aDrawData->FramebufferScale.x, aDrawData->FramebufferScale.y };
        // clang-format on

        int lGlobalVtxOffset = 0;
        int lGlobalIdxOffset = 0;
        for( int n = 0; n < aDrawData->CmdListsCount; n++ )
        {
            const ImDrawList *lImGuiDrawCommands = aDrawData->CmdLists[n];
            
            for( int i = 0; i < lImGuiDrawCommands->CmdBuffer.Size; i++ )
            {
                const ImDrawCmd *lPcmd = &lImGuiDrawCommands->CmdBuffer[i];

                // Project scissor/clipping rectangles into framebuffer space
                ImVec4 lClipRect = ( lPcmd->ClipRect - lOffset ) * lScale;

                if( lClipRect.x < lFramebufferWidth && lClipRect.y < lFramebufferHeight && lClipRect.z >= 0.0f && lClipRect.w >= 0.0f )
                {
                    // Negative offsets are illegal for vkCmdSetScissor
                    if( lClipRect.x < 0.0f ) lClipRect.x = 0.0f;
                    if( lClipRect.y < 0.0f ) lClipRect.y = 0.0f;

                    aRenderContext.GetCurrentCommandBuffer()->SetScissor(
                        { (int32_t)( lClipRect.x ), (int32_t)( lClipRect.y ) },
                        { (uint32_t)( lClipRect.z - lClipRect.x ), (uint32_t)( lClipRect.w - lClipRect.y ) } );

                    // Bind a the descriptor set for the current texture.
                    if( (VkDescriptorSet)lPcmd->TextureId )
                    {
                        VkDescriptorSet lDesc[1] = { (VkDescriptorSet)lPcmd->TextureId };
                        vkCmdBindDescriptorSets( aRenderContext.GetCurrentCommandBuffer()->mVkObject, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                                 mUIRenderPipeline->GetVkPipelineLayoutObject()->mVkObject, 0, 1, lDesc, 0, NULL );

                        aRenderContext.Draw( lPcmd->ElemCount, lPcmd->IdxOffset + lGlobalIdxOffset,
                                             lPcmd->VtxOffset + lGlobalVtxOffset, 1, 0 );
                    }
                }
            }

            lGlobalIdxOffset += lImGuiDrawCommands->IdxBuffer.Size;
            lGlobalVtxOffset += lImGuiDrawCommands->VtxBuffer.Size;
        }
    }

    void UIWindow::Render( ImDrawData *aDrawData )
    {
        mRenderContext.BeginRender();
        Render( mRenderContext, aDrawData );
    }

    void UIWindow::EndRender( ImDrawData *aDrawData )
    {
        mRenderContext.EndRender();
        mRenderContext.Present();
    }

    UIWindow::~UIWindow() {}

} // namespace SE::Core
