#include "UIContext.h"

#include "Engine/Engine.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/Logging.h"

#include "Graphics/Vulkan/VkPipeline.h"

#include "Core/Profiling/BlockTimer.h"
#include "Graphics/Interface/IWindow.h"

// std
#include <stdexcept>

#include "Core/Resource.h"
#include "UI/UI.h"

namespace SE::Core
{

    UIContext::UIContext( Ref<SE::Core::IWindow> aWindow, Ref<VkGraphicContext> aGraphicContext, ARenderContext &aRenderContext,
                          std::string &aImGuiConfigPath, UIConfiguration const &aUIConfiguration )
        : mGraphicContext{ aGraphicContext }
        , mImGuiConfigPath{ aImGuiConfigPath }
    {
        IMGUI_CHECKVERSION();
        mImGUIOverlay  = ImGui::CreateContext();
        mImPlotContext = ImPlot::CreateContext();

        ImGui::SetCurrentContext( mImGUIOverlay );
        ImGuiIO &io            = ImGui::GetIO();
        io.BackendRendererName = "imgui_impl_vulkan";
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset;
        if( !mImGuiConfigPath.empty() )
            io.IniFilename = mImGuiConfigPath.c_str();
        else
            io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_ViewportsEnable;
        io.BackendFlags |= ImGuiBackendFlags_RendererHasViewports;

        ImGui::StyleColorsDark();

        ImGui_ImplGlfw_InitForVulkan( aWindow->GetGLFWWindow(), true );
        mMainWindow = New<UIWindow>( aGraphicContext, aRenderContext );

        DescriptorBindingInfo lDescriptorBinding = {
            0, eDescriptorType::COMBINED_IMAGE_SAMPLER, { Graphics::eShaderStageTypeFlags::FRAGMENT } };
        DescriptorSetLayoutCreateInfo lBindingLayout = { { lDescriptorBinding } };
        mUIDescriptorSetLayout                       = New<DescriptorSetLayout>( mGraphicContext, lBindingLayout );

        // std::string lUIVertexShaderFiles = GetResourcePath( "Shaders\\ui_shader.vert.spv" ).string();
        // mUIVertexShader =
        //     New<Graphics::ShaderModule>( mGraphicContext, lUIVertexShaderFiles, Graphics::eShaderStageTypeFlags::VERTEX );

        // std::string lUIFragmentShaderFiles = GetResourcePath( "Shaders\\ui_shader.frag.spv" ).string();
        // mUIFragmentShader =
        //     New<Graphics::ShaderModule>( mGraphicContext, lUIFragmentShaderFiles, Graphics::eShaderStageTypeFlags::FRAGMENT );
        // GraphicsPipelineCreateInfo lUIPipelineCreateInfo = {};
        // lUIPipelineCreateInfo.mShaderStages              = { { mUIVertexShader, "main" }, { mUIFragmentShader, "main" } };
        // lUIPipelineCreateInfo.InputBufferLayout          = {
        //     { "Position", eBufferDataType::VEC2, 0, 0 },
        //     { "TextureCoords", eBufferDataType::VEC2, 0, 1 },
        //     { "Color", eBufferDataType::COLOR, 0, 2 },
        // };
        // lUIPipelineCreateInfo.Topology      = ePrimitiveTopology::TRIANGLES;
        // lUIPipelineCreateInfo.Culling       = eFaceCulling::NONE;
        // lUIPipelineCreateInfo.SampleCount   = aRenderContext.GetRenderTarget()->mSpec.mSampleCount;
        // lUIPipelineCreateInfo.LineWidth     = 1.0f;
        // lUIPipelineCreateInfo.RenderPass    = aRenderContext.GetRenderPass();
        // lUIPipelineCreateInfo.PushConstants = {
        //     { { Graphics::eShaderStageTypeFlags::VERTEX }, 0, sizeof( float ) * 4 },
        // };
        // lUIPipelineCreateInfo.SetLayouts = { mUIDescriptorSetLayout };

        // mUIRenderPipeline               = New<GraphicsPipeline>( mGraphicContext, lUIPipelineCreateInfo );
        ImGuiViewport *lMainViewport    = ImGui::GetMainViewport();
        lMainViewport->RendererUserData = nullptr;

        static ImWchar lCharRanges[] = { 0x0020, 0x00FF, 0x2070, 0x208e, 0x2100, 0x21FF, 0x2200,
                                         0x2300, 0x2A00, 0x2AFF, 0x370,  0x3ff,  0 };
        static ImWchar lRanges[]     = { ICON_MIN_FA, ICON_MAX_FA, 0 };
        ImFontConfig   lFontConfig;
        lFontConfig.MergeMode        = true;
        lFontConfig.PixelSnapH       = true;
        lFontConfig.GlyphMinAdvanceX = 10.0f;

        mFonts[FontFamilyFlags::DISPLAY] =
            LoadFont( aUIConfiguration.mMainFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize * 2 );
        mFonts[FontFamilyFlags::H1] =
            LoadFont( aUIConfiguration.mBoldFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize * 1.5f );
        mFonts[FontFamilyFlags::H2] =
            LoadFont( aUIConfiguration.mBoldFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize * 1.25f );
        mFonts[FontFamilyFlags::H3] =
            LoadFont( aUIConfiguration.mItalicFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize * 1.25f );
        mFonts[FontFamilyFlags::EM] =
            LoadFont( aUIConfiguration.mItalicFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize * 1.0f );
        mFonts[FontFamilyFlags::HUGE] =
            LoadFont( aUIConfiguration.mMainFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize * 1.5f );
        mFonts[FontFamilyFlags::LARGE] =
            LoadFont( aUIConfiguration.mMainFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize * 1.25f );
        mFonts[FontFamilyFlags::NORMAL] =
            LoadFont( aUIConfiguration.mMainFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize );
        mFonts[FontFamilyFlags::SMALL] =
            LoadFont( aUIConfiguration.mMainFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize * .85f );
        mFonts[FontFamilyFlags::TINY] =
            LoadFont( aUIConfiguration.mMainFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize * .75f );
        mFonts[FontFamilyFlags::MONOSPACE] =
            LoadFont( aUIConfiguration.mMonoFont, aUIConfiguration.mIconFont, aUIConfiguration.mFontSize );

        mUIStyle = UIStyle{ true };

        unsigned char *lFontPixelData;
        int            lWidth, lHeight;
        io.Fonts->GetTexDataAsRGBA32( &lFontPixelData, &lWidth, &lHeight );
        size_t lUploadSize = lWidth * lHeight * 4 * sizeof( char );

        sTextureCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mMipLevels = 1;

        sImageData lImageData{};
        lImageData.mByteSize  = lUploadSize;
        lImageData.mFormat    = eColorFormat::RGBA8_UNORM;
        lImageData.mWidth     = static_cast<uint32_t>( lWidth );
        lImageData.mHeight    = static_cast<uint32_t>( lHeight );
        lImageData.mPixelData = std::vector<uint8_t>( lFontPixelData, lFontPixelData + lUploadSize );
        TextureData2D lTextureImage( lTextureCreateInfo, lImageData );

        sTextureSamplingInfo lSamplingInfo{};
        lSamplingInfo.mFilter   = eSamplerFilter::LINEAR;
        lSamplingInfo.mWrapping = eSamplerWrapping::REPEAT;
        TextureSampler2D lTextureSampler( lTextureImage, lSamplingInfo );

        auto lFontTexture  = New<VkTexture2D>( mGraphicContext, lTextureImage );
        mFontTexture       = New<VkSampler2D>( mGraphicContext, lFontTexture, lSamplingInfo );
        mFontDescriptorSet = AddTexture( mFontTexture );
        io.Fonts->TexID    = (ImTextureID)mFontDescriptorSet->GetVkDescriptorSet();

        // mVertexBuffer = New<VkGpuBuffer>( mGraphicContext, eBufferType::VERTEX_BUFFER, true, true, true, true, 1 );
        // mIndexBuffer  = New<VkGpuBuffer>( mGraphicContext, eBufferType::INDEX_BUFFER, true, true, true, true, 1 );

        auto &lPlatformIO = ImGui::GetPlatformIO();

        lPlatformIO.Platform_CreateWindow       = ImGui_ImplGlfw_CreateWindow;
        lPlatformIO.Platform_DestroyWindow      = ImGui_ImplGlfw_DestroyWindow;
        lPlatformIO.Platform_ShowWindow         = ImGui_ImplGlfw_ShowWindow;
        lPlatformIO.Platform_SetWindowPos       = ImGui_ImplGlfw_SetWindowPos;
        lPlatformIO.Platform_GetWindowPos       = ImGui_ImplGlfw_GetWindowPos;
        lPlatformIO.Platform_SetWindowSize      = ImGui_ImplGlfw_SetWindowSize;
        lPlatformIO.Platform_GetWindowSize      = ImGui_ImplGlfw_GetWindowSize;
        lPlatformIO.Platform_SetWindowFocus     = ImGui_ImplGlfw_SetWindowFocus;
        lPlatformIO.Platform_GetWindowFocus     = ImGui_ImplGlfw_GetWindowFocus;
        lPlatformIO.Platform_GetWindowMinimized = ImGui_ImplGlfw_GetWindowMinimized;
        lPlatformIO.Platform_SetWindowTitle     = ImGui_ImplGlfw_SetWindowTitle;
        lPlatformIO.Platform_RenderWindow       = ImGui_ImplGlfw_RenderWindow;
        lPlatformIO.Platform_SwapBuffers        = ImGui_ImplGlfw_SwapBuffers;
#if GLFW_HAS_WINDOW_ALPHA
        lPlatformIO.Platform_SetWindowAlpha = ImGui_ImplGlfw_SetWindowAlpha;
#endif

        lPlatformIO.Renderer_CreateWindow  = Renderer_CreateWindow;
        lPlatformIO.Renderer_DestroyWindow = Renderer_DestroyWindow;
        lPlatformIO.Renderer_RenderWindow  = Renderer_RenderWindow;
        lPlatformIO.Renderer_SetWindowSize = Renderer_SetWindowSize;
        lPlatformIO.Renderer_SwapBuffers   = Renderer_SwapBuffers;
    }

    void UIContext::Renderer_CreateWindow( ImGuiViewport *vp ) 
    { 
        UIWindow* lNewRenderWindow = new UIWindow( Engine::GetInstance()->UIContext()->GraphicContext(), vp ); 
        
        vp->RendererUserData = lNewRenderWindow;
    }

    void UIContext::Renderer_DestroyWindow( ImGuiViewport *vp )
    {
        delete(UIWindow *)vp->RendererUserData;

        vp->RendererUserData = nullptr;
    }

    void UIContext::Renderer_SetWindowSize( ImGuiViewport *vp, ImVec2 size )
    {
        //
    }

    void UIContext::Renderer_RenderWindow( ImGuiViewport *vp, void *render_arg )
    {
        ( (UIWindow *)( vp->RendererUserData ) )->Render( vp->DrawData );
    }

    void UIContext::Renderer_SwapBuffers( ImGuiViewport *vp, void *render_arg )
    {
        ( (UIWindow *)( vp->RendererUserData ) )->EndRender( vp->DrawData );
    }

    // void UIContext::SetupRenderState( ARenderContext &aRenderContext, ImDrawData *aDrawData )
    // {
    //     SE_PROFILE_FUNCTION();

    //     aRenderContext.Bind( mUIRenderPipeline );

    //     if( aDrawData->TotalVtxCount > 0 ) aRenderContext.Bind( mVertexBuffer, mIndexBuffer );

    //     int lFramebufferWidth  = (int)( aDrawData->DisplaySize.x * aDrawData->FramebufferScale.x );
    //     int lFramebufferHeight = (int)( aDrawData->DisplaySize.y * aDrawData->FramebufferScale.y );

    //     aRenderContext.GetCurrentCommandBuffer()->SetViewport( { 0, 0 }, { lFramebufferWidth, lFramebufferHeight } );

    //     // Setup scale and translation:
    //     // Our visible imgui space lies from aDrawData->DisplayPps (top left) to aDrawData->DisplayPos+data_data->DisplaySize
    //     (bottom
    //     // right). DisplayPos is (0,0) for single viewport apps.
    //     {
    //         float lScale[2];
    //         lScale[0] = 2.0f / aDrawData->DisplaySize.x;
    //         lScale[1] = 2.0f / aDrawData->DisplaySize.y;
    //         float translate[2];
    //         translate[0] = -1.0f - aDrawData->DisplayPos.x * lScale[0];
    //         translate[1] = -1.0f - aDrawData->DisplayPos.y * lScale[1];
    //         aRenderContext.PushConstants( { Graphics::eShaderStageTypeFlags::VERTEX }, 0, lScale );
    //         aRenderContext.PushConstants( { Graphics::eShaderStageTypeFlags::VERTEX }, sizeof( float ) * 2, translate );
    //     }
    // }

    // // Render function
    // void UIContext::RenderDrawData( ARenderContext &aRenderContext, ImDrawData *aDrawData )
    // {
    //     SE_PROFILE_FUNCTION();

    //     // Avoid rendering when minimized, scale coordinates for retina displays (screen coordinates != framebuffer coordinates)
    //     int lFramebufferWidth  = (int)( aDrawData->DisplaySize.x * aDrawData->FramebufferScale.x );
    //     int lFramebufferHeight = (int)( aDrawData->DisplaySize.y * aDrawData->FramebufferScale.y );

    //     if( ( lFramebufferWidth <= 0 ) || ( lFramebufferHeight <= 0 ) ) return;

    //     if( aDrawData->TotalVtxCount > 0 )
    //     {
    //         // Create or resize the vertex/index buffers
    //         size_t vertex_size = aDrawData->TotalVtxCount * sizeof( ImDrawVert );
    //         size_t index_size  = aDrawData->TotalIdxCount * sizeof( ImDrawIdx );

    //         mVertexBuffer->Resize( vertex_size );
    //         mIndexBuffer->Resize( index_size );

    //         int lVertexOffset = 0;
    //         int lIndexOffset  = 0;
    //         for( int n = 0; n < aDrawData->CmdListsCount; n++ )
    //         {
    //             const ImDrawList *lImGuiDrawCommands = aDrawData->CmdLists[n];
    //             mVertexBuffer->Upload( lImGuiDrawCommands->VtxBuffer.Data, lImGuiDrawCommands->VtxBuffer.Size, lVertexOffset );
    //             mIndexBuffer->Upload( lImGuiDrawCommands->IdxBuffer.Data, lImGuiDrawCommands->IdxBuffer.Size, lIndexOffset );
    //             lVertexOffset += lImGuiDrawCommands->VtxBuffer.Size;
    //             lIndexOffset += lImGuiDrawCommands->IdxBuffer.Size;
    //         }
    //     }

    //     // Setup desired Vulkan state
    //     SetupRenderState( aRenderContext, aDrawData );

    //     // Will project scissor/clipping rectangles into framebuffer space
    //     ImVec2 lClipOffset = aDrawData->DisplayPos;       // (0,0) unless using multi-viewports
    //     ImVec2 lClipScale  = aDrawData->FramebufferScale; // (1,1) unless using retina display which are often (2,2)

    //     int lGlobalVtxOffset = 0;
    //     int lGlobalIdxOffset = 0;
    //     for( int n = 0; n < aDrawData->CmdListsCount; n++ )
    //     {
    //         const ImDrawList *lImGuiDrawCommands = aDrawData->CmdLists[n];
    //         for( int cmd_i = 0; cmd_i < lImGuiDrawCommands->CmdBuffer.Size; cmd_i++ )
    //         {
    //             const ImDrawCmd *lPcmd = &lImGuiDrawCommands->CmdBuffer[cmd_i];

    //             // Project scissor/clipping rectangles into framebuffer space
    //             ImVec4 lClipRect;
    //             lClipRect.x = ( lPcmd->ClipRect.x - lClipOffset.x ) * lClipScale.x;
    //             lClipRect.y = ( lPcmd->ClipRect.y - lClipOffset.y ) * lClipScale.y;
    //             lClipRect.z = ( lPcmd->ClipRect.z - lClipOffset.x ) * lClipScale.x;
    //             lClipRect.w = ( lPcmd->ClipRect.w - lClipOffset.y ) * lClipScale.y;

    //             if( lClipRect.x < lFramebufferWidth && lClipRect.y < lFramebufferHeight && lClipRect.z >= 0.0f && lClipRect.w >=
    //             0.0f )
    //             {
    //                 // Negative offsets are illegal for vkCmdSetScissor
    //                 if( lClipRect.x < 0.0f ) lClipRect.x = 0.0f;
    //                 if( lClipRect.y < 0.0f ) lClipRect.y = 0.0f;

    //                 aRenderContext.GetCurrentCommandBuffer()->SetScissor(
    //                     { (int32_t)( lClipRect.x ), (int32_t)( lClipRect.y ) },
    //                     { (uint32_t)( lClipRect.z - lClipRect.x ), (uint32_t)( lClipRect.w - lClipRect.y ) } );

    //                 // Bind a the descriptor set for the current texture.
    //                 if( (VkDescriptorSet)lPcmd->TextureId )
    //                 {
    //                     VkDescriptorSet desc_set[1] = { (VkDescriptorSet)lPcmd->TextureId };
    //                     vkCmdBindDescriptorSets( aRenderContext.GetCurrentCommandBuffer()->mVkObject,
    //                     VK_PIPELINE_BIND_POINT_GRAPHICS,
    //                                              mUIRenderPipeline->GetVkPipelineLayoutObject()->mVkObject, 0, 1, desc_set, 0, NULL
    //                                              );

    //                     aRenderContext.Draw( lPcmd->ElemCount, lPcmd->IdxOffset + lGlobalIdxOffset,
    //                                          lPcmd->VtxOffset + lGlobalVtxOffset, 1, 0 );
    //                 }
    //             }
    //         }
    //         lGlobalIdxOffset += lImGuiDrawCommands->IdxBuffer.Size;
    //         lGlobalVtxOffset += lImGuiDrawCommands->VtxBuffer.Size;
    //     }
    // }

    ImFont *UIContext::LoadFont( fs::path aFontName, fs::path aIconFontName, uint32_t aFontSize )
    {
        static ImWchar      lCharRanges[] = { 0x0020, 0x00FF, 0x2070, 0x208e, 0x2100, 0x21FF, 0x2200,
                                              0x2300, 0x2A00, 0x2AFF, 0x370,  0x3ff,  0 };
        static ImWchar      lRanges[]     = { ICON_MIN_FA, ICON_MAX_FA, 0 };
        static ImFontConfig lFontConfig;
        lFontConfig.MergeMode        = true;
        lFontConfig.PixelSnapH       = true;
        lFontConfig.GlyphMinAdvanceX = 10.0f;

        ImGuiIO &io = ImGui::GetIO();

        ImFont *lFont = io.Fonts->AddFontFromFileTTF( aFontName.string().c_str(), aFontSize, nullptr, lCharRanges );
        io.Fonts->AddFontFromFileTTF( aIconFontName.string().c_str(), aFontSize, &lFontConfig, lRanges );

        return lFont;
    }

    UIContext::~UIContext()
    {
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext( mImPlotContext );
        ImGui::DestroyContext( mImGUIOverlay );
    }

    void UIContext::BeginFrame()
    {
        ImGui::SetCurrentContext( mImGUIOverlay );
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGuizmo::BeginFrame();
        ImGuizmo::AllowAxisFlip( false );
        PushFontFamily( FontFamilyFlags::NORMAL );
    }

    void UIContext::EndFrame( ARenderContext &aRenderContext )
    {
        PopFont();
        ImGui::Render();
        ImGui::UpdatePlatformWindows();
        ImDrawData *drawdata = ImGui::GetDrawData();
        mMainWindow->Render( aRenderContext, drawdata );
        ImGui::RenderPlatformWindowsDefault();
    }

    ImageHandle UIContext::CreateTextureHandle( Ref<VkSampler2D> aTexture ) { return ImageHandle{ AddTexture( aTexture ) }; }

    ImGuiIO &UIContext::GetIO()
    {
        ImGui::SetCurrentContext( mImGUIOverlay );
        return ImGui::GetIO();
    }

    void UIContext::PushFontFamily( FontFamilyFlags aFamily ) { ImGui::PushFont( mFonts[aFamily] ); }

    void UIContext::PopFont() { ImGui::PopFont(); }

    Ref<DescriptorSet> UIContext::AddTexture( Ref<VkSampler2D> aTexture )
    {
        Ref<DescriptorSet> lDescriptorSet = New<DescriptorSet>( mGraphicContext, mUIDescriptorSetLayout );
        lDescriptorSet->Write( aTexture, 0 );
        return lDescriptorSet;
    }

    // Ref<DescriptorSet> UIContext::AddTexture( Ref<VkSampler2D> aTexture, Ref<VkGraphicContext> aGraphicContext,  )
    // {
    //     Ref<DescriptorSet> lDescriptorSet = New<DescriptorSet>( mGraphicContext, mUIDescriptorSetLayout );
    //     lDescriptorSet->Write( aTexture, 0 );
    //     return lDescriptorSet;
    // }

} // namespace SE::Core
