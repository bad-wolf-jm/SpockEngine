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

    UIContext::UIContext( Ref<SE::Core::IWindow> aWindow, Ref<IGraphicContext> aGraphicContext, ARenderContext &aRenderContext,
                          std::string &aImGuiConfigPath, UIConfiguration const &aConfig )
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
        mUIDescriptorSetLayout                       = New<DescriptorSetLayout>( Cast<VkGraphicContext>(mGraphicContext), lBindingLayout );

        ImGuiViewport *lMainViewport    = ImGui::GetMainViewport();
        lMainViewport->RendererUserData = nullptr;

        static ImWchar lCharRanges[] = { 0x0020, 0x00FF, 0x2070, 0x208e, 0x2100, 0x21FF, 0x2200,
                                         0x2300, 0x2A00, 0x2AFF, 0x370,  0x3ff,  0 };
        static ImWchar lRanges[]     = { ICON_MIN_FA, ICON_MAX_FA, 0 };
        ImFontConfig   lFontConfig;
        lFontConfig.MergeMode        = true;
        lFontConfig.PixelSnapH       = true;
        lFontConfig.GlyphMinAdvanceX = 10.0f;

        mFonts[FontFamilyFlags::DISPLAY]   = LoadFont( aConfig.mMainFont, aConfig.mIconFont, aConfig.mFontSize * 2 );
        mFonts[FontFamilyFlags::H1]        = LoadFont( aConfig.mBoldFont, aConfig.mIconFont, aConfig.mFontSize * 1.5f );
        mFonts[FontFamilyFlags::H2]        = LoadFont( aConfig.mBoldFont, aConfig.mIconFont, aConfig.mFontSize * 1.25f );
        mFonts[FontFamilyFlags::H3]        = LoadFont( aConfig.mItalicFont, aConfig.mIconFont, aConfig.mFontSize * 1.25f );
        mFonts[FontFamilyFlags::EM]        = LoadFont( aConfig.mItalicFont, aConfig.mIconFont, aConfig.mFontSize * 1.0f );
        mFonts[FontFamilyFlags::HUGE]      = LoadFont( aConfig.mMainFont, aConfig.mIconFont, aConfig.mFontSize * 1.5f );
        mFonts[FontFamilyFlags::LARGE]     = LoadFont( aConfig.mMainFont, aConfig.mIconFont, aConfig.mFontSize * 1.25f );
        mFonts[FontFamilyFlags::NORMAL]    = LoadFont( aConfig.mMainFont, aConfig.mIconFont, aConfig.mFontSize );
        mFonts[FontFamilyFlags::SMALL]     = LoadFont( aConfig.mMainFont, aConfig.mIconFont, aConfig.mFontSize * .75f );
        mFonts[FontFamilyFlags::TINY]      = LoadFont( aConfig.mMainFont, aConfig.mIconFont, aConfig.mFontSize * .5f );
        mFonts[FontFamilyFlags::MONOSPACE] = LoadFont( aConfig.mMonoFont, aConfig.mIconFont, aConfig.mFontSize );

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

        auto lFontTexture  = New<VkTexture2D>( Cast<VkGraphicContext>(mGraphicContext), lTextureImage );
        mFontTexture       = New<VkSampler2D>( Cast<VkGraphicContext>(mGraphicContext), lFontTexture, lSamplingInfo );
        mFontDescriptorSet = AddTexture( mFontTexture );
        io.Fonts->TexID    = (ImTextureID)mFontDescriptorSet->GetVkDescriptorSet();

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
        UIWindow *lNewRenderWindow = new UIWindow( Engine::GetInstance()->UIContext()->GraphicContext(), vp );

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

    void UIContext::RenderPlatformWindows()
    {
        ImGuiPlatformIO &lPlatformIO = ImGui::GetPlatformIO();

        for( int i = 1; i < lPlatformIO.Viewports.Size; i++ )
        {
            ImGuiViewport *viewport = lPlatformIO.Viewports[i];
            if( viewport->Flags & ImGuiViewportFlags_Minimized ) continue;
            if( lPlatformIO.Platform_RenderWindow ) lPlatformIO.Platform_RenderWindow( viewport, this );
            if( lPlatformIO.Renderer_RenderWindow ) lPlatformIO.Renderer_RenderWindow( viewport, this );
        }

        for( int i = 1; i < lPlatformIO.Viewports.Size; i++ )
        {
            ImGuiViewport *viewport = lPlatformIO.Viewports[i];
            if( viewport->Flags & ImGuiViewportFlags_Minimized ) continue;
            if( lPlatformIO.Platform_SwapBuffers ) lPlatformIO.Platform_SwapBuffers( viewport, this );
            if( lPlatformIO.Renderer_SwapBuffers ) lPlatformIO.Renderer_SwapBuffers( viewport, this );
        }
    }

    void UIContext::EndFrame( ARenderContext &aRenderContext )
    {
        PopFont();
        ImGui::Render();
        ImGui::UpdatePlatformWindows();
        ImDrawData *drawdata = ImGui::GetDrawData();
        mMainWindow->Render( aRenderContext, drawdata );
        RenderPlatformWindows();
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
        Ref<DescriptorSet> lDescriptorSet = New<DescriptorSet>( Cast<VkGraphicContext>(mGraphicContext), mUIDescriptorSetLayout );
        lDescriptorSet->Write( aTexture, 0 );
        return lDescriptorSet;
    }
} // namespace SE::Core
