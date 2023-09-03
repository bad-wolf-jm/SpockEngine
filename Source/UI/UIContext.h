#pragma once

#include <map>

#include "UI/UI.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imguizmo.h>
#include <implot.h>

#include "UIWindow.h"

#include "Core/Memory.h"
#include "Core/Types.h"

#include <stdexcept>

namespace SE::Core
{

    using namespace SE::Graphics;
    using namespace SE::Core::UI;

    struct UIConfiguration
    {
        uint32_t mFontSize;

        fs::path mMainFont;
        fs::path mBoldFont;
        fs::path mItalicFont;
        fs::path mBoldItalicFont;
        fs::path mMonoFont;
        fs::path mIconFont;
    };

    enum class FontFamilyFlags : int32_t
    {
        DISPLAY,
        H1,
        H2,
        H3,
        EM,
        HUGE,
        LARGE,
        NORMAL,
        SMALL,
        TINY,
        MONOSPACE
    };

    class UIContext
    {
      public:
        UIContext( ref_t<SE::Core::IWindow> aWindow, ref_t<IGraphicContext> aDevice, ref_t<IRenderContext> aRenderContext,
                   string_t &aImGuiConfigPath, UIConfiguration const &aUIConfiguration );
        ~UIContext();

        void BeginFrame();
        void EndFrame( ref_t<IRenderContext> aRenderContext );

        void PushFontFamily( FontFamilyFlags aFamily );
        void PopFont();

        ImGuiIO &GetIO();

        ImageHandle         CreateTextureHandle( ref_t<ISampler2D> aTexture );
        ref_t<IDescriptorSet> AddTexture( ref_t<ISampler2D> aTexture );

        ref_t<IGraphicContext> GraphicContext()
        {
            return mGraphicContext;
        }

        ImGuiContext  *mImGUIOverlay;
        ImPlotContext *mImPlotContext;
        string_t    mImGuiConfigPath;

        UIStyle mUIStyle;

        ref_t<IGraphicContext> mGraphicContext{};

        ref_t<IDescriptorSetLayout> mUIDescriptorSetLayout = nullptr;
        ref_t<IDescriptorSet>       mFontDescriptorSet     = nullptr;

        ref_t<ISampler2D> mFontTexture = nullptr;

        std::map<FontFamilyFlags, ImFont *> mFonts;

        ImFont *LoadFont( fs::path aFontName, fs::path aIconFontName, uint32_t aFontSize );

      private:
        ref_t<UIWindow> mMainWindow;

        static void Renderer_CreateWindow( ImGuiViewport *vp );
        static void Renderer_DestroyWindow( ImGuiViewport *vp );
        static void Renderer_SetWindowSize( ImGuiViewport *vp, ImVec2 size );
        static void Renderer_RenderWindow( ImGuiViewport *vp, void *render_arg );
        static void Renderer_SwapBuffers( ImGuiViewport *vp, void *render_arg );

      private:
        void RenderPlatformWindows();

        friend class UIWindow;
    };
} // namespace SE::Core
