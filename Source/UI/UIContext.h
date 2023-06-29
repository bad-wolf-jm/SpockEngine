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

        path_t mMainFont;
        path_t mBoldFont;
        path_t mItalicFont;
        path_t mBoldItalicFont;
        path_t mMonoFont;
        path_t mIconFont;
    };

    enum class FontFamilyFlags : int32_t
    {
        DISPLAY,
        H1,
        H2,
        H3,
        HUGE,
        LARGE,
        EM,
        BOLD,
        NORMAL,
        SMALL,
        TINY,
        MONOSPACE
    };

    class UIContext
    {
      public:
        UIContext( Ref<SE::Core::IWindow> aWindow, Ref<IGraphicContext> aDevice, Ref<IRenderContext> aRenderContext,
                   string_t &aImGuiConfigPath, UIConfiguration const &aUIConfiguration );
        ~UIContext();

        void BeginFrame();
        void EndFrame( Ref<IRenderContext> aRenderContext );

        void PushFontFamily( FontFamilyFlags aFamily );
        void PopFont();

        ImGuiIO &GetIO();

        ImageHandle         CreateTextureHandle( Ref<ISampler2D> aTexture );
        Ref<IDescriptorSet> AddTexture( Ref<ISampler2D> aTexture );

        Ref<IGraphicContext> GraphicContext() { return mGraphicContext; }

        ImGuiContext  *mImGUIOverlay;
        ImPlotContext *mImPlotContext;
        string_t    mImGuiConfigPath;

        UIStyle mUIStyle;

        Ref<IGraphicContext> mGraphicContext{};

        Ref<IDescriptorSetLayout> mUIDescriptorSetLayout = nullptr;
        Ref<IDescriptorSet>       mFontDescriptorSet     = nullptr;

        Ref<ISampler2D> mFontTexture = nullptr;

        std::map<FontFamilyFlags, ImFont *> mFonts;

        ImFont *LoadFont( path_t aFontName, path_t aIconFontName, uint32_t aFontSize );

      private:
        Ref<UIWindow> mMainWindow;

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
