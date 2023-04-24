#pragma once

#include <map>

#include "Graphics/Vulkan/DescriptorSet.h"

#include "Graphics/Vulkan/ARenderContext.h"
#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

#include "UI/UI.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"

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
        UIContext( Ref<SE::Core::IWindow> aWindow, Ref<VkGraphicContext> aDevice, ARenderContext &aRenderContext,
                   std::string &aImGuiConfigPath, UIConfiguration const &aUIConfiguration );
        ~UIContext();

        void BeginFrame();
        void EndFrame( ARenderContext &aRenderContext );

        void PushFontFamily( FontFamilyFlags aFamily );
        void PopFont();

        ImGuiIO &GetIO();

        ImageHandle        CreateTextureHandle( Ref<Graphics::VkSampler2D> aTexture );
        Ref<DescriptorSet> AddTexture( Ref<Graphics::VkSampler2D> aTexture );

        Ref<VkGraphicContext> GraphicContext() { return mGraphicContext; }

        // ImFont *mMonoFont;
        // ImFont *mMainFont;
        // ImFont *mBoldFont;
        // ImFont *mObliqueFont;
        // ImFont *mBoldObliqueFont;

      private:
        ImGuiContext  *mImGUIOverlay;
        ImPlotContext *mImPlotContext;
        std::string    mImGuiConfigPath;

        UIStyle mUIStyle;

        Ref<VkGraphicContext> mGraphicContext{};

        Ref<DescriptorSetLayout> mUIDescriptorSetLayout = nullptr;

        // Ref<ShaderModule>     mUIVertexShader   = nullptr;
        // Ref<ShaderModule>     mUIFragmentShader = nullptr;
        // Ref<GraphicsPipeline> mUIRenderPipeline = nullptr;

        Ref<Graphics::VkSampler2D> mFontTexture       = nullptr;
        Ref<DescriptorSet>         mFontDescriptorSet = nullptr;

        // Ref<VkGpuBuffer> mVertexBuffer;
        // Ref<VkGpuBuffer> mIndexBuffer;

        std::map<FontFamilyFlags, ImFont *> mFonts;

        ImFont *LoadFont( fs::path aFontName, fs::path aIconFontName, uint32_t aFontSize );

      private:
        Ref<UIWindow> mMainWindow;
        // void SetupRenderState( ARenderContext &aRenderContext, ImDrawData *aDrawData );
        // void RenderDrawData( ARenderContext &aRenderContext, ImDrawData *aDrawData );

        static void Renderer_CreateWindow( ImGuiViewport *vp );
        static void Renderer_DestroyWindow( ImGuiViewport *vp );
        static void Renderer_SetWindowSize( ImGuiViewport *vp, ImVec2 size );
        static void Renderer_RenderWindow( ImGuiViewport *vp, void *render_arg );
        static void Renderer_SwapBuffers( ImGuiViewport *vp, void *render_arg );
    };
} // namespace SE::Core
