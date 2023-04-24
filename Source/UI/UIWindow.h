#pragma once

#include <map>

#include "Graphics/Vulkan/DescriptorSet.h"

#include "Graphics/Vulkan/ARenderContext.h"
#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/SwapChain.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"
#include "Graphics/Vulkan/VkGraphicContext.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"
#include "UI/UI.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>

#include "Core/Memory.h"
#include "Core/Types.h"

#include <stdexcept>

namespace SE::Core
{
    using namespace SE::Graphics;
    using namespace SE::Core::UI;

    // struct UIConfiguration
    // {
    //     uint32_t mFontSize;

    //     fs::path mMainFont;
    //     fs::path mBoldFont;
    //     fs::path mItalicFont;
    //     fs::path mBoldItalicFont;
    //     fs::path mMonoFont;
    //     fs::path mIconFont;
    // };

    // enum class FontFamilyFlags : int32_t
    // {
    //     DISPLAY,
    //     H1,
    //     H2,
    //     H3,
    //     EM,
    //     HUGE,
    //     LARGE,
    //     NORMAL,
    //     SMALL,
    //     TINY,
    //     MONOSPACE
    // };

    class UIWindow
    {
      public:
        UIWindow( ImGuiViewport *aViewport );
        UIWindow( Ref<VkGraphicContext> aGraphicContext, ARenderContext &aRenderContext );
        ~UIWindow();

        // void BeginFrame();
        // void EndFrame( ARenderContext &aRenderContext );
        // void PushFontFamily( FontFamilyFlags aFamily );
        // void PopFont();
        // ImGuiIO &GetIO();
        // ImageHandle           CreateTextureHandle( Ref<Graphics::VkSampler2D> aTexture );
        // Ref<DescriptorSet>    AddTexture( Ref<Graphics::VkSampler2D> aTexture );

        Ref<VkGraphicContext> GraphicContext() { return mGraphicContext; }

        // ImFont *mMonoFont;
        // ImFont *mMainFont;
        // ImFont *mBoldFont;
        // ImFont *mObliqueFont;
        // ImFont *mBoldObliqueFont;
        void Render( ARenderContext &aRenderContext, ImDrawData *aDrawData );

      private:
        Ref<IWindow>                 mWindow                 = nullptr;
        Ref<VkGraphicContext>        mGraphicContext         = nullptr;
        Ref<SwapChain>               mSwapChain              = nullptr;

        SE::Graphics::ARenderContext mRenderContext;

        // ImGuiContext  *mImGUIOverlay;
        // ImPlotContext *mImPlotContext;
        // std::string    mImGuiConfigPath;
        // UIStyle mUIStyle;
        // UIContext                 *mUIContext             = nullptr;
        ImGuiViewport             *mViewport              = nullptr;
        Ref<DescriptorSetLayout>   mUIDescriptorSetLayout = nullptr;

        Ref<ShaderModule>          mUIVertexShader        = nullptr;
        Ref<ShaderModule>          mUIFragmentShader      = nullptr;
        Ref<GraphicsPipeline>      mUIRenderPipeline      = nullptr;
        Ref<Graphics::VkSampler2D> mFontTexture           = nullptr;
        Ref<DescriptorSet>         mFontDescriptorSet     = nullptr;
        Ref<VkGpuBuffer>           mVertexBuffer          = nullptr;
        Ref<VkGpuBuffer>           mIndexBuffer           = nullptr;

        // std::map<FontFamilyFlags, ImFont *> mFonts;
        // ImFont *LoadFont( fs::path aFontName, fs::path aIconFontName, uint32_t aFontSize );

      private:
        void SetupRenderState( ARenderContext &aRenderContext, ImDrawData *aDrawData );
    };
} // namespace SE::Core
