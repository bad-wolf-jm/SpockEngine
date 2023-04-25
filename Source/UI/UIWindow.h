#pragma once

#include <map>

#include "Graphics/Vulkan/DescriptorSet.h"

#include "Graphics/Vulkan/ARenderContext.h"
#include "Graphics/Vulkan/DescriptorSet.h"
#include "Graphics/Vulkan/GraphicsPipeline.h"
#include "Graphics/Vulkan/SwapChain.h"
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

    class UIWindow
    {
      public:
        UIWindow( Ref<VkGraphicContext> aGraphicContext, ImGuiViewport *aViewport );
        UIWindow( Ref<VkGraphicContext> aGraphicContext, ARenderContext &aRenderContext );
        ~UIWindow();

        Ref<VkGraphicContext> GraphicContext() { return mGraphicContext; }

        void Render( ARenderContext &aRenderContext, ImDrawData *aDrawData );
        void Render( ImDrawData *aDrawData );
        void Render( ARenderContext &aRenderContext, ImDrawList const *aDrawData, int aVertexOffset, int aIndexOffset, int aFbWidth,
                     int aFbHeight, ImVec2 aPosition, ImVec2 aScale );
        void EndRender( ImDrawData *aDrawData );

      private:
        Ref<IWindow>          mWindow         = nullptr;
        Ref<VkGraphicContext> mGraphicContext = nullptr;
        Ref<SwapChain>        mSwapChain      = nullptr;

        SE::Graphics::ARenderContext mRenderContext;

        ImGuiViewport           *mViewport              = nullptr;
        Ref<DescriptorSetLayout> mUIDescriptorSetLayout = nullptr;

        Ref<ShaderModule>          mUIVertexShader    = nullptr;
        Ref<ShaderModule>          mUIFragmentShader  = nullptr;
        Ref<GraphicsPipeline>      mUIRenderPipeline  = nullptr;
        Ref<Graphics::VkSampler2D> mFontTexture       = nullptr;
        Ref<DescriptorSet>         mFontDescriptorSet = nullptr;
        Ref<VkGpuBuffer>           mVertexBuffer      = nullptr;
        Ref<VkGpuBuffer>           mIndexBuffer       = nullptr;

      private:
        void SetupRenderState( ARenderContext &aRenderContext, ImDrawData *aDrawData );
        void CreatePipeline();
    };
} // namespace SE::Core
