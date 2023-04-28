#pragma once

#include <map>

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

#include "Graphics/API.h"

namespace SE::Core
{
    using namespace SE::Graphics;
    using namespace SE::Core::UI;

    class UIWindow
    {
      public:
        UIWindow( Ref<IGraphicContext> aGraphicContext, ImGuiViewport *aViewport );
        UIWindow( Ref<IGraphicContext> aGraphicContext, Ref<IRenderContext> aRenderContext );
        ~UIWindow();

        template <typename _Ty>
        Ref<_Ty> GraphicContext()
        {
            return std::reinterpret_pointer_cast<_Ty>( mGraphicContext );
        }

        void Render( Ref<IRenderContext> aRenderContext, ImDrawData *aDrawData );
        void Render( ImDrawData *aDrawData );
        void Render( Ref<IRenderContext> aRenderContext, ImDrawList const *aDrawData, int aVertexOffset, int aIndexOffset,
                     int aFbWidth, int aFbHeight, ImVec2 aPosition, ImVec2 aScale );
        void EndRender( ImDrawData *aDrawData );

      private:
        ImGuiViewport *mViewport = nullptr;

        Ref<IGraphicContext> mGraphicContext = nullptr;

        Ref<IWindow>           mWindow           = nullptr;
        Ref<SwapChain>         mSwapChain        = nullptr;
        Ref<IRenderContext>    mRenderContext    = nullptr;
        Ref<IGraphicsPipeline> mUIRenderPipeline = nullptr;

        Ref<IGraphicBuffer> mVertexBuffer = nullptr;
        Ref<IGraphicBuffer> mIndexBuffer  = nullptr;

      private:
        void SetupRenderState( Ref<IRenderContext> aRenderContext, ImDrawData *aDrawData );
        void CreatePipeline();
    };
} // namespace SE::Core
