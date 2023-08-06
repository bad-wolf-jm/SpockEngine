#pragma once

#include <map>
#include <stdexcept>

#include "UI/UI.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>

#include "Core/Memory.h"
#include "Core/Types.h"

#include "Graphics/API.h"

namespace SE::Core
{
    using namespace SE::Graphics;
    using namespace SE::Core::UI;

    class UIWindow
    {
      public:
        UIWindow( ref_t<IGraphicContext> aGraphicContext, ImGuiViewport *aViewport );
        UIWindow( ref_t<IGraphicContext> aGraphicContext, ref_t<IRenderContext> aRenderContext );
        ~UIWindow();

        template <typename _Ty>
        ref_t<_Ty> GraphicContext()
        {
            return std::reinterpret_pointer_cast<_Ty>( mGraphicContext );
        }

        void Render( ref_t<IRenderContext> aRenderContext, ImDrawData *aDrawData );
        void Render( ImDrawData *aDrawData );
        void Render( ref_t<IRenderContext> aRenderContext, ImDrawList const *aDrawData, int aVertexOffset, int aIndexOffset,
                     int aFbWidth, int aFbHeight, ImVec2 aPosition, ImVec2 aScale );
        void EndRender( ImDrawData *aDrawData );

      private:
        ImGuiViewport *mViewport = nullptr;

        ref_t<IGraphicContext> mGraphicContext = nullptr;

        ref_t<IWindow>           mWindow           = nullptr;
        ref_t<ISwapChain>        mSwapChain        = nullptr;
        ref_t<IRenderContext>    mRenderContext    = nullptr;
        ref_t<IGraphicsPipeline> mUIRenderPipeline = nullptr;

        ref_t<IGraphicBuffer> mVertexBuffer = nullptr;
        ref_t<IGraphicBuffer> mIndexBuffer  = nullptr;

      private:
        void SetupRenderState( ref_t<IRenderContext> aRenderContext, ImDrawData *aDrawData );
        void CreatePipeline();
    };
} // namespace SE::Core
