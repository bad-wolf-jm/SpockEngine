#pragma once

#include "Developer/GraphicContext/DescriptorSet.h"

#include "Developer/Core/Vulkan/VkPipeline.h"
#include "Developer/GraphicContext/DescriptorSet.h"
#include "Developer/GraphicContext/GraphicContext.h"
#include "Developer/GraphicContext/GraphicsPipeline.h"
#include "Developer/GraphicContext/RenderContext.h"

#include "Developer/UI/UI.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imguizmo.h>
#include <implot.h>

#include "Core/Memory.h"
#include "Core/Types.h"

#include <stdexcept>

namespace LTSE::Core
{

    using namespace LTSE::Graphics;
    using namespace LTSE::Core::UI;

    enum class FontFamilyFlags : uint8_t
    {
        NORMAL = 0,
        BOLD   = ( 1 << 0 ),
        ITALIC = ( 1 << 1 ),
        MONO   = ( 1 << 2 )
    };
    using FontFamily = EnumSet<FontFamilyFlags, 0xff>;

    class UIContext
    {
      public:
        UIContext( Ref<LTSE::Core::ViewportClient> aWindow, GraphicContext &aDevice, RenderContext &aRenderContext, std::string &aImGuiConfigPath );
        ~UIContext();
        void BeginFrame();
        void EndFrame( RenderContext &aRenderContext );

        void PushFontFamily( FontFamily aFamily );
        void PopFont();

        ImGuiIO &GetIO();

        ImageHandle CreateTextureHandle( Ref<Texture2D> aTexture );
        Ref<DescriptorSet> AddTexture( Ref<Texture2D> aTexture );

        ImFont *mMonoFont;
        ImFont *mMainFont;
        ImFont *mBoldFont;
        ImFont *mObliqueFont;
        ImFont *mBoldObliqueFont;

      private:
        ImGuiContext *mImGUIOverlay;
        ImPlotContext *mImPlotContext;
        std::string mImGuiConfigPath;

        UIStyle mUIStyle;

        GraphicContext mGraphicContext{};

        Ref<DescriptorSetLayout> mUIDescriptorSetLayout = nullptr;

        Ref<LTSE::Graphics::Internal::ShaderModule> mUIVertexShader   = nullptr;
        Ref<LTSE::Graphics::Internal::ShaderModule> mUIFragmentShader = nullptr;
        Ref<GraphicsPipeline> mUIRenderPipeline                       = nullptr;

        Ref<Texture2D> mFontTexture           = nullptr;
        Ref<DescriptorSet> mFontDescriptorSet = nullptr;

        Ref<Buffer> mVertexBuffer;
        Ref<Buffer> mIndexBuffer;

      private:
        void SetupRenderState( RenderContext &aRenderContext, ImDrawData *aDrawData );
        void RenderDrawData( RenderContext &aRenderContext, ImDrawData *aDrawData );
    };
} // namespace LTSE::Core
