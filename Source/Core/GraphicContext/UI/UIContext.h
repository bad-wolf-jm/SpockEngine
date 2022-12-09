#pragma once

#include "Core/GraphicContext//DescriptorSet.h"

#include "Core/GraphicContext//ARenderContext.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//GraphicsPipeline.h"

#include "UI/UI.h"

#include "Graphics/Vulkan/VkGpuBuffer.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_impl_glfw.h>
#include <imgui_internal.h>
#include <imguizmo.h>
#include <implot.h>

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
        UIContext( Ref<SE::Core::Window> aWindow, GraphicContext &aDevice, ARenderContext &aRenderContext,
                   std::string &aImGuiConfigPath, UIConfiguration const &aUIConfiguration );
        ~UIContext();

        void BeginFrame();
        void EndFrame( ARenderContext &aRenderContext );

        void PushFontFamily( FontFamily aFamily );
        void PopFont();

        ImGuiIO &GetIO();

        ImageHandle        CreateTextureHandle( Ref<Graphics::VkSampler2D> aTexture );
        Ref<DescriptorSet> AddTexture( Ref<Graphics::VkSampler2D> aTexture );

        ImFont *mMonoFont;
        ImFont *mMainFont;
        ImFont *mBoldFont;
        ImFont *mObliqueFont;
        ImFont *mBoldObliqueFont;

      private:
        ImGuiContext  *mImGUIOverlay;
        ImPlotContext *mImPlotContext;
        std::string    mImGuiConfigPath;

        UIStyle mUIStyle;

        GraphicContext mGraphicContext{};

        Ref<DescriptorSetLayout> mUIDescriptorSetLayout = nullptr;

        Ref<SE::Graphics::Internal::ShaderModule> mUIVertexShader   = nullptr;
        Ref<SE::Graphics::Internal::ShaderModule> mUIFragmentShader = nullptr;
        Ref<GraphicsPipeline>                     mUIRenderPipeline = nullptr;

        Ref<Graphics::VkSampler2D> mFontTexture       = nullptr;
        Ref<DescriptorSet>         mFontDescriptorSet = nullptr;

        Ref<VkGpuBuffer> mVertexBuffer;
        Ref<VkGpuBuffer> mIndexBuffer;

      private:
        void SetupRenderState( ARenderContext &aRenderContext, ImDrawData *aDrawData );
        void RenderDrawData( ARenderContext &aRenderContext, ImDrawData *aDrawData );
    };
} // namespace SE::Core
