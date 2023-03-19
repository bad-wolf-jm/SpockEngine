#pragma once

#include "Core/Math/Types.h"

#include "Core/CUDA/Texture/TextureTypes.h"

#include "Graphics/Vulkan/ARenderContext.h"
#include "Graphics/Vulkan/VkGpuBuffer.h"
#include "Graphics/Vulkan/VkGraphicContext.h"
#include "Graphics/Vulkan/VkSampler2D.h"
#include "Graphics/Vulkan/VkTexture2D.h"
#include "Graphics/Vulkan/VkRenderTarget.h"

namespace SE::Core
{
    using namespace SE::Graphics;

    class Figure
    {
      public:
        Figure()  = default;
        ~Figure() = default;

        Figure( Ref<VkGraphicContext> aGraphicContext );

        Figure( Figure const &Figure ) = default;

        Ref<VkTexture2D> GetOutputImage();

        void Render();
        void ResizeOutput( uint32_t aOutputWidth, uint32_t aOutputHeight );

      protected:
        Ref<VkGraphicContext> mGraphicContext{};

        uint32_t mViewWidth  = 0;
        uint32_t mViewHeight = 0;

        eColorFormat mOutputFormat = eColorFormat::RGBA8_UNORM;

        math::mat4 mProjectionMatrix{};
        math::mat4 mViewMatrix{};

        ARenderContext      mRenderContext{};
        Ref<VkRenderTarget> mRenderTarget = nullptr;
    };
} // namespace SE::Core