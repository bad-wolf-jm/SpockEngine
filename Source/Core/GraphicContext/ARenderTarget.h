#pragma once

#include "Graphics/Interface/IWindow.h"
#include "Core/Memory.h"

#include "Core/CUDA/Texture/ColorFormat.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Graphics/Vulkan/VkCoreMacros.h"

#include "Graphics/Vulkan/VkRenderTarget.h"
#include "Graphics/Vulkan/VkTexture2D.h"

namespace SE::Graphics
{
    using namespace Internal;
    using namespace SE::Core;

    struct sRenderTargetDescription
    {
        uint32_t mSampleCount = 1;
        uint32_t mWidth       = 0;
        uint32_t mHeight      = 0;
    };

    enum class eAttachmentLoadOp : uint8_t
    {
        UNSPECIFIED = 0,
        CLEAR       = 1,
        LOAD        = 2
    };

    enum class eAttachmentStoreOp : uint8_t
    {
        UNSPECIFIED = 0,
        STORE       = 1
    };

    enum class eAttachmentType
    {
        COLOR        = 0,
        DEPTH        = 1,
        MSAA_RESOLVE = 2
    };

    enum class eAttachmentLayout
    {
        SHADER_READ_ONLY = 0,
        COLOR_ATTACHMENT = 1,
        DEPTH_STENCIL    = 2
    };

    struct sAttachmentDescription
    {
        eAttachmentType    mType        = eAttachmentType::COLOR;
        eColorFormat       mFormat      = eColorFormat::RGBA8_UNORM;
        math::vec4         mClearColor  = { 0.0f, 0.0f, 0.0f, 0.0f };
        bool               mIsSampled   = false;
        bool               mIsPresented = false;
        bool               mIsDefined   = false;
        eAttachmentLoadOp  mLoadOp      = eAttachmentLoadOp::UNSPECIFIED;
        eAttachmentStoreOp mStoreOp     = eAttachmentStoreOp::UNSPECIFIED;

        sAttachmentDescription()  = default;
        ~sAttachmentDescription() = default;

        sAttachmentDescription( sAttachmentDescription const & ) = default;
    };

    class ARenderTarget
    {
      public:
        sRenderTargetDescription mSpec;

        ARenderTarget() = default;
        ARenderTarget( GraphicContext &aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription );

        ~ARenderTarget() = default;

        uint32_t GetImageCount() { return mImageCount; }

        void AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo );
        void AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                            Ref<VkTexture2D> aFramebufferImage );

        ARenderTarget &AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
                                      math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp,
                                      eAttachmentStoreOp eStoreOp );

        ARenderTarget &AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
                                      math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp,
                                      eAttachmentStoreOp eStoreOp, Ref<VkTexture2D> aFramebufferImage );

        void Finalize();

        Ref<sVkAbstractRenderPassObject> GetRenderPass() { return mRenderPassObject; }

        virtual bool BeginRender();
        virtual void EndRender();
        virtual void Present();

        virtual std::vector<VkClearValue> GetClearValues();
        virtual Ref<VkRenderTarget>       GetFramebuffer();

        virtual VkSemaphore GetImageAvailableSemaphore( uint32_t i );
        virtual VkSemaphore GetRenderFinishedSemaphore( uint32_t i );
        virtual VkFence     GetInFlightFence( uint32_t i );

        virtual uint32_t GetCurrentImage();

        virtual Ref<sVkCommandBufferObject> GetCurrentCommandBuffer() { return mCommandBufferObject[0]; }

        Ref<VkTexture2D> &GetAttachment( std::string const &aKey );

      protected:
        Ref<sVkAbstractRenderPassObject> CreateDefaultRenderPass();

        void InitializeCommandBuffers();

        uint32_t mImageCount = 0;

        GraphicContext mGraphicContext{};

        std::vector<VkClearValue>           mClearValues    = {};
        std::vector<sAttachmentDescription> mAttachmentInfo = {};
        std::vector<std::string>            mAttachmentIDs  = {};

        std::unordered_map<std::string, Ref<VkTexture2D>> mAttachments = {};

        Ref<sVkAbstractRenderPassObject> mRenderPassObject  = nullptr;
        Ref<VkRenderTarget>              mFramebufferObject = nullptr;

        std::vector<Ref<sVkCommandBufferObject>> mCommandBufferObject = {};
    };
} // namespace SE::Graphics