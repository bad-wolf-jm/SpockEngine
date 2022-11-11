#pragma once

#include "Core/Memory.h"
#include "Core/Platform/ViewportClient.h"

#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/Textures/ColorFormat.h"
#include "Core/Vulkan/VkCoreMacros.h"

namespace LTSE::Graphics
{
    using namespace Internal;
    using namespace LTSE::Core;

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

    struct sAttachmentDescription
    {
        eAttachmentType    mType        = eAttachmentType::COLOR;
        eColorFormat       mFormat      = eColorFormat::RGBA8_UNORM;
        math::vec4         mClearColor  = { 0.0f, 0.0f, 0.0f, 0.0f };
        bool               mIsSampled   = false;
        bool               mIsPresented = false;
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
        ARenderTarget( GraphicContext &aGraphicContext );

        ~ARenderTarget() = default;

        uint32_t GetImageCount() { return mImageCount; }
        void     Initialize( sRenderTargetDescription &aSpec );
        void     InitializeCommandBuffers();

        void AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo );

        ARenderTarget &AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat,
            math::vec4 aClearColor, bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp, eAttachmentStoreOp eStoreOp );

        void                             Finalize();
        Ref<sVkAbstractRenderPassObject> GetRenderPass() { return mRenderPassObject; }

        virtual bool BeginRender();
        virtual void EndRender();
        virtual void Present();

        virtual Ref<sVkFramebufferObject> GetFramebuffer();

        virtual Ref<sVkCommandBufferObject> GetCommandBuffer( uint32_t i );
        virtual VkSemaphore                 GetImageAvailableSemaphore( uint32_t i );
        virtual VkSemaphore                 GetRenderFinishedSemaphore( uint32_t i );
        virtual VkFence                     GetInFlightFence( uint32_t i );

        virtual uint32_t GetCurrentImage();

      protected:
        Ref<sVkAbstractRenderPassObject> CreateDefaultRenderPass();

        uint32_t mImageCount = 0;

        GraphicContext mGraphicContext{};

        std::vector<sAttachmentDescription> mAttachmentInfo = {};
        std::vector<std::string>            mAttachmentIDs  = {};

        std::unordered_map<std::string, Ref<sVkFramebufferImage>> mAttachments = {};

        Ref<sVkAbstractRenderPassObject> mRenderPassObject  = nullptr;
        Ref<sVkFramebufferObject>        mFramebufferObject = nullptr;
    };
} // namespace LTSE::Graphics