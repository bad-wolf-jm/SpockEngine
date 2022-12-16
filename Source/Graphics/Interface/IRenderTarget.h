#pragma once

#include "Core/Memory.h"

#include "IGraphicContext.h"
#include "ITexture2D.h"

namespace SE::Graphics
{
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

    class IRenderTarget
    {
      public:
        sRenderTargetDescription mSpec;

        IRenderTarget() = default;
        IRenderTarget( Ref<IGraphicContext> aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription );

        ~IRenderTarget() = default;

        uint32_t GetImageCount() { return mImageCount; }

        void AddAttachment( std::string const &aAttachmentID, sAttachmentDescription const &aCreateInfo,
                            Ref<ITexture2D> aFramebufferImage );

        void AddAttachment( std::string const &aAttachmentID, eAttachmentType aType, eColorFormat aFormat, math::vec4 aClearColor,
                            bool aIsSampled, bool aIsPresented, eAttachmentLoadOp aLoadOp, eAttachmentStoreOp eStoreOp,
                            Ref<ITexture2D> aFramebufferImage );

        virtual void Finalize();
        virtual bool BeginRender();
        virtual void EndRender();
        virtual void Present();

        Ref<ITexture2D> GetAttachment( std::string const &aKey );

      protected:
        Ref<IGraphicContext> mGraphicContext{};

        uint32_t mImageCount = 0;

        std::vector<sAttachmentDescription> mAttachmentInfo = {};
        std::vector<std::string>            mAttachmentIDs  = {};

        std::unordered_map<std::string, Ref<ITexture2D>> mAttachments = {};
    };
} // namespace SE::Graphics