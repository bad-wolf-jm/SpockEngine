#include "IRenderTarget.h"

namespace SE::Graphics
{
    IRenderTarget::IRenderTarget( Ref<IGraphicContext> aGraphicContext, sRenderTargetDescription const &aRenderTargetDescription )
        : IRenderPass{ aGraphicContext, aRenderTargetDescription.mSampleCount }
        , mSpec{ aRenderTargetDescription }
    {
        mImageCount = 1;
    }

    Ref<ITexture> IRenderTarget::GetAttachment( std::string const &aKey ) { return mAttachments[aKey].mTexture; }

    bool IRenderTarget::BeginRender() { return true; }

    void IRenderTarget::EndRender() {}

    void IRenderTarget::Present() {}

    void IRenderTarget::Finalize() {}
} // namespace SE::Graphics
