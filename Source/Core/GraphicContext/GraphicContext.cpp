#include "GraphicContext.h"

namespace SE::Graphics
{
    GraphicContext::GraphicContext( Ref<Internal::VkGraphicContext> aContext, Ref<IWindow> aWindow )
        : mViewportClient{ aWindow }
        , mContext{ aContext }
    {
    }
} // namespace SE::Graphics