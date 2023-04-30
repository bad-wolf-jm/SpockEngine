#include "ICommandBuffer.h"

#include "Core/Core.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

namespace SE::Graphics
{
    ICommandBuffer::ICommandBuffer( Ref<IGraphicContext> aGraphicContext )
        : mGraphicContext{ aGraphicContext }
    {
    }
} // namespace SE::Graphics
