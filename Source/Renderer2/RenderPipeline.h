#include "Core/Math/Types.h"

#include "Core/Memory.h"

#include "Graphics/API.h"

#include "Scene/VertexData.h"

namespace SE::Core
{
    class RenderPipeline
    {
      private:
        Ref<IGraphicsPipeline> mPipeline = nullptr;
    }
} // namespace SE::Core