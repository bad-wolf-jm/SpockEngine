#pragma once

#include <memory>

#include "Core/Memory.h"

#include "IGraphicContext.h"

#include "Core/CUDA/Array/PointerView.h"

namespace SE::Graphics
{
    using namespace SE::Core;
    using namespace SE::Graphics::Internal;

    class IGraphicResource : public Cuda::Internal::sGPUDevicePointerView
    {
      public:
        bool mIsHostVisible         = true;
        bool mIsGraphicsOnly        = true;
        bool mIsTransferSource      = true;
        bool mIsTransferDestination = true;

        IGraphicResource()                     = default;
        IGraphicResource( IGraphicResource & ) = default;

        IGraphicResource( Ref<IGraphicContext> aGraphicContext, bool aIsHostVisible, bool aIsGraphicsOnly, bool aIsTransferSource,
                          bool aIsTransferDestination, size_t aSize )
            : Cuda::Internal::sGPUDevicePointerView( aSize, nullptr )
            , mGraphicConext{ aGraphicContext }
            , mIsHostVisible{ aIsHostVisible }
            , mIsTransferSource{ aIsTransferSource }
            , mIsTransferDestination{ aIsTransferDestination }
            , mIsGraphicsOnly{ aIsGraphicsOnly }
        {
        }

        ~IGraphicResource() = default;

      private:
        Ref<IGraphicContext> mGraphicContext = nullptr;
    };
} // namespace SE::Graphics
