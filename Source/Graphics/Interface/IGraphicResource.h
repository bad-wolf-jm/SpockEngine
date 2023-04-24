#pragma once

#include <memory>

#include "Core/Memory.h"

#include "IGraphicContext.h"

namespace SE::Graphics
{
    using namespace SE::Core;

    class IGraphicResource
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
            : mGraphicContext{ aGraphicContext }
            , mIsHostVisible{ aIsHostVisible }
            , mIsTransferSource{ aIsTransferSource }
            , mIsTransferDestination{ aIsTransferDestination }
            , mIsGraphicsOnly{ aIsGraphicsOnly }
        {
        }

        ~IGraphicResource() = default;

        template <typename _GCSubtype>
        Ref<_GCSubtype> GraphicContext()
        {
            return std::reinterpret_pointer_cast<_GCSubtype>( mGraphicContext );
        }

      protected:
        Ref<IGraphicContext> mGraphicContext = nullptr;
    };
} // namespace SE::Graphics
