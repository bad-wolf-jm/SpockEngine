#pragma once

#include "BaseImage.h"

#include "Core/Math/Types.h"

#include "UI/UIContext.h"

namespace SE::Core
{
    class UIImage : public UIBaseImage
    {
      public:
        UIImage() = default;
        UIImage( path_t const &aImagePath, math::vec2 aSize );
        UIImage( Ref<ISampler2D> aImage, math::vec2 aSize );

      private:
        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core