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
        UIImage( fs::path const &aImagePath, math::vec2 aSize );
        UIImage( ref_t<ISampler2D> aImage, math::vec2 aSize );

      private:
        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UIImage_Create();
        static void *UIImage_CreateWithPath( void* aText, math::vec2 aSize );
        static void  UIImage_Destroy( void *aInstance );
    };
} // namespace SE::Core