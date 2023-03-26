#pragma once

#include "BaseImage.h"

#include "Core/Math/Types.h"

#include "Core/GraphicContext/UI/UIContext.h"

namespace SE::Core
{
    class UIImageButton : public UIBaseImage
    {
      public:
        UIImageButton() = default;

        UIImageButton( fs::path const &aImagePath, math::vec2 aSize );
        UIImageButton( Ref<VkSampler2D> aImage, math::vec2 aSize );
        UIImageButton( fs::path const &aImagePath, math::vec2 aSize, std::function<void()> aOnClick );
        UIImageButton( Ref<VkSampler2D> aImage, math::vec2 aSize, std::function<void()> aOnClick );

        void OnClick( std::function<void()> aOnClick );

      private:
        std::function<void()> mOnClick;

      private:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UIImageButton_Create();
        static void *UIImageButton_CreateWithPath( void* aText, math::vec2 *aSize );
        static void  UIImageButton_Destroy( void *aInstance );
    };
} // namespace SE::Core