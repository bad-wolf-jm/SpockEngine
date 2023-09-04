#pragma once

#include "BaseImage.h"
#include "Component.h"

namespace SE::Core
{
    class UIImageToggleButton : public UIComponent
    {
      public:
        UIImageToggleButton() = default;

        UIImageToggleButton( std::function<bool( bool )> aOnClick );

        void OnClick( std::function<bool( bool )> aOnClick );
        void OnChanged( std::function<void()> aOnChanged );

        bool IsActive();
        void SetActive( bool aValue );

        void SetActiveImage( UIBaseImage *aImage );
        void SetInactiveImage( UIBaseImage *aImage );

      private:
        std::function<bool( bool )> mOnClicked;
        std::function<void()>       mOnChanged;

        bool mActivated = false;

        UIBaseImage *mActiveImage   = nullptr;
        UIBaseImage *mInactiveImage = nullptr;

      private:
        void PushStyles();
        void PopStyles();

        void PushStyles( bool aEnabled );
        void PopStyles( bool aEnabled );

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core