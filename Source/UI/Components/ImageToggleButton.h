#pragma once

#include "BaseImage.h"
#include "Component.h"

namespace SE::Core
{
    class UIImageToggleButton : public UIComponent
    {
      public:
        UIImageToggleButton() = default;

        UIImageToggleButton( std::function<void( bool )> aOnClick );

        void OnChange( std::function<void( bool )> aOnClick );

        void SetActiveImage( UIBaseImage const &aImage );
        void SetInactiveImage( UIBaseImage const &aImage );

      private:
        std::function<void( bool )> mOnChange;

        bool        mActivated = false;
        UIBaseImage mActiveImage;
        UIBaseImage mInactiveImage;

      private:
        void PushStyles();
        void PopStyles();

        void PushStyles( bool aEnabled );
        void PopStyles( bool aEnabled );

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };
} // namespace SE::Core