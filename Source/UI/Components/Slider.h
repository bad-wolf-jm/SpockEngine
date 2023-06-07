#pragma once

#include "Component.h"

namespace SE::Core
{
    class UISlider : public UIComponent
    {
      public:
        UISlider() = default;

        ImVec2 RequiredSize();

      protected:
        float mValue;

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );

    //   public:
    //     static void *UISlider_Create();
    //     static void  UISlider_Destroy( void *aInstance );
    };
} // namespace SE::Core