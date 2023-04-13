#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIProgressBar : public UIComponent
    {
      public:
        UIProgressBar() = default;

        UIProgressBar( std::string const &aText );

        void SetText( std::string const &aValue );
        void SetTextColor( math::vec4 aColor );
        void SetProgressValue( float aValue );
        void SetProgressColor( math::vec4 aColor );
        void SetThickness( float aValue );

      protected:
        std::string mText;
        ImVec4      mTextColor;
        float       mValue;
        ImVec4      mProgressColor;
        float       mThickness;

      protected:
        void PushStyles();
        void PopStyles();

        ImVec2 RequiredSize();
        void   DrawContent( ImVec2 aPosition, ImVec2 aSize );

      public:
        static void *UIProgressBar_Create();
        static void  UIProgressBar_Destroy( void *aInstance );
        static void  UIProgressBar_SetProgressValue( void *aInstance, float aValue );
        static void  UIProgressBar_SetProgressColor( void *aInstance, math::vec4 aProgressColor );
        static void  UIProgressBar_SetText( void *aInstance, void *aValue );
        static void  UIProgressBar_SetTextColor( void *aInstance, math::vec4 aTextColor );
        static void  UIProgressBar_SetThickness( void *aInstance, float aValue );
    };
} // namespace SE::Core