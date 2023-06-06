#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIVectorInputBase : public UIComponent
    {
      public:
        UIVectorInputBase( int aDimension );

        UIVectorInputBase( std::string const &aText );

        ImVec2 RequiredSize();

      protected:
        int         mDimension{};
        math::vec4  mValues{};
        math::vec4  mResetValues{};
        std::string mFormat = ".2f";

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };

    class UIVec2Input : public UIVectorInputBase
    {
        UIVec2Input()
            : UIVectorInputBase( 2 )
        {
        }

        math::vec2 Value() { return math::vec2{ mValues.x, mValues.y }; }
        void       SetValue( math::vec2 const &aValue ) { mValues = math::vec4{ aValue, 0.0f, 0.0f }; }
    };

    class UIVec3Input : public UIVectorInputBase
    {
        UIVec3Input()
            : UIVectorInputBase( 3 )
        {
        }

        math::vec3 Value() { return math::vec3{ mValues.x, mValues.y, mValues.z }; }
        void       SetValue( math::vec3 const &aValue ) { mValues = math::vec4{ aValue, 0.0f }; }
    };

    class UIVec4Input : public UIVectorInputBase
    {
        UIVec4Input()
            : UIVectorInputBase( 4 )
        {
        }

        math::vec4 Value() { return mValues; }
        void       SetValue( math::vec4 const &aValue ) { mValues = aValue; }
    };

} // namespace SE::Core