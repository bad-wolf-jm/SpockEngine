#pragma once

#include "Component.h"

namespace SE::Core
{
    class UIVectorInputBase : public UIComponent
    {
        friend class UIVec2Input;
        friend class UIVec3Input;
        friend class UIVec4Input;

      public:
        UIVectorInputBase( int aDimension );

        UIVectorInputBase( string_t const &aText );

        ImVec2 RequiredSize();
        void   OnChanged( std::function<void( math::vec4 )> aOnChanged );
        void   SetFormat( string_t const &aFormat )
        {
            mFormat = aFormat;
        }

      protected:
        int        mDimension{};
        math::vec4 mValues{};
        math::vec4 mResetValues{};
        string_t   mFormat = "%.2f";

      public:
        std::function<void( math::vec4 )> mOnChanged;

        void *mOnChangeDelegate       = nullptr;
        int   mOnChangeDelegateHandle = -1;

      protected:
        void PushStyles();
        void PopStyles();

        void DrawContent( ImVec2 aPosition, ImVec2 aSize );
    };

    class UIVec2Input : public UIVectorInputBase
    {
      public:
        UIVec2Input()
            : UIVectorInputBase( 2 )
        {
        }

        math::vec2 Value()
        {
            return math::vec2{ mValues.x, mValues.y };
        }
        void SetValue( math::vec2 const &aValue )
        {
            mValues = math::vec4{ aValue, 0.0f, 0.0f };
        }
        void SetResetValues( math::vec2 const &aValue )
        {
            mResetValues = math::vec4{ aValue, 0.0f, 0.0f };
        }
    };

    class UIVec3Input : public UIVectorInputBase
    {
      public:
        UIVec3Input()
            : UIVectorInputBase( 3 )
        {
        }

        math::vec3 Value()
        {
            return math::vec3{ mValues.x, mValues.y, mValues.z };
        }
        void SetValue( math::vec3 const &aValue )
        {
            mValues = math::vec4{ aValue, 0.0f };
        }
        void SetResetValues( math::vec3 const &aValue )
        {
            mResetValues = math::vec4{ aValue, 0.0f };
        }
    };

    class UIVec4Input : public UIVectorInputBase
    {
      public:
        UIVec4Input()
            : UIVectorInputBase( 4 )
        {
        }

        math::vec4 Value()
        {
            return mValues;
        }
        void SetValue( math::vec4 const &aValue )
        {
            mValues = aValue;
        }
        void SetResetValues( math::vec4 const &aValue )
        {
            mResetValues = aValue;
        }
    };

} // namespace SE::Core