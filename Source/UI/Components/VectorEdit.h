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

        UIVectorInputBase( std::string const &aText );

        ImVec2 RequiredSize();
        void   OnChanged( std::function<void( math::vec4 )> aOnChanged );

      protected:
        int         mDimension{};
        math::vec4  mValues{};
        math::vec4  mResetValues{};
        std::string mFormat = ".2f";

      protected:
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
        UIVec2Input()
            : UIVectorInputBase( 2 )
        {
        }

        math::vec2 Value() { return math::vec2{ mValues.x, mValues.y }; }
        void       SetValue( math::vec2 const &aValue ) { mValues = math::vec4{ aValue, 0.0f, 0.0f }; }

      public:
        static void      *UIVec2Input_Create();
        static void       UIVec2Input_Destroy( void *aInstance );
        static void       UIVec2Input_OnChanged( void *aInstance, void *aDelegate );
        static void       UIVec2Input_SetValue( void *aInstance, math::vec2 aValue );
        static math::vec2 UIVec2Input_GetValue( void *aInstance );
    };

    class UIVec3Input : public UIVectorInputBase
    {
        UIVec3Input()
            : UIVectorInputBase( 3 )
        {
        }

        math::vec3 Value() { return math::vec3{ mValues.x, mValues.y, mValues.z }; }
        void       SetValue( math::vec3 const &aValue ) { mValues = math::vec4{ aValue, 0.0f }; }

      public:
        static void      *UIVec3Input_Create();
        static void       UIVec3Input_Destroy( void *aInstance );
        static void       UIVec3Input_OnChanged( void *aInstance, void *aDelegate );
        static void       UIVec3Input_SetValue( void *aInstance, math::vec3 aValue );
        static math::vec3 UIVec3Input_GetValue( void *aInstance );
    };

    class UIVec4Input : public UIVectorInputBase
    {
        UIVec4Input()
            : UIVectorInputBase( 4 )
        {
        }

        math::vec4 Value() { return mValues; }
        void       SetValue( math::vec4 const &aValue ) { mValues = aValue; }

      public:
        static void      *UIVec4Input_Create();
        static void       UIVec4Input_Destroy( void *aInstance );
        static void       UIVec4Input_OnChanged( void *aInstance, void *aDelegate );
        static void       UIVec4Input_SetValue( void *aInstance, math::vec4 aValue );
        static math::vec4 UIVec4Input_GetValue( void *aInstance );
    };

} // namespace SE::Core