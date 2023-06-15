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
        void   SetFormat( string_t const &aFormat ) { mFormat = aFormat; }

      protected:
        int         mDimension{};
        math::vec4  mValues{};
        math::vec4  mResetValues{};
        string_t mFormat = "%.2f";

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

        math::vec2 Value() { return math::vec2{ mValues.x, mValues.y }; }
        void       SetValue( math::vec2 const &aValue ) { mValues = math::vec4{ aValue, 0.0f, 0.0f }; }
        void       SetResetValues( math::vec2 const &aValue ) { mResetValues = math::vec4{ aValue, 0.0f, 0.0f }; }

    //   public:
    //     static void      *UIVec2Input_Create();
    //     static void       UIVec2Input_Destroy( void *aInstance );
    //     static void       UIVec2Input_OnChanged( void *aInstance, void *aDelegate );
    //     static void       UIVec2Input_SetValue( void *aInstance, math::vec2 aValue );
    //     static math::vec2 UIVec2Input_GetValue( void *aInstance );
    //     static void       UIVec2Input_SetFormat( void *aInstance, void *aFormat );
    //     static void       UIVec2Input_SetResetValues( void *aInstance, math::vec2 aValues );
    };

    class UIVec3Input : public UIVectorInputBase
    {
      public:
        UIVec3Input()
            : UIVectorInputBase( 3 )
        {
        }

        math::vec3 Value() { return math::vec3{ mValues.x, mValues.y, mValues.z }; }
        void       SetValue( math::vec3 const &aValue ) { mValues = math::vec4{ aValue, 0.0f }; }
        void       SetResetValues( math::vec3 const &aValue ) { mResetValues = math::vec4{ aValue, 0.0f }; }

    //   public:
    //     static void      *UIVec3Input_Create();
    //     static void       UIVec3Input_Destroy( void *aInstance );
    //     static void       UIVec3Input_OnChanged( void *aInstance, void *aDelegate );
    //     static void       UIVec3Input_SetValue( void *aInstance, math::vec3 aValue );
    //     static math::vec3 UIVec3Input_GetValue( void *aInstance );
    //     static void       UIVec3Input_SetFormat( void *aInstance, void *aFormat );
    //     static void       UIVec3Input_SetResetValues( void *aInstance, math::vec3 aValues );
    };

    class UIVec4Input : public UIVectorInputBase
    {
      public:
        UIVec4Input()
            : UIVectorInputBase( 4 )
        {
        }

        math::vec4 Value() { return mValues; }
        void       SetValue( math::vec4 const &aValue ) { mValues = aValue; }
        void       SetResetValues( math::vec4 const &aValue ) { mResetValues = aValue; }

    //   public:
    //     static void      *UIVec4Input_Create();
    //     static void       UIVec4Input_Destroy( void *aInstance );
    //     static void       UIVec4Input_OnChanged( void *aInstance, void *aDelegate );
    //     static void       UIVec4Input_SetValue( void *aInstance, math::vec4 aValue );
    //     static math::vec4 UIVec4Input_GetValue( void *aInstance );
    //     static void       UIVec4Input_SetFormat( void *aInstance, void *aFormat );
    //     static void       UIVec4Input_SetResetValues( void *aInstance, math::vec4 aValues );
    };

} // namespace SE::Core