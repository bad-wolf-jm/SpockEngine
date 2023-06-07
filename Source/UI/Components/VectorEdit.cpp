#include "VectorEdit.h"
#include "DotNet/Runtime.h"

namespace SE::Core
{
    static constexpr ImVec4 gXColors[] = { ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f }, ImVec4{ 0.9f, 0.2f, 0.2f, 1.0f },
                                           ImVec4{ 0.8f, 0.1f, 0.15f, 1.0f } };

    static constexpr ImVec4 gYColors[] = { ImVec4{ 0.2f, 0.5f, 0.2f, 1.0f }, ImVec4{ 0.3f, 0.6f, 0.3f, 1.0f },
                                           ImVec4{ 0.2f, 0.5f, 0.2f, 1.0f } };

    static constexpr ImVec4 gZColors[] = { ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f }, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f },
                                           ImVec4{ 0.1f, 0.25f, 0.8f, 1.0f } };

    static constexpr ImVec4 gWColors[] = { ImVec4{ 0.1f, 0.05f, 0.8f, 1.0f }, ImVec4{ 0.2f, 0.35f, 0.9f, 1.0f },
                                           ImVec4{ 0.1f, 0.05f, 0.8f, 1.0f } };

    static bool EditVectorComponent( const char *aLabel, const char *aFormat, float *aValue, float aResetValue, ImVec2 aButtonSize,
                                     float aWidth, ImVec4 const *aColors )
    {
        bool lHasChanged = false;

        ImGui::PushStyleColor( ImGuiCol_Button, aColors[0] );
        ImGui::PushStyleColor( ImGuiCol_ButtonHovered, aColors[1] );
        ImGui::PushStyleColor( ImGuiCol_ButtonActive, aColors[2] );
        if( ImGui::Button( aLabel, aButtonSize ) )
        {
            if( *aValue != aResetValue )
            {
                lHasChanged = true;
                *aValue     = aResetValue;
            }
        }
        ImGui::PopStyleColor( 3 );

        ImGui::SameLine();

        ImGui::PushID( aLabel );
        ImGui::SetNextItemWidth( aWidth - aButtonSize.x );
        lHasChanged |= ImGui::DragFloat( "##INPUT", aValue, 0.1f, 0.0f, 0.0f, aFormat, ImGuiSliderFlags_AlwaysClamp );
        ImGui::PopID();

        return lHasChanged;
    }

    static bool VectorComponentEditor( const char *aFormat, int aDimension, float *aValues, float *aResetValue, float aWidth )
    {
        bool   lHasChanged = false;
        float  lLineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
        ImVec2 lButtonSize = { lLineHeight + 3.0f, lLineHeight };

        if( aDimension >= 1 )
            lHasChanged |= EditVectorComponent( "X", aFormat, &aValues[0], aResetValue[0], lButtonSize, aWidth, gXColors );

        if( aDimension >= 2 )
            lHasChanged |= EditVectorComponent( "Y", aFormat, &aValues[1], aResetValue[1], lButtonSize, aWidth, gYColors );

        if( aDimension >= 3 )
            lHasChanged |= EditVectorComponent( "Z", aFormat, &aValues[2], aResetValue[2], lButtonSize, aWidth, gZColors );

        if( aDimension >= 4 )
            lHasChanged |= EditVectorComponent( "W", aFormat, &aValues[3], aResetValue[3], lButtonSize, aWidth, gWColors );

        return lHasChanged;
    }

    UIVectorInputBase::UIVectorInputBase( int aDimension )
        : mDimension{ aDimension }
    {
    }

    void UIVectorInputBase::PushStyles()
    {
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2{ 2, 2 } );
        ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2{ 0, 1 } );
        ImGui::PushStyleColor( ImGuiCol_FrameBg, ImVec4{ .03, 0.03, 0.03, 1.0 } );
        ImGui::PushStyleColor( ImGuiCol_FrameBgHovered, ImVec4{ .04, 0.04, 0.04, 1.0 } );
    }

    void UIVectorInputBase::PopStyles()
    {
        ImGui::PopStyleColor( 2 );
        ImGui::PopStyleVar( 2 );
    }

    void UIVectorInputBase::OnChanged( std::function<void( math::vec4 )> aOnChanged ) { mOnChanged = aOnChanged; }

    ImVec2 UIVectorInputBase::RequiredSize()
    {
        return ImVec2{ 100.0f, ( GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f ) * mDimension };
    }

    void UIVectorInputBase::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        VectorComponentEditor( mFormat.c_str(), mDimension, (float *)&mValues, (float *)&mResetValues, aSize.x );
    }

    // void *UIVec2Input::UIVec2Input_Create()
    // {
    //     auto lNewVecInput = new UIVec2Input();

    //     return static_cast<void *>( lNewVecInput );
    // }

    // void UIVec2Input::UIVec2Input_Destroy( void *aInstance ) { delete static_cast<UIVec2Input *>( aInstance ); }

    // void UIVec2Input::UIVec2Input_OnChanged( void *aInstance, void *aDelegate )
    // {
    //     auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
    //     auto lDelegate = static_cast<MonoObject *>( aDelegate );

    //     if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

    //     lInstance->mOnChangeDelegate       = aDelegate;
    //     lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

    //     lInstance->OnChanged(
    //         [lInstance, lDelegate]( math::vec4 aVector )
    //         {
    //             auto lDelegateClass = mono_object_get_class( lDelegate );
    //             auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

    //             math::vec2 lProjection = math::vec2{ aVector.x, aVector.y };
    //             void      *lParams[]   = { (void *)&lProjection };
    //             auto       lValue      = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
    //         } );
    // }

    // void UIVec2Input::UIVec2Input_SetValue( void *aInstance, math::vec2 aValue )
    // {
    //     static_cast<UIVec2Input *>( aInstance )->SetValue( aValue );
    // }

    // math::vec2 UIVec2Input::UIVec2Input_GetValue( void *aInstance ) { return static_cast<UIVec2Input *>( aInstance )->Value(); }

    // void UIVec2Input::UIVec2Input_SetResetValues( void *aInstance, math::vec2 aValue )
    // {
    //     static_cast<UIVec2Input *>( aInstance )->SetResetValues( aValue );
    // }

    // void UIVec2Input::UIVec2Input_SetFormat( void *aInstance, void *aText )
    // {
    //     auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
    //     auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

    //     lInstance->SetFormat( lString );
    // }

    // void *UIVec3Input::UIVec3Input_Create()
    // {
    //     auto lNewVecInput = new UIVec3Input();

    //     return static_cast<void *>( lNewVecInput );
    // }

    // void UIVec3Input::UIVec3Input_Destroy( void *aInstance ) { delete static_cast<UIVec3Input *>( aInstance ); }

    // void UIVec3Input::UIVec3Input_OnChanged( void *aInstance, void *aDelegate )
    // {
    //     auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
    //     auto lDelegate = static_cast<MonoObject *>( aDelegate );

    //     if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

    //     lInstance->mOnChangeDelegate       = aDelegate;
    //     lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

    //     lInstance->OnChanged(
    //         [lInstance, lDelegate]( math::vec4 aVector )
    //         {
    //             auto lDelegateClass = mono_object_get_class( lDelegate );
    //             auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

    //             math::vec3 lProjection = math::vec3{ aVector.x, aVector.y, aVector.z };
    //             void      *lParams[]   = { (void *)&lProjection };
    //             auto       lValue      = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
    //         } );
    // }

    // void UIVec3Input::UIVec3Input_SetValue( void *aInstance, math::vec3 aValue )
    // {
    //     static_cast<UIVec3Input *>( aInstance )->SetValue( aValue );
    // }

    // math::vec3 UIVec3Input::UIVec3Input_GetValue( void *aInstance ) { return static_cast<UIVec3Input *>( aInstance )->Value(); }

    // void UIVec3Input::UIVec3Input_SetResetValues( void *aInstance, math::vec3 aValue )
    // {
    //     static_cast<UIVec3Input *>( aInstance )->SetResetValues( aValue );
    // }

    // void UIVec3Input::UIVec3Input_SetFormat( void *aInstance, void *aText )
    // {
    //     auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
    //     auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

    //     lInstance->SetFormat( lString );
    // }

    // void *UIVec4Input::UIVec4Input_Create()
    // {
    //     auto lNewVecInput = new UIVec4Input();

    //     return static_cast<void *>( lNewVecInput );
    // }

    // void UIVec4Input::UIVec4Input_Destroy( void *aInstance ) { delete static_cast<UIVec4Input *>( aInstance ); }

    // void UIVec4Input::UIVec4Input_OnChanged( void *aInstance, void *aDelegate )
    // {
    //     auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
    //     auto lDelegate = static_cast<MonoObject *>( aDelegate );

    //     if( lInstance->mOnChangeDelegate != nullptr ) mono_gchandle_free( lInstance->mOnChangeDelegateHandle );

    //     lInstance->mOnChangeDelegate       = aDelegate;
    //     lInstance->mOnChangeDelegateHandle = mono_gchandle_new( static_cast<MonoObject *>( aDelegate ), true );

    //     lInstance->OnChanged(
    //         [lInstance, lDelegate]( math::vec4 aVector )
    //         {
    //             auto lDelegateClass = mono_object_get_class( lDelegate );
    //             auto lInvokeMethod  = mono_get_delegate_invoke( lDelegateClass );

    //             void *lParams[] = { (void *)&aVector };
    //             auto  lValue    = mono_runtime_invoke( lInvokeMethod, lDelegate, lParams, nullptr );
    //         } );
    // }

    // void UIVec4Input::UIVec4Input_SetValue( void *aInstance, math::vec4 aValue )
    // {
    //     static_cast<UIVec4Input *>( aInstance )->SetValue( aValue );
    // }

    // math::vec4 UIVec4Input::UIVec4Input_GetValue( void *aInstance ) { return static_cast<UIVec4Input *>( aInstance )->Value(); }

    // void UIVec4Input::UIVec4Input_SetResetValues( void *aInstance, math::vec4 aValue )
    // {
    //     static_cast<UIVec4Input *>( aInstance )->SetResetValues( aValue );
    // }

    // void UIVec4Input::UIVec4Input_SetFormat( void *aInstance, void *aText )
    // {
    //     auto lInstance = static_cast<UIVectorInputBase *>( aInstance );
    //     auto lString   = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

    //     lInstance->SetFormat( lString );
    // }

} // namespace SE::Core