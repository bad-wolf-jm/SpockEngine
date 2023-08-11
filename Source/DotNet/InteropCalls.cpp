#include "InteropCalls.h"

#include "Core/File.h"
#include "Core/Profiling/BlockTimer.h"
#include "Core/String.h"

#include "Engine/Engine.h"

namespace SE::Core::Interop
{
    static vec2 vec( CLRVec2 v )
    {
        return vec2{ v.x, v.y };
    }
    static vec3 vec( CLRVec3 v )
    {
        return vec3{ v.x, v.y, v.z };
    }
    static vec4 vec( CLRVec4 v )
    {
        return vec4{ v.x, v.y, v.z, v.w };
    }

    static CLRVec2 vec( ImVec2 v )
    {
        return CLRVec2{ v.x, v.y };
    }
    static CLRVec4 vec( ImVec4 v )
    {
        return CLRVec4{ v.x, v.y, v.z, v.w };
    }
    static ImVec2 imvec( CLRVec2 v )
    {
        return ImVec2{ v.x, v.y };
    }
    static ImVec4 imvec( CLRVec4 v )
    {
        return ImVec4{ v.x, v.y, v.z, v.w };
    }

    static CLRVec2 vec( vec2 v )
    {
        return CLRVec2{ v.x, v.y };
    }
    static CLRVec3 vec( vec3 v )
    {
        return CLRVec3{ v.x, v.y, v.z };
    }
    static CLRVec4 vec( vec4 v )
    {
        return CLRVec4{ v.x, v.y, v.z, v.w };
    }

    extern "C"
    {
#define CONSTRUCT_WITHOUT_PARAMETERS( _Ty ) \
    void *_Ty##_Create()                    \
    {                                       \
        auto lNewObject = new _Ty();        \
        return CAST( void, lNewObject );    \
    }
#define DESTROY_INTERFACE( _Ty )     \
    void _Ty##_Destroy( _Ty *aSelf ) \
    {                                \
        delete aSelf;                \
    }
#define CAST( _Ty, v ) static_cast<_Ty *>( v )

        CLRVec4 UIColors_GetStyleColor( ImGuiCol aColor )
        {
            auto const &lColor = ImGui::GetStyleColorVec4( aColor );

            return CLRVec4{ lColor.x, lColor.y, lColor.z, lColor.w };
        }

        wchar_t *OpenFile( wchar_t *aFilter )
        {
            auto     lFilter     = std::wstring( aFilter );
            wchar_t *lCharacters = lFilter.data();

            for( uint32_t i = 0; i < lFilter.size(); i++ )
                lCharacters[i] = ( lCharacters[i] == '|' ) ? '\0' : lCharacters[i];
            auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(), lFilter.c_str() );

            if( lFilePath.has_value() )
                return CopyCharactersForCoreClr( lFilePath.value() );

            return nullptr;
        }

#pragma region UIBaseImage
        CONSTRUCT_WITHOUT_PARAMETERS( UIBaseImage )
        DESTROY_INTERFACE( UIBaseImage )

        void UIBaseImage_SetImage( UIBaseImage *aSelf, wchar_t *aPath )
        {
            aSelf->SetImage( ConvertStringForCoreclr( aPath ) );
        }

        void UIBaseImage_SetSize( UIBaseImage *aSelf, CLRVec2 aSize )
        {
            aSelf->SetSize( vec( aSize ) );
        }

        CLRVec2 UIBaseImage_GetSize( UIBaseImage *aSelf )
        {
            return vec( aSelf->Size() );
        }

        void UIBaseImage_SetTopLeft( UIBaseImage *aSelf, CLRVec2 aTopLeft )
        {
            aSelf->SetTopLeft( vec2{ aTopLeft.x, aTopLeft.y } );
        }

        CLRVec2 UIBaseImage_GetTopLeft( UIBaseImage *aSelf )
        {
            return vec( aSelf->TopLeft() );
        }

        void UIBaseImage_SetBottomRight( UIBaseImage *aSelf, CLRVec2 aBottomRight )
        {
            aSelf->SetBottomRight( vec( aBottomRight ) );
        }

        CLRVec2 UIBaseImage_GetBottomRight( UIBaseImage *aSelf )
        {
            return vec( aSelf->BottomRight() );
        }

        void UIBaseImage_SetTintColor( UIBaseImage *aSelf, CLRVec4 aColor )
        {
            aSelf->SetTintColor( vec( aColor ) );
        }

        CLRVec4 UIBaseImage_GetTintColor( UIBaseImage *aSelf )
        {
            return vec( aSelf->TintColor() );
        }
#pragma endregion

#pragma region UIButton
        CONSTRUCT_WITHOUT_PARAMETERS( UIButton )
        DESTROY_INTERFACE( UIButton )

        void UIButton_SetText( UIButton *aSelf, wchar_t *aText )
        {
            aSelf->SetText( ConvertStringForCoreclr( aText ) );
        }

        void UIButton_OnClick( UIButton *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )();
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnClick( [lInstance, lDelegate]() { lDelegate(); } );
        }
#pragma endregion

#pragma region UICheckBox
        CONSTRUCT_WITHOUT_PARAMETERS( UICheckBox )
        DESTROY_INTERFACE( UICheckBox )

        void UICheckBox_OnClick( UICheckBox *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )();
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnClick( [lInstance, lDelegate]() { lDelegate(); } );
        }

        bool UICheckBox_IsChecked( UICheckBox *aSelf )
        {
            return aSelf->IsChecked();
        }

        void UICheckBox_SetIsChecked( UICheckBox *aSelf, bool aValue )
        {
            aSelf->SetIsChecked( aValue );
        }
#pragma endregion

#pragma region UIColorButton
        CONSTRUCT_WITHOUT_PARAMETERS( UIColorButton )
        DESTROY_INTERFACE( UIColorButton )
#pragma endregion

#pragma region UIComboBox
        CONSTRUCT_WITHOUT_PARAMETERS( UIComboBox )
        DESTROY_INTERFACE( UIComboBox )

        int UIComboBox_GetCurrent( UIComboBox *aSelf )
        {
            return aSelf->Current();
        }

        void UIComboBox_SetCurrent( UIComboBox *aSelf, int aValue )
        {
            aSelf->SetCurrent( aValue );
        }

        void UIComboBox_SetItemList( UIComboBox *aSelf, wchar_t **aItems, int aLength )
        {
            vector_t<string_t> lItemVector;
            for( int i = 0; i < aLength; i++ )
                lItemVector.emplace_back( ConvertStringForCoreclr( aItems[i] ) );

            aSelf->SetItemList( lItemVector );
        }

        void UIComboBox_OnChanged( UIComboBox *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )( int );
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnChange( [lInstance, lDelegate]( int i ) { lDelegate( i ); } );
        }
#pragma endregion

#pragma region UIComponent
        void UIComponent_SetIsVisible( UIComponent *aSelf, bool aIsVisible )
        {
            aSelf->mIsVisible = aIsVisible;
        }

        void UIComponent_SetIsEnabled( UIComponent *aSelf, bool aIsEnabled )
        {
            aSelf->mIsEnabled = aIsEnabled;
        }

        void UIComponent_SetAllowDragDrop( UIComponent *aSelf, bool aAllowDragDrop )
        {
            aSelf->mAllowDragDrop = aAllowDragDrop;
        }

        void UIComponent_SetPaddingAll( UIComponent *aSelf, float aPaddingAll )
        {
            aSelf->SetPadding( aPaddingAll );
        }

        void UIComponent_SetPaddingPairs( UIComponent *aSelf, float aPaddingTopBottom, float aPaddingLeftRight )
        {
            aSelf->SetPadding( aPaddingTopBottom, aPaddingLeftRight );
        }

        void UIComponent_SetPaddingIndividual( UIComponent *aSelf, float aPaddingTop, float aPaddingBottom, float aPaddingLeft,
                                               float aPaddingRight )
        {
            aSelf->SetPadding( aPaddingTop, aPaddingBottom, aPaddingLeft, aPaddingRight );
        }

        void UIComponent_SetAlignment( UIComponent *aSelf, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
        {
            aSelf->SetAlignment( aHAlignment, aVAlignment );
        }

        void UIComponent_SetHorizontalAlignment( UIComponent *aSelf, eHorizontalAlignment aAlignment )
        {
            aSelf->SetHorizontalAlignment( aAlignment );
        }

        void UIComponent_SetVerticalAlignment( UIComponent *aSelf, eVerticalAlignment aAlignment )
        {
            aSelf->SetVerticalAlignment( aAlignment );
        }

        void UIComponent_SetBackgroundColor( UIComponent *aSelf, CLRVec4 aColor )
        {
            aSelf->SetBackgroundColor( vec( aColor ) );
        }

        void UIComponent_SetFont( UIComponent *aSelf, FontFamilyFlags aFont )
        {
            aSelf->SetFont( aFont );
        }

        void UIComponent_SetTooltip( UIComponent *aSelf, void *aTooltip )
        {
            auto lTooltip = CAST( UIComponent, aTooltip );

            aSelf->SetTooltip( lTooltip );
        }
#pragma endregion

#pragma region UIDropdownButton
        CONSTRUCT_WITHOUT_PARAMETERS( UIDropdownButton )
        DESTROY_INTERFACE( UIDropdownButton )

        void UIDropdownButton_SetContent( UIDropdownButton *aSelf, UIComponent *aContent )
        {
            return aSelf->SetContent( aContent );
        }

        void UIDropdownButton_SetContentSize( UIDropdownButton *aSelf, CLRVec2 aContentSizse )
        {
            return aSelf->SetContentSize( vec( aContentSizse ) );
        }

        void UIDropdownButton_SetImage( UIDropdownButton *aSelf, UIImage *aImage )
        {
            aSelf->SetImage( aImage );
        }

        void UIDropdownButton_SetText( UIDropdownButton *aSelf, wchar_t *aText )
        {
            aSelf->SetText( ConvertStringForCoreclr( aText ) );
        }

        void UIDropdownButton_SetTextColor( UIDropdownButton *aSelf, CLRVec4 aColor )
        {
            aSelf->SetTextColor( vec( aColor ) );
        }
#pragma endregion

#pragma region UIImage
        CONSTRUCT_WITHOUT_PARAMETERS( UIImage )
        DESTROY_INTERFACE( UIImage )
#pragma endregion

#pragma region UIImageButton
        CONSTRUCT_WITHOUT_PARAMETERS( UIImageButton )
        DESTROY_INTERFACE( UIImageButton )

        void UIImageButton_OnClick( UIImageButton *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )();
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnClick( [lInstance, lDelegate]() { lDelegate(); } );
        }
#pragma endregion

#pragma region UIImageToggleButton
        CONSTRUCT_WITHOUT_PARAMETERS( UIImageToggleButton )
        DESTROY_INTERFACE( UIImageToggleButton )

        bool UIImageToggleButton_IsActive( UIImageToggleButton *aSelf )
        {
            return aSelf->IsActive();
        }

        void UIImageToggleButton_SetActive( UIImageToggleButton *aSelf, bool aValue )
        {
            aSelf->SetActive( aValue );
        }

        void UIImageToggleButton_SetActiveImage( UIImageToggleButton *aSelf, UIBaseImage *aImage )
        {
            aSelf->SetActiveImage( aImage );
        }

        void UIImageToggleButton_SetInactiveImage( UIImageToggleButton *aSelf, UIBaseImage *aImage )
        {
            aSelf->SetInactiveImage( aImage );
        }

        void UIImageToggleButton_OnClicked( UIImageToggleButton *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef bool ( *fptr )( bool );
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnClick( [lInstance, lDelegate]( bool i ) { return lDelegate( i ); } );
        }

        void UIImageToggleButton_OnChanged( UIImageToggleButton *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef bool ( *fptr )();
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnChanged( [lInstance, lDelegate]() { return lDelegate(); } );
        }
#pragma endregion

#pragma region UILabel
        CONSTRUCT_WITHOUT_PARAMETERS( UILabel )
        DESTROY_INTERFACE( UILabel )

        void UILabel_SetText( UILabel *aSelf, wchar_t *aText )
        {
            aSelf->SetText( ConvertStringForCoreclr( aText ) );
        }

        void UILabel_SetTextColor( UILabel *aSelf, CLRVec4 aTextColor )
        {
            aSelf->SetTextColor( vec( aTextColor ) );
        }
#pragma endregion

#pragma region UIMenuItem
        CONSTRUCT_WITHOUT_PARAMETERS( UIMenuItem )
        DESTROY_INTERFACE( UIMenuItem )

        void UIMenuItem_SetText( UIMenuItem *aSelf, wchar_t *aText )
        {
            aSelf->SetText( ConvertStringForCoreclr( aText ) );
        }

        void UIMenuItem_SetShortcut( UIMenuItem *aSelf, wchar_t *aShortcut )
        {
            aSelf->SetShortcut( ConvertStringForCoreclr( aShortcut ) );
        }

        void UIMenuItem_SetTextColor( UIMenuItem *aSelf, CLRVec4 aTextColor )
        {
            aSelf->SetTextColor( vec( aTextColor ) );
        }

        void UIMenuItem_OnTrigger( UIMenuItem *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )();
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnTrigger( [lInstance, lDelegate]() { lDelegate(); } );
        }
#pragma endregion

#pragma region UIMenuSeparator
        CONSTRUCT_WITHOUT_PARAMETERS( UIMenuSeparator )
        DESTROY_INTERFACE( UIMenuSeparator )
#pragma endregion

#pragma region UIMenu
        CONSTRUCT_WITHOUT_PARAMETERS( UIMenu )
        DESTROY_INTERFACE( UIMenu )

        void *UIMenu_AddAction( UIMenu *aSelf, wchar_t *aText, wchar_t *aShortcut )
        {
            return CAST( void, aSelf->AddActionRaw( ConvertStringForCoreclr( aText ), ConvertStringForCoreclr( aShortcut ) ) );
        }

        void *UIMenu_AddMenu( UIMenu *aSelf, wchar_t *aText )
        {
            return CAST( void, aSelf->AddMenuRaw( ConvertStringForCoreclr( aText ) ) );
        }

        void *UIMenu_AddSeparator( UIMenu *aSelf )
        {
            return CAST( void, aSelf->AddSeparatorRaw() );
        }

        void UIMenu_Update( UIMenu *aSelf )
        {
            aSelf->Update();
        }
#pragma endregion

#pragma region UIPlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIPlot )
        DESTROY_INTERFACE( UIPlot )

        void UIPlot_Clear( UIPlot *aSelf )
        {
            aSelf->Clear();
        }

        void UIPlot_ConfigureLegend( UIPlot *aSelf, CLRVec2 aLegendPadding, CLRVec2 aLegendInnerPadding, CLRVec2 aLegendSpacing )
        {
            aSelf->ConfigureLegend( vec( aLegendPadding ), vec( aLegendInnerPadding ), vec( aLegendSpacing ) );
        }

        void UIPlot_Add( UIPlot *aSelf, UIPlotData *aPlot )
        {
            aSelf->Add( aPlot );
        }

        void UIPlot_SetAxisLimits( UIPlot *aSelf, int aAxis, double aMin, double aMax )
        {
            auto lSelf = aSelf;

            lSelf->mAxisConfiguration[aAxis].mSetLimitRequest = true;
            lSelf->mAxisConfiguration[aAxis].mMin             = static_cast<float>( aMin );
            lSelf->mAxisConfiguration[aAxis].mMax             = static_cast<float>( aMax );
        }

        void UIPlot_SetAxisTitle( UIPlot *aSelf, int aAxis, wchar_t *aTitle )
        {
            aSelf->mAxisConfiguration[aAxis].mTitle = ConvertStringForCoreclr( aTitle );
        }

        wchar_t *UIPlot_GetAxisTitle( UIPlot *aSelf, int aAxis )
        {
            return ConvertStringForCoreclr( aSelf->mAxisConfiguration[aAxis].mTitle.data() ).data();
        }
#pragma endregion

#pragma region UIPlotData
        void UIPlotData_SetLegend( UIPlotData *aSelf, wchar_t *aText )
        {
            aSelf->mLegend = ConvertStringForCoreclr( aText );
        }

        void UIPlotData_SetThickness( UIPlotData *aSelf, float aThickness )
        {
            aSelf->mThickness = aThickness;
        }

        void UIPlotData_SetColor( UIPlotData *aSelf, CLRVec4 aColor )
        {
            aSelf->mColor = vec( aColor );
        }

        void UIPlotData_SetXAxis( UIPlotData *aSelf, int aAxis )
        {
            aSelf->mXAxis = static_cast<UIPlotAxis>( aAxis );
        }

        void UIPlotData_SetYAxis( UIPlotData *aSelf, int aAxis )
        {
            aSelf->mYAxis = static_cast<UIPlotAxis>( aAxis );
        }
#pragma endregion

#pragma region UIFloat64LinePlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIFloat64LinePlot )
        DESTROY_INTERFACE( UIFloat64LinePlot )

        void UIFloat64LinePlot_SetX( UIFloat64LinePlot *aSelf, double *aValue, int aLength )
        {
            aSelf->mX = vector_t<double>( aValue, aValue + aLength );
        }

        void UIFloat64LinePlot_SetY( UIFloat64LinePlot *aSelf, double *aValue, int aLength )
        {
            aSelf->mY = vector_t<double>( aValue, aValue + aLength );
        }
#pragma endregion

#pragma region UIFloat64ScatterPlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIFloat64ScatterPlot )
        DESTROY_INTERFACE( UIFloat64ScatterPlot )

        void UIFloat64ScatterPlot_SetX( UIFloat64ScatterPlot *aSelf, double *aValue, int aLength )
        {
            aSelf->mX = vector_t<double>( aValue, aValue + aLength );
        }

        void UIFloat64ScatterPlot_SetY( UIFloat64ScatterPlot *aSelf, double *aValue, int aLength )
        {
            aSelf->mY = vector_t<double>( aValue, aValue + aLength );
        }
#pragma endregion

#pragma region UIVLinePlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIVLinePlot )
        DESTROY_INTERFACE( UIVLinePlot )

        void UIVLinePlot_SetX( UIVLinePlot *aSelf, double *aValue, int aLength )
        {
            aSelf->mX = vector_t<double>( aValue, aValue + aLength );
        }
#pragma endregion

#pragma region UIHLinePlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIHLinePlot )
        DESTROY_INTERFACE( UIHLinePlot )

        void UIHLinePlot_SetY( UIHLinePlot *aSelf, double *aValue, int aLength )
        {
            aSelf->mY = vector_t<double>( aValue, aValue + aLength );
        }
#pragma endregion

#pragma region UIAxisTag
        CONSTRUCT_WITHOUT_PARAMETERS( UIAxisTag )
        DESTROY_INTERFACE( UIAxisTag )

        void UIAxisTag_SetX( UIAxisTag *aSelf, double aValue )
        {
            aSelf->mX = aValue;
        }

        void UIAxisTag_SetText( UIAxisTag *aSelf, wchar_t *aText )
        {
            aSelf->mText = ConvertStringForCoreclr( aText );
        }

        void UIAxisTag_SetColor( UIAxisTag *aSelf, CLRVec4 aColor )
        {
            aSelf->mColor = vec( aColor );
        }

        CLRVec4 UIAxisTag_GetColor( UIAxisTag *aSelf )
        {
            return vec( aSelf->mColor );
        }

        void UIAxisTag_SetAxis( UIAxisTag *aSelf, int aAxis )
        {
            aSelf->mAxis = static_cast<UIPlotAxis>( aAxis );
        }

        int UIAxisTag_GetAxis( UIAxisTag *aSelf )
        {
            return static_cast<int>( aSelf->mXAxis );
        }
#pragma endregion

#pragma region UIVRangePlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIVRangePlot )
        DESTROY_INTERFACE( UIVRangePlot )

        void UIVRangePlot_SetMin( UIVRangePlot *aSelf, double aValue )
        {
            aSelf->mX0 = aValue;
        }

        double UIVRangePlot_GetMin( UIVRangePlot *aSelf )
        {
            return (double)aSelf->mX0;
        }

        void UIVRangePlot_SetMax( UIVRangePlot *aSelf, double aValue )
        {
            aSelf->mX1 = aValue;
        }

        double UIVRangePlot_GetMax( UIVRangePlot *aSelf )
        {
            return (double)aSelf->mX1;
        }
#pragma endregion

#pragma region UIHRangePlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIHRangePlot )
        DESTROY_INTERFACE( UIHRangePlot )

        void UIHRangePlot_SetMin( UIHRangePlot *aSelf, double aValue )
        {
            aSelf->mY0 = aValue;
        }

        double UIHRangePlot_GetMin( UIHRangePlot *aSelf )
        {
            return (double)aSelf->mY0;
        }

        void UIHRangePlot_SetMax( UIHRangePlot *aSelf, double aValue )
        {
            aSelf->mY1 = aValue;
        }

        double UIHRangePlot_GetMax( UIHRangePlot *aSelf )
        {
            return (double)aSelf->mY1;
        }
#pragma endregion

#pragma region UIProgressBar
        CONSTRUCT_WITHOUT_PARAMETERS( UIProgressBar )
        DESTROY_INTERFACE( UIProgressBar )

        void UIProgressBar_SetProgressValue( UIProgressBar *aSelf, float aValue )
        {
            aSelf->SetProgressValue( aValue );
        }

        void UIProgressBar_SetProgressColor( UIProgressBar *aSelf, CLRVec4 aTextColor )
        {
            aSelf->SetProgressColor( vec( aTextColor ) );
        }

        void UIProgressBar_SetText( UIProgressBar *aSelf, wchar_t *aText )
        {
            aSelf->SetText( ConvertStringForCoreclr( aText ) );
        }

        void UIProgressBar_SetTextColor( UIProgressBar *aSelf, CLRVec4 aTextColor )
        {
            aSelf->SetTextColor( vec( aTextColor ) );
        }

        void UIProgressBar_SetThickness( UIProgressBar *aSelf, float aValue )
        {
            aSelf->SetThickness( aValue );
        }
#pragma endregion

#pragma region UIPropertyValue
        CONSTRUCT_WITHOUT_PARAMETERS( UIPropertyValue )
        DESTROY_INTERFACE( UIPropertyValue )

        void UIPropertyValue_SetText( UIPropertyValue *aSelf, wchar_t *aText )
        {
            //
            aSelf->SetText( ConvertStringForCoreclr( aText ) );
        }

        void UIPropertyValue_SetOrientation( UIPropertyValue *aSelf, eBoxLayoutOrientation aOrientation )
        {
            aSelf->SetOrientation( aOrientation );
        }

        void UIPropertyValue_SetValue( UIPropertyValue *aSelf, wchar_t *aText )
        {
            aSelf->SetValue( ConvertStringForCoreclr( aText ) );
        }

        void UIPropertyValue_SetValueFont( UIPropertyValue *aSelf, FontFamilyFlags aFont )
        {
            aSelf->SetValueFont( aFont );
        }

        void UIPropertyValue_SetNameFont( UIPropertyValue *aSelf, FontFamilyFlags aFont )
        {
            aSelf->SetNameFont( aFont );
        }
#pragma endregion

#pragma region UISlider
        CONSTRUCT_WITHOUT_PARAMETERS( UISlider )
        DESTROY_INTERFACE( UISlider )
#pragma endregion

#pragma region UITableColumn
        void UITableColumn_SetHeader( UITableColumn *aSelf, wchar_t *aHeader )
        {
            aSelf->mHeader = ConvertStringForCoreclr( aHeader );
        }

        void UITableColumn_SetInitialSize( UITableColumn *aSelf, float aValue )
        {
            aSelf->mInitialSize = aValue;
        }

        void UITableColumn_SetTooltip( UITableColumn *aSelf, UIComponent **aTooptip, int aLength )
        {
            aSelf->mToolTip = std::vector( aTooptip, aTooptip + aLength );
        }

        void UITableColumn_SetForegroundColor( UITableColumn *aSelf, CLRVec4 *aForegroundColor, int aLength )
        {
            vector_t<uint32_t> lColors( aLength );
            for( int i = 0; i < aLength; i++ )
                lColors[i] = ImColor( imvec( aForegroundColor[i] ) );

            aSelf->mForegroundColor = lColors;
        }

        void UITableColumn_SetBackgroundColor( UITableColumn *aSelf, CLRVec4 *aBackroundColor, int aLength )
        {
            vector_t<uint32_t> lColors( aLength );
            for( int i = 0; i < aLength; i++ )
                lColors[i] = ImColor( imvec( aBackroundColor[i] ) );

            aSelf->mForegroundColor = lColors;
        }
#pragma endregion

#pragma region UITable
        CONSTRUCT_WITHOUT_PARAMETERS( UITable )
        DESTROY_INTERFACE( UITable )

        void UITable_OnRowClicked( UITable *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )( int );
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnRowClicked( [lInstance, lDelegate]( int i ) { lDelegate( i ); } );
        }

        void UITable_AddColumn( UITable *aSelf, UITableColumn *aColumn )
        {
            aSelf->AddColumn( aColumn );
        }

        void UITable_SetRowHeight( UITable *aSelf, float aRowHeight )
        {
            aSelf->SetRowHeight( aRowHeight );
        }

        void UITable_ClearRowBackgroundColor( UITable *aSelf )
        {
            aSelf->mRowBackgroundColor.clear();
        }

        void UITable_SetRowBackgroundColor( UITable *aSelf, CLRVec4 *aValue, int aLength )
        {
            vector_t<uint32_t> lColors( aLength );
            for( int i = 0; i < aLength; i++ )
                lColors[i] = ImColor( imvec( aValue[i] ) );

            aSelf->mRowBackgroundColor = lColors;
        }

        void UITable_SetDisplayedRowIndices( UITable *aSelf, int *aValue, int aLength )
        {
            if( aValue == nullptr )
                aSelf->mDisplayedRowIndices.reset();
            else
                aSelf->mDisplayedRowIndices = vector_t<int>( aValue, aValue + aLength );
        }
#pragma endregion

#pragma region UIStringColumn
        CONSTRUCT_WITHOUT_PARAMETERS( UIStringColumn )
        DESTROY_INTERFACE( UIStringColumn )

        void UIStringColumn_Clear( UIStringColumn *aSelf )
        {
            aSelf->Clear();
        }

        void UIStringColumn_SetData( UIStringColumn *aSelf, wchar_t **aValue, int aLength )
        {
            aSelf->mData.clear();
            aSelf->mData.resize( aLength );

            for( int i = 0; i < aLength; i++ )
                aSelf->mData[i] = ConvertStringForCoreclr( aValue[i] );
        }
#pragma endregion

#pragma region UITextInput
        CONSTRUCT_WITHOUT_PARAMETERS( UITextInput )
        DESTROY_INTERFACE( UITextInput )

        void UITextInput_SetHintText( UITextInput *aSelf, wchar_t *aText )
        {
            aSelf->SetHintText( ConvertStringForCoreclr( aText ) );
        }

        void UITextInput_SetText( UITextInput *aSelf, wchar_t *aText )
        {
            aSelf->SetText( ConvertStringForCoreclr( aText ) );
        }
        
        wchar_t *UITextInput_GetText( UITextInput *aSelf )
        {
            return CopyCharactersForCoreClr( aSelf->GetText() );
        }

        void UITextInput_SetTextColor( UITextInput *aSelf, CLRVec4 aTextColor )
        {
            aSelf->SetTextColor( vec( aTextColor ) );
        }

        void UITextInput_SetBufferSize( UITextInput *aSelf, uint32_t aBufferSize )
        {
            aSelf->SetBuffersize( aBufferSize );
        }

        void UITextInput_OnTextChanged( UITextInput *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )( wchar_t * );
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnTextChanged( [lInstance, lDelegate]( string_t aString )
                                      { lDelegate( CopyCharactersForCoreClr( aString ) ); } );
        }
#pragma endregion

#pragma region UITextOverlay
        CONSTRUCT_WITHOUT_PARAMETERS( UITextOverlay )
        DESTROY_INTERFACE( UITextOverlay )

        void UITextOverlay_AddText( UITextOverlay *aSelf, wchar_t *aText )
        {
            aSelf->AddText( ConvertStringForCoreclr( aText ) );
        }
        void UITextOverlay_AddBytes( UITextOverlay *aSelf, char *aText, int32_t aOffset, int32_t aCount )
        {
            aSelf->AddText( aText, aOffset, aCount );
        }

        void UITextOverlay_Clear( UITextOverlay *aSelf )
        {
            aSelf->Clear();
        }
#pragma endregion

#pragma region UITextToggleButton
        CONSTRUCT_WITHOUT_PARAMETERS( UITextToggleButton )
        DESTROY_INTERFACE( UITextToggleButton )

        bool UITextToggleButton_IsActive( UITextToggleButton *aSelf )
        {
            return aSelf->IsActive();
        }

        void UITextToggleButton_SetActive( UITextToggleButton *aSelf, bool aValue )
        {
            aSelf->SetActive( aValue );
        }

        void UITextToggleButton_SetActiveColor( UITextToggleButton *aSelf, CLRVec4 aColor )
        {
            aSelf->SetActiveColor( vec( aColor ) );
        }

        void UITextToggleButton_SetInactiveColor( UITextToggleButton *aSelf, CLRVec4 aColor )
        {
            aSelf->SetInactiveColor( vec( aColor ) );
        }

        void UITextToggleButton_OnClicked( UITextToggleButton *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef bool ( *fptr )( bool );
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnClick( [lInstance, lDelegate]( bool i ) { return lDelegate( i ); } );
        }

        void UITextToggleButton_OnChanged( UITextToggleButton *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef bool ( *fptr )();
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnChanged( [lInstance, lDelegate]() { return lDelegate(); } );
        }
#pragma endregion

#pragma region UITreeViewNode
        CONSTRUCT_WITHOUT_PARAMETERS( UITreeViewNode )
        DESTROY_INTERFACE( UITreeViewNode )

        void UITreeViewNode_SetText( UITreeViewNode *aSelf, wchar_t *aText )
        {
            aSelf->SetText( ConvertStringForCoreclr( aText ) );
        }

        void UITreeViewNode_SetTextColor( UITreeViewNode *aSelf, CLRVec4 aTextColor )
        {
            aSelf->SetTextColor( vec( aTextColor ) );
        }

        void UITreeViewNode_SetIcon( UITreeViewNode *aSelf, UIImage *aIcon )
        {
            aSelf->SetIcon( aIcon );
        }

        void UITreeViewNode_SetIndicator( UITreeViewNode *aSelf, UIComponent *aIndicator )
        {
            aSelf->SetIndicator( aIndicator );
        }

        void *UITreeViewNode_Add( UITreeViewNode *aSelf )
        {
            return CAST( void, aSelf->Add() );
        }
#pragma endregion

#pragma region UITreeView
        CONSTRUCT_WITHOUT_PARAMETERS( UITreeView )
        DESTROY_INTERFACE( UITreeView )

        void UITreeView_SetIndent( UITreeView *aSelf, float aIndent )
        {
            aSelf->SetIndent( aIndent );
        }

        void UITreeView_SetIconSpacing( UITreeView *aSelf, float aSpacing )
        {
            aSelf->SetIconSpacing( aSpacing );
        }

        void *UITreeView_Add( UITreeView *aSelf )
        {
            return CAST( void, aSelf->Add() );
        }
#pragma endregion

#pragma region UIVec2Input
        CONSTRUCT_WITHOUT_PARAMETERS( UIVec2Input )
        DESTROY_INTERFACE( UIVec2Input )

        void UIVec2Input_OnChanged( UIVec2Input *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )( CLRVec2 aValue );
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnChanged( [lInstance, lDelegate]( vec4 aVector ) { return lDelegate( CLRVec2{ aVector.x, aVector.y } ); } );
        }

        void UIVec2Input_SetValue( UIVec2Input *aSelf, CLRVec2 aValue )
        {
            aSelf->SetValue( vec( aValue ) );
        }

        CLRVec2 UIVec2Input_GetValue( UIVec2Input *aSelf )
        {
            return vec( aSelf->Value() );
        }

        void UIVec2Input_SetResetValues( UIVec2Input *aSelf, CLRVec2 aValue )
        {
            aSelf->SetResetValues( vec( aValue ) );
        }

        void UIVec2Input_SetFormat( UIVec2Input *aSelf, wchar_t *aText )
        {
            aSelf->SetFormat( ConvertStringForCoreclr( aText ) );
        }
#pragma endregion

#pragma region UIVec3Input
        CONSTRUCT_WITHOUT_PARAMETERS( UIVec3Input )
        DESTROY_INTERFACE( UIVec3Input )

        void UIVec3Input_OnChanged( UIVec3Input *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )( CLRVec3 aValue );
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnChanged(
                [lInstance, lDelegate]( vec4 aVector ) {
                    return lDelegate( CLRVec3{ aVector.x, aVector.y, aVector.z } );
                } );
        }

        void UIVec3Input_SetValue( UIVec3Input *aSelf, CLRVec3 aValue )
        {
            aSelf->SetValue( vec( aValue ) );
        }

        CLRVec3 UIVec3Input_GetValue( UIVec3Input *aSelf )
        {
            return vec( aSelf->Value() );
        }

        void UIVec3Input_SetResetValues( UIVec3Input *aSelf, CLRVec3 aValue )
        {
            aSelf->SetResetValues( vec( aValue ) );
        }

        void UIVec3Input_SetFormat( UIVec3Input *aSelf, wchar_t *aText )
        {
            aSelf->SetFormat( ConvertStringForCoreclr( aText ) );
        }
#pragma endregion

#pragma region UIVec4Input
        CONSTRUCT_WITHOUT_PARAMETERS( UIVec4Input )
        DESTROY_INTERFACE( UIVec4Input )

        void UIVec4Input_OnChanged( UIVec4Input *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )( CLRVec4 aValue );
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnChanged( [lInstance, lDelegate]( vec4 aVector ) { return lDelegate( vec( aVector ) ); } );
        }

        void UIVec4Input_SetValue( UIVec4Input *aSelf, CLRVec4 aValue )
        {
            aSelf->SetValue( vec( aValue ) );
        }

        CLRVec4 UIVec4Input_GetValue( UIVec4Input *aSelf )
        {
            return vec( aSelf->Value() );
        }

        void UIVec4Input_SetResetValues( UIVec4Input *aSelf, CLRVec4 aValue )
        {
            aSelf->SetResetValues( vec( aValue ) );
        }

        void UIVec4Input_SetFormat( UIVec4Input *aSelf, wchar_t *aText )
        {
            aSelf->SetFormat( ConvertStringForCoreclr( aText ) );
        }
#pragma endregion

#pragma region UIWorkspaceDocument
        CONSTRUCT_WITHOUT_PARAMETERS( UIWorkspaceDocument )
        DESTROY_INTERFACE( UIWorkspaceDocument )

        void UIWorkspaceDocument_SetContent( UIWorkspaceDocument *aSelf, UIComponent *aContent )
        {
            aSelf->SetContent( aContent );
        }

        void UIWorkspaceDocument_Update( UIWorkspaceDocument *aSelf )
        {
            aSelf->Update();
        }

        void UIWorkspaceDocument_SetName( UIWorkspaceDocument *aSelf, wchar_t *aName )
        {
            aSelf->mName = ConvertStringForCoreclr( aName );
        }

        bool UIWorkspaceDocument_IsDirty( UIWorkspaceDocument *aSelf )
        {
            return aSelf->mDirty;
        }

        void UIWorkspaceDocument_MarkAsDirty( UIWorkspaceDocument *aSelf, bool aDirty )
        {
            aSelf->mDirty = aDirty;
        }

        void UIWorkspaceDocument_Open( UIWorkspaceDocument *aSelf )
        {
            aSelf->DoOpen();
        }

        void UIWorkspaceDocument_RequestClose( UIWorkspaceDocument *aSelf )
        {
            aSelf->DoQueueClose();
        }

        void UIWorkspaceDocument_ForceClose( UIWorkspaceDocument *aSelf )
        {
            aSelf->DoForceClose();
        }

        void UIWorkspaceDocument_RegisterSaveDelegate( UIWorkspaceDocument *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef bool ( *fptr )();
            fptr lDelegate     = (fptr)aDelegate;
            lInstance->mDoSave = [lInstance, lDelegate]() { return lDelegate(); };
        }
#pragma endregion

#pragma region UIWorkspace
        CONSTRUCT_WITHOUT_PARAMETERS( UIWorkspace )
        DESTROY_INTERFACE( UIWorkspace )

        void UIWorkspace_Add( UIWorkspace *aSelf, UIWorkspaceDocument *aDocument )
        {
            aSelf->Add( aDocument );
        }

        void UIWorkspace_RegisterCloseDocumentDelegate( UIWorkspace *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )( void *aDocs );
            fptr lDelegate               = (fptr)aDelegate;
            lInstance->mOnCloseDocuments = [lInstance, lDelegate]( vector_t<UIWorkspaceDocument *> aDocuments )
            { return lDelegate( aDocuments.data() ); };
        }
#pragma endregion

#pragma region UIBoxLayout
        CONSTRUCT_WITHOUT_PARAMETERS( UIBoxLayout )
        DESTROY_INTERFACE( UIBoxLayout )

        void UIBoxLayout_AddAlignedNonFixed( UIBoxLayout *aSelf, UIComponent *aChild, bool aExpand, bool aFill,
                                             eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
        {
            aSelf->Add( aChild, aExpand, aFill, aHAlignment, aVAlignment );
        }

        void UIBoxLayout_AddNonAlignedNonFixed( UIBoxLayout *aSelf, UIComponent *aChild, bool aExpand, bool aFill )
        {
            aSelf->Add( aChild, aExpand, aFill );
        }

        void UIBoxLayout_AddAlignedFixed( UIBoxLayout *aSelf, UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill,
                                          eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
        {
            aSelf->Add( aChild, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment );
        }

        void UIBoxLayout_AddNonAlignedFixed( UIBoxLayout *aSelf, UIComponent *aChild, float aFixedSize, bool aExpand, bool aFill )
        {
            aSelf->Add( aChild, aFixedSize, aExpand, aFill );
        }

        void UIBoxLayout_AddSeparator( UIBoxLayout *aSelf )
        {
            aSelf->AddSeparator();
        }

        void UIBoxLayout_SetItemSpacing( UIBoxLayout *aSelf, float aItemSpacing )
        {
            aSelf->SetItemSpacing( aItemSpacing );
        }

        void UIBoxLayout_SetOrientation( UIBoxLayout *aSelf, eBoxLayoutOrientation aValue )
        {
            aSelf->SetOrientation( aValue );
        }

        void UIBoxLayout_Clear( UIBoxLayout *aSelf )
        {
            aSelf->Clear();
        }
#pragma endregion

#pragma region UIContainer
        CONSTRUCT_WITHOUT_PARAMETERS( UIContainer )
        DESTROY_INTERFACE( UIContainer )

        void UIContainer_SetContent( UIContainer *aSelf, UIComponent *aChild )
        {
            aSelf->SetContent( aChild );
        }
#pragma endregion

#pragma region UISplitter
        CONSTRUCT_WITHOUT_PARAMETERS( UISplitter )
        DESTROY_INTERFACE( UISplitter )

        void UISplitter_Add1( UISplitter *aSelf, UIComponent *aChild )
        {
            aSelf->Add1( aChild );
        }

        void UISplitter_Add2( UISplitter *aSelf, UIComponent *aChild )
        {
            aSelf->Add2( aChild );
        }

        void UISplitter_SetOrientation( UISplitter *aSelf, eBoxLayoutOrientation aValue )
        {
            aSelf->SetOrientation( aValue );
        }

        void UISplitter_SetItemSpacing( UISplitter *aSelf, float aItemSpacing )
        {
            aSelf->SetItemSpacing( aItemSpacing );
        }
#pragma endregion

#pragma region UIStackLayout
        CONSTRUCT_WITHOUT_PARAMETERS( UIStackLayout )
        DESTROY_INTERFACE( UIStackLayout )

        void UIStackLayout_Add( UIStackLayout *aSelf, UIComponent *aChild, wchar_t *aKey )
        {
            aSelf->Add( aChild, ConvertStringForCoreclr( aKey ) );
        }

        void UIStackLayout_SetCurrent( UIStackLayout *aSelf, wchar_t *aKey )
        {
            aSelf->SetCurrent( ConvertStringForCoreclr( aKey ) );
        }
#pragma endregion

#pragma region UIZLayout
        CONSTRUCT_WITHOUT_PARAMETERS( UIZLayout )
        DESTROY_INTERFACE( UIZLayout )

        void UIZLayout_AddAlignedNonFixed( UIZLayout *aSelf, UIComponent *aChild, bool aExpand, bool aFill,
                                           eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
        {
            aSelf->Add( aChild, aExpand, aFill, aHAlignment, aVAlignment );
        }

        void UIZLayout_AddNonAlignedNonFixed( UIZLayout *aSelf, UIComponent *aChild, bool aExpand, bool aFill )
        {
            aSelf->Add( aChild, aExpand, aFill );
        }

        void UIZLayout_AddAlignedFixed( UIZLayout *aSelf, UIComponent *aChild, CLRVec2 aSize, CLRVec2 aPosition, bool aExpand,
                                        bool aFill, eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
        {
            aSelf->Add( aChild, vec( aSize ), vec( aPosition ), aExpand, aFill, aHAlignment, aVAlignment );
        }

        void UIZLayout_AddNonAlignedFixed( UIZLayout *aSelf, UIComponent *aChild, CLRVec2 aSize, CLRVec2 aPosition, bool aExpand,
                                           bool aFill )
        {
            aSelf->Add( aChild, vec( aSize ), vec( aPosition ), aExpand, aFill );
        }
#pragma endregion

#pragma region UIFileTree
        CONSTRUCT_WITHOUT_PARAMETERS( UIFileTree )
        DESTROY_INTERFACE( UIFileTree )

        void *UIFileTree_Add( UIFileTree *aSelf, wchar_t *aPath )
        {
            return CAST( void, aSelf->Add( ConvertStringForCoreclr( aPath ) ) );
        }

        void UIFileTree_Remove( UIFileTree *aSelf, wchar_t *aPath )
        {
            return aSelf->Remove( ConvertStringForCoreclr( aPath ) );
        }

        void UIFileTree_Update( UIFileTree *aSelf )
        {
            return aSelf->UpdateRows();
        }

        void UIFileTree_OnSelected( UIFileTree *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )( wchar_t *aPath );
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnSelected(
                [lInstance, lDelegate]( path_t const &aPath )
                {
                    auto *lStr = CopyCharactersForCoreClr( aPath.string() );

                    lDelegate( lStr );
                } );
        }

#pragma endregion

#pragma region UIDialog
        CONSTRUCT_WITHOUT_PARAMETERS( UIDialog )
        DESTROY_INTERFACE( UIDialog )

        void UIDialog_SetTitle( UIDialog *aSelf, wchar_t *aTitle )
        {
            aSelf->SetTitle( ConvertStringForCoreclr( aTitle ) );
        }

        void UIDialog_SetSize( UIDialog *aSelf, CLRVec2 aSize )
        {
            aSelf->SetSize( vec( aSize ) );
        }

        void UIDialog_SetContent( UIDialog *aSelf, UIComponent *aContent )
        {
            aSelf->SetContent( aContent );
        }

        void UIDialog_Open( UIDialog *aSelf )
        {
            aSelf->Open();
        }

        void UIDialog_Close( UIDialog *aSelf )
        {
            aSelf->Close();
        }

        void UIDialog_Update( UIDialog *aSelf )
        {
            aSelf->Update();
        }
#pragma endregion

#pragma region UIForm
        CONSTRUCT_WITHOUT_PARAMETERS( UIForm )
        DESTROY_INTERFACE( UIForm )

        void UIForm_SetTitle( UIForm *aSelf, wchar_t *aTitle )
        {
            aSelf->SetTitle( ConvertStringForCoreclr( aTitle ) );
        }

        void UIForm_SetContent( UIForm *aSelf, UIComponent *aContent )
        {
            aSelf->SetContent( aContent );
        }

        void UIForm_Update( UIForm *aSelf )
        {
            aSelf->Update();
        }

        void UIForm_SetSize( UIForm *aSelf, float aWidth, float aHeight )
        {
            aSelf->SetSize( aWidth, aHeight );
        }
#pragma endregion

#pragma region UICodeEditor
        CONSTRUCT_WITHOUT_PARAMETERS( UICodeEditor )
        DESTROY_INTERFACE( UICodeEditor )

        void UICodeEditor_SetText( UICodeEditor *aSelf, wchar_t *aText )
        {
            aSelf->SetText( ConvertStringForCoreclr( aText ) );
        }
        wchar_t *UICodeEditor_GetText( UICodeEditor *aSelf )
        {
            return CopyCharactersForCoreClr( aSelf->GetText() );
        }

        void UICodeEditor_MoveUp( UICodeEditor *aSelf, int aAmount, bool aSelect )
        {
            aSelf->MoveUp( aAmount, aSelect );
        }
        void UICodeEditor_MoveDown( UICodeEditor *aSelf, int aAmount, bool aSelect )
        {
            aSelf->MoveDown( aAmount, aSelect );
        }
        void UICodeEditor_MoveLeft( UICodeEditor *aSelf, int aAmount, bool aSelect, bool aWordMode )
        {
            aSelf->MoveLeft( aAmount, aSelect );
        }
        void UICodeEditor_MoveRight( UICodeEditor *aSelf, int aAmount, bool aSelect, bool aWordMode )
        {
            aSelf->MoveRight( aAmount, aSelect );
        }
        void UICodeEditor_MoveTop( UICodeEditor *aSelf, bool aSelect )
        {
            aSelf->MoveTop( aSelect );
        }
        void UICodeEditor_MoveBottom( UICodeEditor *aSelf, bool aSelect )
        {
            aSelf->MoveBottom( aSelect );
        }
        void UICodeEditor_MoveHome( UICodeEditor *aSelf, bool aSelect )
        {
            aSelf->MoveHome( aSelect );
        }
        void UICodeEditor_MoveEnd( UICodeEditor *aSelf, bool aSelect )
        {
            aSelf->MoveEnd( aSelect );
        }

        struct sCoords
        {
            int mLine;
            int mColumn;
        };

        void UICodeEditor_SetSelectionStart( UICodeEditor *aSelf, sCoords aPosition )
        {
            aSelf->SetSelectionStart( UICodeEditor::Coordinates( aPosition.mLine, aPosition.mColumn ) );
        }

        void UICodeEditor_SetSelectionEnd( UICodeEditor *aSelf, sCoords aPosition )
        {
            aSelf->SetSelectionEnd( UICodeEditor::Coordinates( aPosition.mLine, aPosition.mColumn ) );
        }

        void UICodeEditor_SetSelection( UICodeEditor *aSelf, sCoords aStart, sCoords aEnd, int aMode )
        {
            aSelf->SetSelection( UICodeEditor::Coordinates( aStart.mLine, aStart.mColumn ),
                                 UICodeEditor::Coordinates( aEnd.mLine, aEnd.mColumn ), (UICodeEditor::SelectionMode)aMode );
        }

        void UICodeEditor_SelectWordUnderCursor( UICodeEditor *aSelf )
        {
            aSelf->SelectWordUnderCursor();
        }
        void UICodeEditor_SelectAll( UICodeEditor *aSelf )
        {
            aSelf->SelectAll();
        }
        bool UICodeEditor_HasSelection( UICodeEditor *aSelf )
        {
            return aSelf->HasSelection();
        }
        void UICodeEditor_Cut( UICodeEditor *aSelf )
        {
            return aSelf->Copy();
        }
        void UICodeEditor_Copy( UICodeEditor *aSelf )
        {
            return aSelf->Copy();
        }
        void UICodeEditor_Paste( UICodeEditor *aSelf )
        {
            return aSelf->Paste();
        }
        void UICodeEditor_Delete( UICodeEditor *aSelf )
        {
            return aSelf->Delete();
        }
        bool UICodeEditor_CanUndo( UICodeEditor *aSelf )
        {
            return aSelf->CanUndo();
        }
        bool UICodeEditor_CanRedo( UICodeEditor *aSelf )
        {
            return aSelf->CanRedo();
        }
        void UICodeEditor_Undo( UICodeEditor *aSelf )
        {
            return aSelf->Undo();
        }
        void UICodeEditor_Redo( UICodeEditor *aSelf )
        {
            return aSelf->Redo();
        }

        void UICodeEditor_InsertText( UICodeEditor *aSelf, wchar_t *aText )
        {
            aSelf->InsertText( ConvertStringForCoreclr( aText ) );
        }
        wchar_t *UICodeEditor_GetSelectedText( UICodeEditor *aSelf )
        {
            return CopyCharactersForCoreClr( aSelf->GetSelectedText() );
        }
        wchar_t *UICodeEditor_GetCurrentLineText( UICodeEditor *aSelf )
        {
            return CopyCharactersForCoreClr( aSelf->GetCurrentLineText() );
        }
        bool UICodeEditor_GetReadOnly( UICodeEditor *aSelf )
        {
            return aSelf->IsReadOnly();
        }
        void UICodeEditor_SetReadOnly( UICodeEditor *aSelf, bool aValue )
        {
            aSelf->SetReadOnly( aValue );
        }
        sCoords UICodeEditor_GetCursorPosition( UICodeEditor *aSelf )
        {
            auto const &x = aSelf->GetCursorPosition();

            return sCoords{ x.mLine, x.mColumn };
        }
        void UICodeEditor_SetCursorPosition( UICodeEditor *aSelf, sCoords aValue )
        {
            aSelf->SetCursorPosition( UICodeEditor::Coordinates( aValue.mLine, aValue.mColumn ) );
        }
        bool UICodeEditor_GetShowWhitespace( UICodeEditor *aSelf )
        {
            return aSelf->IsShowingWhitespaces();
        }
        void UICodeEditor_SetShowWhitespace( UICodeEditor *aSelf, bool aValue )
        {
            aSelf->SetShowWhitespaces( aValue );
        }
        int UICodeEditor_GetTabSize( UICodeEditor *aSelf )
        {
            return aSelf->GetTabSize();
        }
        void UICodeEditor_SetTabSize( UICodeEditor *aSelf, int aValue )
        {
            aSelf->SetTabSize( aValue );
        }

#pragma endregion

#pragma region UIMarkdown
        CONSTRUCT_WITHOUT_PARAMETERS( UIMarkdown )
        DESTROY_INTERFACE( UIMarkdown )

        void UIMarkdown_SetText( UIMarkdown *aSelf, wchar_t *aText )
        {
            aSelf->SetText( ConvertStringForCoreclr( aText ) );
        }

        void UIMarkdown_SetTextColor( UIMarkdown *aSelf, CLRVec4 aTextColor )
        {
            aSelf->SetTextColor( vec( aTextColor ) );
        }
#pragma endregion
    }
} // namespace SE::Core::Interop