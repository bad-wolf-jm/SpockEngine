#include "InteropCalls.h"
#include "DotNet/Runtime.h"

namespace SE::Core::Interop
{
    static vec2 vec( CLRVec2 v ) { return vec2{ v.x, v.y }; }
    static vec3 vec( CLRVec3 v ) { return vec3{ v.x, v.y, v.z }; }
    static vec4 vec( CLRVec4 v ) { return vec4{ v.x, v.y, v.z, v.w }; }

    static CLRVec2 vec( ImVec2 v ) { return CLRVec2{ v.x, v.y }; }
    // static vec3 vec( ImVec3 v ) { return vec3{ v.x, v.y, v.z }; }
    static CLRVec4 vec( ImVec4 v ) { return CLRVec4{ v.x, v.y, v.z, v.w }; }

    static CLRVec2 vec( vec2 v ) { return CLRVec2{ v.x, v.y }; }
    static CLRVec3 vec( vec3 v ) { return CLRVec3{ v.x, v.y, v.z }; }
    static CLRVec4 vec( vec4 v ) { return CLRVec4{ v.x, v.y, v.z, v.w }; }

    extern "C"
    {
#define CONSTRUCT_WITHOUT_PARAMETERS( _Ty ) \
    void *_Ty##_Create()                    \
    {                                       \
        auto lNewObject = new _Ty();        \
        return CAST( void, lNewObject );    \
    }
#define DESTROY_INTERFACE( _Ty ) \
    void _Ty##_Destroy( _Ty *aSelf ) { delete aSelf; }
#define CAST( _Ty, v ) static_cast<_Ty *>( v )

#pragma region UIBaseImage
        CONSTRUCT_WITHOUT_PARAMETERS( UIBaseImage )
        void *UIBaseImage_CreateWithPath( void *aText, CLRVec2 aSize )
        {
            auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewImage = new UIBaseImage( lString, vec( aSize ) );

            return CAST( void, lNewImage );
        }

        DESTROY_INTERFACE( UIBaseImage )

        void UIBaseImage_SetImage( UIBaseImage *aSelf, void *aPath )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aPath ) );

            aSelf->SetImage( lString );
        }

        void UIBaseImage_SetSize( UIBaseImage *aSelf, CLRVec2 aSize ) { aSelf->SetSize( vec( aSize ) ); }

        CLRVec2 UIBaseImage_GetSize( UIBaseImage *aSelf ) { return vec( aSelf->Size() ); }

        void UIBaseImage_SetTopLeft( UIBaseImage *aSelf, CLRVec2 aTopLeft ) { aSelf->SetTopLeft( vec2{ aTopLeft.x, aTopLeft.y } ); }

        CLRVec2 UIBaseImage_GetTopLeft( UIBaseImage *aSelf ) { return vec( aSelf->TopLeft() ); }

        void UIBaseImage_SetBottomRight( UIBaseImage *aSelf, CLRVec2 aBottomRight ) { aSelf->SetBottomRight( vec( aBottomRight ) ); }

        CLRVec2 UIBaseImage_GetBottomRight( UIBaseImage *aSelf ) { return vec( aSelf->BottomRight() ); }

        void UIBaseImage_SetTintColor( UIBaseImage *aSelf, CLRVec4 aColor ) { aSelf->SetTintColor( vec( aColor ) ); }

        CLRVec4 UIBaseImage_GetTintColor( UIBaseImage *aSelf ) { return vec( aSelf->TintColor() ); }
#pragma endregion

#pragma region UIButton
        CONSTRUCT_WITHOUT_PARAMETERS( UIButton )
        DESTROY_INTERFACE( UIButton )

        void *UIButton_CreateWithText( void *aText )
        {
            auto lString    = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewButton = new UIButton( lString );

            return CAST( void, lNewButton );
        }

        void UIButton_SetText( UIButton *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetText( lString );
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

        bool UICheckBox_IsChecked( UICheckBox *aSelf ) { return aSelf->IsChecked(); }

        void UICheckBox_SetIsChecked( UICheckBox *aSelf, bool aValue ) { aSelf->SetIsChecked( aValue ); }
#pragma endregion

#pragma region UIColorButton
        CONSTRUCT_WITHOUT_PARAMETERS( UIColorButton )
        DESTROY_INTERFACE( UIColorButton )
#pragma endregion

#pragma region UIComboBox
        CONSTRUCT_WITHOUT_PARAMETERS( UIComboBox )
        DESTROY_INTERFACE( UIComboBox )

        void *UIComboBox_CreateWithItems( void *aItems )
        {
            std::vector<std::string> lItemVector;
            for( auto const &x : DotNetRuntime::AsVector<MonoString *>( CAST( MonoObject, aItems ) ) )
                lItemVector.emplace_back( DotNetRuntime::NewString( x ) );

            auto lNewComboBox = new UIComboBox( lItemVector );

            return CAST( void, lNewComboBox );
        }

        int UIComboBox_GetCurrent( UIComboBox *aSelf ) { return aSelf->Current(); }

        void UIComboBox_SetCurrent( UIComboBox *aSelf, int aValue ) { aSelf->SetCurrent( aValue ); }

        void UIComboBox_SetItemList( UIComboBox *aSelf, void *aItems )
        {
            std::vector<std::string> lItemVector;
            for( auto const &x : DotNetRuntime::AsVector<MonoString *>( CAST( MonoObject, aItems ) ) )
                lItemVector.emplace_back( DotNetRuntime::NewString( x ) );

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
        void UIComponent_SetIsVisible( UIComponent *aSelf, bool aIsVisible ) { aSelf->mIsVisible = aIsVisible; }

        void UIComponent_SetIsEnabled( UIComponent *aSelf, bool aIsEnabled ) { aSelf->mIsEnabled = aIsEnabled; }

        void UIComponent_SetAllowDragDrop( UIComponent *aSelf, bool aAllowDragDrop ) { aSelf->mAllowDragDrop = aAllowDragDrop; }

        void UIComponent_SetPaddingAll( UIComponent *aSelf, float aPaddingAll ) { aSelf->SetPadding( aPaddingAll ); }

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

        void UIComponent_SetBackgroundColor( UIComponent *aSelf, CLRVec4 aColor ) { aSelf->SetBackgroundColor( vec( aColor ) ); }

        void UIComponent_SetFont( UIComponent *aSelf, FontFamilyFlags aFont ) { aSelf->SetFont( aFont ); }

        void UIComponent_SetTooltip( UIComponent *aSelf, void *aTooltip )
        {
            auto lTooltip = CAST( UIComponent, aTooltip );

            aSelf->SetTooltip( lTooltip );
        }
#pragma endregion

#pragma region UIDropdownButton
        CONSTRUCT_WITHOUT_PARAMETERS( UIDropdownButton )
        DESTROY_INTERFACE( UIDropdownButton )

        void UIDropdownButton_SetContent( UIDropdownButton *aSelf, void *aContent )
        {
            auto lContent = CAST( UIComponent, aContent );

            return aSelf->SetContent( lContent );
        }

        void UIDropdownButton_SetContentSize( UIDropdownButton *aSelf, CLRVec2 aContentSizse )
        {
            return aSelf->SetContentSize( vec( aContentSizse ) );
        }

        void UIDropdownButton_SetImage( UIDropdownButton *aSelf, void *aImage )
        {
            auto lImage = CAST( UIBaseImage, aImage );

            aSelf->SetImage( lImage );
        }

        void UIDropdownButton_SetText( UIDropdownButton *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetText( lString );
        }

        void UIDropdownButton_SetTextColor( UIDropdownButton *aSelf, CLRVec4 aColor ) { aSelf->SetTextColor( vec( aColor ) ); }
#pragma endregion

#pragma region UIImage
        CONSTRUCT_WITHOUT_PARAMETERS( UIImage )
        DESTROY_INTERFACE( UIImage )

        void *UIImage_CreateWithPath( void *aText, CLRVec2 aSize )
        {
            auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewImage = new UIImage( lString, vec( aSize ) );

            return CAST( void, lNewImage );
        }
#pragma endregion

#pragma region UIImageButton
        CONSTRUCT_WITHOUT_PARAMETERS( UIImageButton )
        DESTROY_INTERFACE( UIImageButton )

        void *UIImageButton_CreateWithPath( char *aText, CLRVec2 aSize )
        {
            // auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewImage = new UIImageButton( std::string(aText), vec( aSize ) );

            return CAST( void, lNewImage );
        }

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

        bool UIImageToggleButton_IsActive( UIImageToggleButton *aSelf ) { return aSelf->IsActive(); }

        void UIImageToggleButton_SetActive( UIImageToggleButton *aSelf, bool aValue ) { aSelf->SetActive( aValue ); }

        void UIImageToggleButton_SetActiveImage( UIImageToggleButton *aSelf, void *aImage )
        {
            auto lImage = CAST( UIBaseImage, aImage );

            aSelf->SetActiveImage( lImage );
        }

        void UIImageToggleButton_SetInactiveImage( UIImageToggleButton *aSelf, void *aImage )
        {
            auto lImage = CAST( UIBaseImage, aImage );

            aSelf->SetInactiveImage( lImage );
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

        void *UILabel_CreateWithText( void *aText )
        {
            auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewLabel = new UILabel( lString );

            return CAST( void, lNewLabel );
        }

        void UILabel_SetText( UILabel *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetText( lString );
        }

        void UILabel_SetTextColor( UILabel *aSelf, CLRVec4 aTextColor ) { aSelf->SetTextColor( vec( aTextColor ) ); }
#pragma endregion

#pragma region UIMenuItem
        CONSTRUCT_WITHOUT_PARAMETERS( UIMenuItem )
        DESTROY_INTERFACE( UIMenuItem )

        void *UIMenuItem_CreateWithText( void *aText )
        {
            auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewLabel = new UIMenuItem( lString );

            return CAST( void, lNewLabel );
        }

        void *UIMenuItem_CreateWithTextAndShortcut( void *aText, void *aShortcut )
        {
            auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lShortcut = DotNetRuntime::NewString( CAST( MonoString, aShortcut ) );
            auto lNewLabel = new UIMenuItem( lString, lShortcut );

            return CAST( void, lNewLabel );
        }

        void UIMenuItem_SetText( UIMenuItem *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetText( lString );
        }

        void UIMenuItem_SetShortcut( UIMenuItem *aSelf, void *aShortcut )
        {
            auto lShortcut = DotNetRuntime::NewString( CAST( MonoString, aShortcut ) );

            aSelf->SetShortcut( lShortcut );
        }

        void UIMenuItem_SetTextColor( UIMenuItem *aSelf, CLRVec4 aTextColor ) { aSelf->SetTextColor( vec( aTextColor ) ); }

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

        void *UIMenu_CreateWithText( void *aText )
        {
            auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewLabel = new UIMenu( lString );

            return CAST( void, lNewLabel );
        }

        void *UIMenu_AddAction( UIMenu *aSelf, void *aText, void *aShortcut )
        {
            auto lString    = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lShortcut  = DotNetRuntime::NewString( CAST( MonoString, aShortcut ) );
            auto lNewAction = aSelf->AddActionRaw( lString, lShortcut );

            return CAST( void, lNewAction );
        }

        void *UIMenu_AddMenu( UIMenu *aSelf, void *aText )
        {
            auto lString  = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewMenu = aSelf->AddMenuRaw( lString );

            return CAST( void, lNewMenu );
        }

        void *UIMenu_AddSeparator( UIMenu *aSelf )
        {
            auto lNewSeparator = aSelf->AddSeparatorRaw();

            return CAST( void, lNewSeparator );
        }

        void UIMenu_Update( UIMenu *aSelf ) { aSelf->Update(); }
#pragma endregion

#pragma region UIPlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIPlot )
        DESTROY_INTERFACE( UIPlot )

        void UIPlot_Clear( UIPlot *aSelf ) { aSelf->Clear(); }

        void UIPlot_ConfigureLegend( UIPlot *aSelf, CLRVec2 aLegendPadding, CLRVec2 aLegendInnerPadding, CLRVec2 aLegendSpacing )
        {
            aSelf->ConfigureLegend( vec( aLegendPadding ), vec( aLegendInnerPadding ), vec( aLegendSpacing ) );
        }

        void UIPlot_Add( UIPlot *aSelf, UIPlotData *aPlot ) { aSelf->Add( aPlot ); }

        void UIPlot_SetAxisLimits( UIPlot *aSelf, int aAxis, double aMin, double aMax )
        {
            auto lSelf = aSelf;

            lSelf->mAxisConfiguration[aAxis].mSetLimitRequest = true;
            lSelf->mAxisConfiguration[aAxis].mMin             = static_cast<float>( aMin );
            lSelf->mAxisConfiguration[aAxis].mMax             = static_cast<float>( aMax );
        }

        void UIPlot_SetAxisTitle( UIPlot *aSelf, int aAxis, void *aTitle )
        {
            aSelf->mAxisConfiguration[aAxis].mTitle = DotNetRuntime::NewString( CAST( MonoString, aTitle ) );
        }

        void *UIPlot_GetAxisTitle( UIPlot *aSelf, int aAxis )
        {
            return DotNetRuntime::NewString( aSelf->mAxisConfiguration[aAxis].mTitle );
        }
#pragma endregion

#pragma region UIPlotData
        void UIPlotData_SetLegend( UIPlotData *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->mLegend = lString;
        }

        void UIPlotData_SetThickness( UIPlotData *aSelf, float aThickness ) { aSelf->mThickness = aThickness; }

        void UIPlotData_SetColor( UIPlotData *aSelf, CLRVec4 aColor ) { aSelf->mColor = vec( aColor ); }

        void UIPlotData_SetXAxis( UIPlotData *aSelf, int aAxis ) { aSelf->mXAxis = static_cast<UIPlotAxis>( aAxis ); }

        void UIPlotData_SetYAxis( UIPlotData *aSelf, int aAxis ) { aSelf->mYAxis = static_cast<UIPlotAxis>( aAxis ); }
#pragma endregion

#pragma region UIFloat64LinePlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIFloat64LinePlot )
        DESTROY_INTERFACE( UIFloat64LinePlot )

        void UIFloat64LinePlot_SetX( UIFloat64LinePlot *aSelf, void *aValue )
        {
            aSelf->mX = DotNetRuntime::AsVector<double>( CAST( MonoObject, aValue ) );
        }

        void UIFloat64LinePlot_SetY( UIFloat64LinePlot *aSelf, void *aValue )
        {
            aSelf->mY = DotNetRuntime::AsVector<double>( CAST( MonoObject, aValue ) );
        }
#pragma endregion

#pragma region UIFloat64ScatterPlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIFloat64ScatterPlot )
        DESTROY_INTERFACE( UIFloat64ScatterPlot )

        void UIFloat64ScatterPlot_SetX( UIFloat64ScatterPlot *aSelf, void *aValue )
        {
            aSelf->mX = DotNetRuntime::AsVector<double>( CAST( MonoObject, aValue ) );
        }

        void UIFloat64ScatterPlot_SetY( UIFloat64ScatterPlot *aSelf, void *aValue )
        {
            aSelf->mY = DotNetRuntime::AsVector<double>( CAST( MonoObject, aValue ) );
        }
#pragma endregion

#pragma region UIVLinePlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIVLinePlot )
        DESTROY_INTERFACE( UIVLinePlot )

        void UIVLinePlot_SetX( UIVLinePlot *aSelf, void *aValue )
        {
            aSelf->mX = DotNetRuntime::AsVector<double>( CAST( MonoObject, aValue ) );
        }
#pragma endregion

#pragma region UIHLinePlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIHLinePlot )
        DESTROY_INTERFACE( UIHLinePlot )

        void UIHLinePlot_SetY( UIHLinePlot *aSelf, void *aValue )
        {
            aSelf->mY = DotNetRuntime::AsVector<double>( CAST( MonoObject, aValue ) );
        }
#pragma endregion

#pragma region UIAxisTag
        CONSTRUCT_WITHOUT_PARAMETERS( UIAxisTag )
        DESTROY_INTERFACE( UIAxisTag )

        void *UIAxisTag_CreateWithTextAndColor( UIPlotAxis aAxis, double aX, void *aText, CLRVec4 aColor )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            auto lSelf = new UIAxisTag( aAxis, aX, lString, vec( aColor ) );

            return CAST( void, lSelf );
        }

        void UIAxisTag_SetX( UIAxisTag *aSelf, double aValue ) { aSelf->mX = aValue; }

        void UIAxisTag_SetText( UIAxisTag *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->mText = lString;
        }

        void UIAxisTag_SetColor( UIAxisTag *aSelf, CLRVec4 aColor ) { aSelf->mColor = vec( aColor ); }

        CLRVec4 UIAxisTag_GetColor( UIAxisTag *aSelf ) { return vec( aSelf->mColor ); }

        void UIAxisTag_SetAxis( UIAxisTag *aSelf, int aAxis ) { aSelf->mAxis = static_cast<UIPlotAxis>( aAxis ); }

        int UIAxisTag_GetAxis( UIAxisTag *aSelf ) { return static_cast<int>( aSelf->mXAxis ); }
#pragma endregion

#pragma region UIVRangePlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIVRangePlot )
        DESTROY_INTERFACE( UIVRangePlot )

        void UIVRangePlot_SetMin( UIVRangePlot *aSelf, double aValue ) { aSelf->mX0 = aValue; }

        double UIVRangePlot_GetMin( UIVRangePlot *aSelf ) { return (double)aSelf->mX0; }

        void UIVRangePlot_SetMax( UIVRangePlot *aSelf, double aValue ) { aSelf->mX1 = aValue; }

        double UIVRangePlot_GetMax( UIVRangePlot *aSelf ) { return (double)aSelf->mX1; }
#pragma endregion

#pragma region UIHRangePlot
        CONSTRUCT_WITHOUT_PARAMETERS( UIHRangePlot )
        DESTROY_INTERFACE( UIHRangePlot )

        void UIHRangePlot_SetMin( UIHRangePlot *aSelf, double aValue ) { aSelf->mY0 = aValue; }

        double UIHRangePlot_GetMin( UIHRangePlot *aSelf ) { return (double)aSelf->mY0; }

        void UIHRangePlot_SetMax( UIHRangePlot *aSelf, double aValue ) { aSelf->mY1 = aValue; }

        double UIHRangePlot_GetMax( UIHRangePlot *aSelf ) { return (double)aSelf->mY1; }
#pragma endregion

#pragma region UIProgressBar
        CONSTRUCT_WITHOUT_PARAMETERS( UIProgressBar )
        DESTROY_INTERFACE( UIProgressBar )

        void UIProgressBar_SetProgressValue( UIProgressBar *aSelf, float aValue ) { aSelf->SetProgressValue( aValue ); }

        void UIProgressBar_SetProgressColor( UIProgressBar *aSelf, CLRVec4 aTextColor )
        {
            aSelf->SetProgressColor( vec( aTextColor ) );
        }

        void UIProgressBar_SetText( UIProgressBar *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetText( lString );
        }

        void UIProgressBar_SetTextColor( UIProgressBar *aSelf, CLRVec4 aTextColor ) { aSelf->SetTextColor( vec( aTextColor ) ); }

        void UIProgressBar_SetThickness( UIProgressBar *aSelf, float aValue ) { aSelf->SetThickness( aValue ); }
#pragma endregion

#pragma region UIPropertyValue
        CONSTRUCT_WITHOUT_PARAMETERS( UIPropertyValue )
        DESTROY_INTERFACE( UIPropertyValue )

        void *UIPropertyValue_CreateWithText( void *aText )
        {
            auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewLabel = new UIPropertyValue( lString );

            return CAST( void, lNewLabel );
        }

        void *UIPropertyValue_CreateWithTextAndOrientation( void *aText, eBoxLayoutOrientation aOrientation )
        {
            auto lString   = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewLabel = new UIPropertyValue( lString, aOrientation );

            return CAST( void, lNewLabel );
        }

        void UIPropertyValue_SetValue( UIPropertyValue *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetValue( lString );
        }

        void UIPropertyValue_SetValueFont( UIPropertyValue *aSelf, FontFamilyFlags aFont ) { aSelf->SetValueFont( aFont ); }

        void UIPropertyValue_SetNameFont( UIPropertyValue *aSelf, FontFamilyFlags aFont ) { aSelf->SetNameFont( aFont ); }
#pragma endregion

#pragma region UISlider
        CONSTRUCT_WITHOUT_PARAMETERS( UISlider )
        DESTROY_INTERFACE( UISlider )
#pragma endregion

#pragma region UITableColumn
        void UITableColumn_SetTooltip( UITableColumn *aSelf, void *aTooptip )
        {
            auto lSelf = aSelf;

            lSelf->mToolTip.clear();
            for( auto const &x : DotNetRuntime::AsVector<UIComponent *>( CAST( MonoObject, aTooptip ) ) )
                lSelf->mToolTip.push_back( x );
        }

        void UITableColumn_SetForegroundColor( UITableColumn *aSelf, void *aForegroundColor )
        {
            auto lSelf = aSelf;

            lSelf->mForegroundColor.clear();
            for( auto const &x : DotNetRuntime::AsVector<ImVec4>( CAST( MonoObject, aForegroundColor ) ) )
                lSelf->mForegroundColor.push_back( ImColor( x ) );
        }

        void UITableColumn_SetBackgroundColor( UITableColumn *aSelf, void *aBackroundColor )
        {
            auto lSelf = aSelf;

            lSelf->mBackgroundColor.clear();
            for( auto const &x : DotNetRuntime::AsVector<ImVec4>( CAST( MonoObject, aBackroundColor ) ) )
                lSelf->mBackgroundColor.push_back( ImColor( x ) );
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

        void UITable_AddColumn( UITable *aSelf, void *aColumn ) { aSelf->AddColumn( CAST( UITableColumn, aColumn ) ); }

        void UITable_SetRowHeight( UITable *aSelf, float aRowHeight ) { aSelf->SetRowHeight( aRowHeight ); }

        void UITable_ClearRowBackgroundColor( UITable *aSelf ) { aSelf->mRowBackgroundColor.clear(); }

        void UITable_SetRowBackgroundColor( UITable *aSelf, void *aValue )
        {
            auto lSelf = aSelf;

            lSelf->mRowBackgroundColor.clear();
            for( auto &x : DotNetRuntime::AsVector<ImVec4>( CAST( MonoObject, aValue ) ) )
                lSelf->mRowBackgroundColor.push_back( ImColor( x ) );
        }

        void UITable_SetDisplayedRowIndices( UITable *aSelf, void *aValue )
        {
            auto lSelf = aSelf;
            if( aValue == nullptr )
                lSelf->mDisplayedRowIndices.reset();
            else
                lSelf->mDisplayedRowIndices = DotNetRuntime::AsVector<int>( CAST( MonoObject, aValue ) );
        }
#pragma endregion

#pragma region UIFloat64Column
        CONSTRUCT_WITHOUT_PARAMETERS( UIFloat64Column )
        DESTROY_INTERFACE( UIFloat64Column )

        void *UIFloat64Column_CreateFull( void *aHeader, float aInitialSize, void *aFormat, void *aNaNFormat )
        {
            auto lHeader    = DotNetRuntime::NewString( CAST( MonoString, aHeader ) );
            auto lFormat    = DotNetRuntime::NewString( CAST( MonoString, aFormat ) );
            auto lNaNFormat = DotNetRuntime::NewString( CAST( MonoString, aNaNFormat ) );
            auto lNewColumn = new UIFloat64Column( lHeader, aInitialSize, lFormat, lNaNFormat );

            return CAST( void, lNewColumn );
        }

        void UIFloat64Column_Clear( UIFloat64Column *aSelf ) { aSelf->Clear(); }

        void UIFloat64Column_SetData( UIFloat64Column *aSelf, void *aValue )
        {
            aSelf->mData = DotNetRuntime::AsVector<double>( CAST( MonoObject, aValue ) );
        }
#pragma endregion

#pragma region UIUint32Column
        CONSTRUCT_WITHOUT_PARAMETERS( UIUint32Column )
        DESTROY_INTERFACE( UIUint32Column )

        void *UIUint32Column_CreateFull( void *aHeader, float aInitialSize )
        {
            auto lHeader    = DotNetRuntime::NewString( CAST( MonoString, aHeader ) );
            auto lNewColumn = new UIUint32Column( lHeader, aInitialSize );

            return CAST( void, lNewColumn );
        }

        void UIUint32Column_Clear( UIUint32Column *aSelf ) { aSelf->Clear(); }

        void UIUint32Column_SetData( UIUint32Column *aSelf, void *aValue )
        {
            aSelf->mData = DotNetRuntime::AsVector<uint32_t>( CAST( MonoObject, aValue ) );
        }
#pragma endregion

#pragma region UIStringColumn
        CONSTRUCT_WITHOUT_PARAMETERS( UIStringColumn )
        DESTROY_INTERFACE( UIStringColumn )

        void *UIStringColumn_CreateFull( void *aHeader, float aInitialSize )
        {
            auto lHeader    = DotNetRuntime::NewString( CAST( MonoString, aHeader ) );
            auto lNewColumn = new UIStringColumn( lHeader, aInitialSize );

            return CAST( void, lNewColumn );
        }

        void UIStringColumn_Clear( UIStringColumn *aSelf ) { aSelf->Clear(); }

        void UIStringColumn_SetData( UIStringColumn *aSelf, void *aValue )
        {
            auto lSelf = aSelf;

            lSelf->mData.clear();
            for( auto const &x : DotNetRuntime::AsVector<MonoString *>( CAST( MonoObject, aValue ) ) )
                lSelf->mData.push_back( DotNetRuntime::NewString( x ) );
        }
#pragma endregion

#pragma region UITextInput
        CONSTRUCT_WITHOUT_PARAMETERS( UITextInput )
        DESTROY_INTERFACE( UITextInput )

        void *UITextInput_CreateWithText( void *aText )
        {
            auto lString       = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewTextInput = new UITextInput( lString );

            return CAST( void, lNewTextInput );
        }

        void UITextInput_SetHintText( UITextInput *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetHintText( lString );
        }

        void *UITextInput_GetText( UITextInput *aSelf ) { return DotNetRuntime::NewString( aSelf->GetText() ); }

        void UITextInput_SetTextColor( UITextInput *aSelf, CLRVec4 aTextColor ) { aSelf->SetTextColor( vec( aTextColor ) ); }

        void UITextInput_SetBufferSize( UITextInput *aSelf, uint32_t aBufferSize ) { aSelf->SetBuffersize( aBufferSize ); }

        void UITextInput_OnTextChanged( UITextInput *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )( char * );
            fptr lDelegate = (fptr)aDelegate;
            lInstance->OnTextChanged( [lInstance, lDelegate]( std::string aString ) { lDelegate( aString.data() ); } );
        }
#pragma endregion

#pragma region UITextOverlay
        CONSTRUCT_WITHOUT_PARAMETERS( UITextOverlay )
        DESTROY_INTERFACE( UITextOverlay )

        void UITextOverlay_AddText( UITextOverlay *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->AddText( lString );
        }

        void UITextOverlay_Clear( UITextOverlay *aSelf ) { aSelf->Clear(); }
#pragma endregion

#pragma region UITextToggleButton
        CONSTRUCT_WITHOUT_PARAMETERS( UITextToggleButton )
        DESTROY_INTERFACE( UITextToggleButton )

        void *UITextToggleButton_CreateWithText( void *aText )
        {
            auto lString    = DotNetRuntime::NewString( CAST( MonoString, aText ) );
            auto lNewButton = new UITextToggleButton( lString );

            return CAST( void, lNewButton );
        }

        bool UITextToggleButton_IsActive( UITextToggleButton *aSelf ) { return aSelf->IsActive(); }

        void UITextToggleButton_SetActive( UITextToggleButton *aSelf, bool aValue ) { aSelf->SetActive( aValue ); }

        void UITextToggleButton_SetActiveColor( UITextToggleButton *aSelf, CLRVec4 aColor ) { aSelf->SetActiveColor( vec( aColor ) ); }

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

        void UITreeViewNode_SetText( UITreeViewNode *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetText( lString );
        }

        void UITreeViewNode_SetTextColor( UITreeViewNode *aSelf, CLRVec4 aTextColor ) { aSelf->SetTextColor( vec( aTextColor ) ); }

        void UITreeViewNode_SetIcon( UITreeViewNode *aSelf, void *aIcon )
        {
            auto lImage = CAST( UIImage, aIcon );

            aSelf->SetIcon( lImage );
        }

        void UITreeViewNode_SetIndicator( UITreeViewNode *aSelf, void *aIndicator )
        {
            auto lImage = CAST( UIComponent, aIndicator );

            aSelf->SetIndicator( lImage );
        }

        void *UITreeViewNode_Add( UITreeViewNode *aSelf ) { return CAST( void, aSelf->Add() ); }
#pragma endregion

#pragma region UITreeView
        CONSTRUCT_WITHOUT_PARAMETERS( UITreeView )
        DESTROY_INTERFACE( UITreeView )

        void UITreeView_SetIndent( UITreeView *aSelf, float aIndent ) { aSelf->SetIndent( aIndent ); }

        void UITreeView_SetIconSpacing( UITreeView *aSelf, float aSpacing ) { aSelf->SetIconSpacing( aSpacing ); }

        void *UITreeView_Add( UITreeView *aSelf ) { return CAST( void, aSelf->Add() ); }
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

        void UIVec2Input_SetValue( UIVec2Input *aSelf, CLRVec2 aValue ) { aSelf->SetValue( vec( aValue ) ); }

        CLRVec2 UIVec2Input_GetValue( UIVec2Input *aSelf ) { return vec( aSelf->Value() ); }

        void UIVec2Input_SetResetValues( UIVec2Input *aSelf, CLRVec2 aValue ) { aSelf->SetResetValues( vec( aValue ) ); }

        void UIVec2Input_SetFormat( UIVec2Input *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetFormat( lString );
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

        void UIVec3Input_SetValue( UIVec3Input *aSelf, CLRVec3 aValue ) { aSelf->SetValue( vec( aValue ) ); }

        CLRVec3 UIVec3Input_GetValue( UIVec3Input *aSelf ) { return vec( aSelf->Value() ); }

        void UIVec3Input_SetResetValues( UIVec3Input *aSelf, CLRVec3 aValue ) { aSelf->SetResetValues( vec( aValue ) ); }

        void UIVec3Input_SetFormat( UIVec3Input *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetFormat( lString );
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

        void UIVec4Input_SetValue( UIVec4Input *aSelf, CLRVec4 aValue ) { aSelf->SetValue( vec( aValue ) ); }

        CLRVec4 UIVec4Input_GetValue( UIVec4Input *aSelf ) { return vec( aSelf->Value() ); }

        void UIVec4Input_SetResetValues( UIVec4Input *aSelf, CLRVec4 aValue ) { aSelf->SetResetValues( vec( aValue ) ); }

        void UIVec4Input_SetFormat( UIVec4Input *aSelf, void *aText )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aText ) );

            aSelf->SetFormat( lString );
        }
#pragma endregion

#pragma region UIWorkspaceDocument
        CONSTRUCT_WITHOUT_PARAMETERS( UIWorkspaceDocument )
        DESTROY_INTERFACE( UIWorkspaceDocument )

        void UIWorkspaceDocument_SetContent( UIWorkspaceDocument *aSelf, void *aContent )
        {
            auto lContent = CAST( UIComponent, aContent );

            aSelf->SetContent( lContent );
        }

        void UIWorkspaceDocument_Update( UIWorkspaceDocument *aSelf ) { aSelf->Update(); }

        void UIWorkspaceDocument_SetName( UIWorkspaceDocument *aSelf, void *aName )
        {
            auto lName = DotNetRuntime::NewString( CAST( MonoString, aName ) );

            aSelf->mName = lName;
        }

        bool UIWorkspaceDocument_IsDirty( UIWorkspaceDocument *aSelf ) { return aSelf->mDirty; }

        void UIWorkspaceDocument_MarkAsDirty( UIWorkspaceDocument *aSelf, bool aDirty ) { aSelf->mDirty = aDirty; }

        void UIWorkspaceDocument_Open( UIWorkspaceDocument *aSelf ) { aSelf->DoOpen(); }

        void UIWorkspaceDocument_RequestClose( UIWorkspaceDocument *aSelf ) { aSelf->DoQueueClose(); }

        void UIWorkspaceDocument_ForceClose( UIWorkspaceDocument *aSelf ) { aSelf->DoForceClose(); }

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

        void UIWorkspace_Add( UIWorkspace *aSelf, void *aDocument )
        {
            auto lDocument = CAST( UIWorkspaceDocument, aDocument );

            aSelf->Add( lDocument );
        }

        void UIWorkspace_RegisterCloseDocumentDelegate( UIWorkspace *aSelf, void *aDelegate )
        {
            auto lInstance = aSelf;

            typedef void ( *fptr )( void *aDocs );
            fptr lDelegate               = (fptr)aDelegate;
            lInstance->mOnCloseDocuments = [lInstance, lDelegate]( std::vector<UIWorkspaceDocument *> aDocuments )
            { return lDelegate( aDocuments.data() ); };
        }
#pragma endregion

#pragma region UIBoxLayout
        void *UIBoxLayout_CreateWithOrientation( eBoxLayoutOrientation aOrientation )
        {
            auto lNewLayout = new UIBoxLayout( aOrientation );

            return CAST( void, lNewLayout );
        }

        DESTROY_INTERFACE( UIBoxLayout )

        void UIBoxLayout_AddAlignedNonFixed( UIBoxLayout *aSelf, void *aChild, bool aExpand, bool aFill,
                                             eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->Add( lChild, aExpand, aFill, aHAlignment, aVAlignment );
        }

        void UIBoxLayout_AddNonAlignedNonFixed( UIBoxLayout *aSelf, void *aChild, bool aExpand, bool aFill )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->Add( lChild, aExpand, aFill );
        }

        void UIBoxLayout_AddAlignedFixed( UIBoxLayout *aSelf, void *aChild, float aFixedSize, bool aExpand, bool aFill,
                                          eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->Add( lChild, aFixedSize, aExpand, aFill, aHAlignment, aVAlignment );
        }

        void UIBoxLayout_AddNonAlignedFixed( UIBoxLayout *aSelf, void *aChild, float aFixedSize, bool aExpand, bool aFill )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->Add( lChild, aFixedSize, aExpand, aFill );
        }

        void UIBoxLayout_AddSeparator( UIBoxLayout *aSelf ) { aSelf->AddSeparator(); }

        void UIBoxLayout_SetItemSpacing( UIBoxLayout *aSelf, float aItemSpacing )
        {
            auto lInstance = aSelf;

            lInstance->SetItemSpacing( aItemSpacing );
        }

        void UIBoxLayout_Clear( UIBoxLayout *aSelf ) { aSelf->Clear(); }
#pragma endregion

#pragma region UIContainer
        CONSTRUCT_WITHOUT_PARAMETERS( UIContainer )
        DESTROY_INTERFACE( UIContainer )

        void UIContainer_SetContent( UIContainer *aSelf, void *aChild )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->SetContent( lChild );
        }
#pragma endregion

#pragma region UISplitter
        CONSTRUCT_WITHOUT_PARAMETERS( UISplitter )
        DESTROY_INTERFACE( UISplitter )

        void *UISplitter_CreateWithOrientation( eBoxLayoutOrientation aOrientation )
        {
            auto lNewLayout = new UISplitter( aOrientation );

            return CAST( void, lNewLayout );
        }

        void UISplitter_Add1( UISplitter *aSelf, void *aChild )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->Add1( lChild );
        }

        void UISplitter_Add2( UISplitter *aSelf, void *aChild )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->Add2( lChild );
        }

        void UISplitter_SetItemSpacing( UISplitter *aSelf, float aItemSpacing ) { aSelf->SetItemSpacing( aItemSpacing ); }
#pragma endregion

#pragma region UIStackLayout
        CONSTRUCT_WITHOUT_PARAMETERS( UIStackLayout )
        DESTROY_INTERFACE( UIStackLayout )

        void UIStackLayout_Add( UIStackLayout *aSelf, void *aChild, void *aKey )
        {
            auto lChild  = CAST( UIComponent, aChild );
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aKey ) );

            aSelf->Add( lChild, lString );
        }

        void UIStackLayout_SetCurrent( UIStackLayout *aSelf, void *aKey )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aKey ) );

            aSelf->SetCurrent( lString );
        }
#pragma endregion

#pragma region UIZLayout
        CONSTRUCT_WITHOUT_PARAMETERS( UIZLayout )
        DESTROY_INTERFACE( UIZLayout )

        void UIZLayout_AddAlignedNonFixed( UIZLayout *aSelf, void *aChild, bool aExpand, bool aFill, eHorizontalAlignment aHAlignment,
                                           eVerticalAlignment aVAlignment )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->Add( lChild, aExpand, aFill, aHAlignment, aVAlignment );
        }

        void UIZLayout_AddNonAlignedNonFixed( UIZLayout *aSelf, void *aChild, bool aExpand, bool aFill )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->Add( lChild, aExpand, aFill );
        }

        void UIZLayout_AddAlignedFixed( UIZLayout *aSelf, void *aChild, CLRVec2 aSize, CLRVec2 aPosition, bool aExpand, bool aFill,
                                        eHorizontalAlignment aHAlignment, eVerticalAlignment aVAlignment )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->Add( lChild, vec( aSize ), vec( aPosition ), aExpand, aFill, aHAlignment, aVAlignment );
        }

        void UIZLayout_AddNonAlignedFixed( UIZLayout *aSelf, void *aChild, CLRVec2 aSize, CLRVec2 aPosition, bool aExpand, bool aFill )
        {
            auto lChild = CAST( UIComponent, aChild );

            aSelf->Add( lChild, vec( aSize ), vec( aPosition ), aExpand, aFill );
        }
#pragma endregion

#pragma region UIFileTree
        CONSTRUCT_WITHOUT_PARAMETERS( UIFileTree )
        DESTROY_INTERFACE( UIFileTree )

        void *UIFileTree_Add( UIFileTree *aSelf, void *aPath )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aPath ) );

            return CAST( void, aSelf->Add( lString ) );
        }
#pragma endregion

#pragma region UIDialog
        CONSTRUCT_WITHOUT_PARAMETERS( UIDialog )
        DESTROY_INTERFACE( UIDialog )

        void *UIDialog_CreateWithTitleAndSize( void *aTitle, CLRVec2 aSize )
        {
            auto lString    = DotNetRuntime::NewString( CAST( MonoString, aTitle ) );
            auto lNewDialog = new UIDialog( lString, vec( aSize ) );

            return CAST( void, lNewDialog );
        }

        void UIDialog_SetTitle( UIDialog *aSelf, void *aTitle )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aTitle ) );

            aSelf->SetTitle( lString );
        }

        void UIDialog_SetSize( UIDialog *aSelf, CLRVec2 aSize ) { aSelf->SetSize( vec( aSize ) ); }

        void UIDialog_SetContent( UIDialog *aSelf, void *aContent )
        {
            auto lContent = CAST( UIComponent, aContent );

            aSelf->SetContent( lContent );
        }

        void UIDialog_Open( UIDialog *aSelf ) { aSelf->Open(); }

        void UIDialog_Close( UIDialog *aSelf ) { aSelf->Close(); }

        void UIDialog_Update( UIDialog *aSelf ) { aSelf->Update(); }
#pragma endregion

#pragma region UIForm
        CONSTRUCT_WITHOUT_PARAMETERS( UIForm )
        DESTROY_INTERFACE( UIForm )

        void UIForm_SetTitle( UIForm *aSelf, void *aTitle )
        {
            auto lString = DotNetRuntime::NewString( CAST( MonoString, aTitle ) );

            aSelf->SetTitle( lString );
        }

        void UIForm_SetContent( UIForm *aSelf, void *aContent )
        {
            auto lContent = CAST( UIComponent, aContent );

            aSelf->SetContent( lContent );
        }

        void UIForm_Update( UIForm *aSelf ) { aSelf->Update(); }

        void UIForm_SetSize( UIForm *aSelf, float aWidth, float aHeight ) { aSelf->SetSize( aWidth, aHeight ); }
#pragma endregion
    }
} // namespace SE::Core::Interop