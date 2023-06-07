#include "Plot.h"
#include "implot_internal.h"

#include "DotNet/Runtime.h"

namespace SE::Core
{
    UIPlot::UIPlot()
    {
        mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mAxis = UIPlotAxis::X1;
        mAxisConfiguration[static_cast<int>( UIPlotAxis::X2 )].mAxis = UIPlotAxis::X2;
        mAxisConfiguration[static_cast<int>( UIPlotAxis::X3 )].mAxis = UIPlotAxis::X3;

        mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mAxis = UIPlotAxis::Y1;
        mAxisConfiguration[static_cast<int>( UIPlotAxis::Y2 )].mAxis = UIPlotAxis::Y2;
        mAxisConfiguration[static_cast<int>( UIPlotAxis::Y3 )].mAxis = UIPlotAxis::Y3;

        mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mInUse = true;
        mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mInUse = true;
    }

    void UIPlot::PushStyles() {}
    void UIPlot::PopStyles() {}

    void UIPlot::Add( Ref<sPlotData> aPlot ) { mElements.push_back( aPlot.get() ); };
    void UIPlot::Add( sPlotData *aPlot ) { mElements.push_back( aPlot ); };
    void UIPlot::Clear() { mElements.clear(); };
    void UIPlot::ConfigureLegend( math::vec2 aLegendPadding, math::vec2 aLegendInnerPadding, math::vec2 aLegendSpacing )
    {
        mLegendPadding      = ImVec2{ aLegendPadding.x, aLegendPadding.y };
        mLegendInnerPadding = ImVec2{ aLegendInnerPadding.x, aLegendInnerPadding.y };
        mLegendSpacing      = ImVec2{ aLegendSpacing.x, aLegendSpacing.y };
    }

    ImVec2 UIPlot::RequiredSize() { return ImVec2{ 150.0f, 150.0f }; }

    void UIPlot::DrawContent( ImVec2 aPosition, ImVec2 aSize )
    {
        auto &lStyle              = ImPlot::GetStyle();
        lStyle.LegendPadding      = mLegendPadding;
        lStyle.LegendInnerPadding = mLegendInnerPadding;
        lStyle.LegendSpacing      = mLegendSpacing;

        ImGui::SetCursorPos( aPosition );
        ImPlot::PushStyleVar( ImPlotStyleVar_PlotPadding, ImVec2( 0, 0 ) );

        for( auto &lAxisConfig : mAxisConfiguration ) lAxisConfig.mInUse = false;
        mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mInUse = true;
        mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mInUse = true;

        for( auto const &lPlotElement : mElements )
        {
            mAxisConfiguration[static_cast<int>( lPlotElement->mXAxis )].mInUse = true;
            mAxisConfiguration[static_cast<int>( lPlotElement->mYAxis )].mInUse = true;
        }

        if( ImPlot::BeginPlot( "##", aSize, ImPlotFlags_Crosshairs | ImPlotFlags_NoChild | ImPlotFlags_AntiAliased ) )
        {
            auto *lPlot = ImPlot::GetCurrentPlot();
            for( auto &lAxis : mAxisConfiguration )
            {
                if( lAxis.mInUse )
                {
                    ImPlot::SetupAxis( static_cast<ImAxis>( lAxis.mAxis ), lAxis.mTitle.empty() ? NULL : lAxis.mTitle.c_str(),
                                       ImPlotAxisFlags_None );
                    if( lAxis.mSetLimitRequest )
                    {
                        lPlot->Axes[static_cast<ImAxis>( lAxis.mAxis )].SetRange( lAxis.mMin, lAxis.mMax );

                        lAxis.mSetLimitRequest = false;
                    }
                }
            }

            ImPlotLegendFlags flags = ImPlotLegendFlags_None;
            // ImPlot::SetupLegend( mLegendPosition, flags );
            uint32_t lIndex = 0;
            for( auto const &lPlotElement : mElements )
            {
                if( lPlotElement != nullptr ) lPlotElement->Render( this );
            }
            ImPlot::EndPlot();
        }
        ImPlot::PopStyleVar();
    }

    // void *UIPlot::UIPlot_Create()
    // {
    //     auto lNewPlot = new UIPlot();

    //     return static_cast<void *>( lNewPlot );
    // }

    // void UIPlot::UIPlot_Destroy( void *aInstance ) { delete static_cast<UIPlot *>( aInstance ); }

    // void UIPlot::UIPlot_Clear( void *aInstance )
    // {
    //     auto lSelf = static_cast<UIPlot *>( aInstance );

    //     lSelf->Clear();
    // }

    // void UIPlot::UIPlot_ConfigureLegend( void *aInstance, math::vec2 *aLegendPadding, math::vec2 *aLegendInnerPadding,
    //                                      math::vec2 *aLegendSpacing )
    // {
    //     auto lSelf = static_cast<UIPlot *>( aInstance );

    //     lSelf->ConfigureLegend( *aLegendPadding, *aLegendInnerPadding, *aLegendSpacing );
    // }

    // void UIPlot::UIPlot_Add( void *aInstance, void *aPlot )
    // {
    //     auto lSelf = static_cast<UIPlot *>( aInstance );
    //     auto lPlot = static_cast<sPlotData *>( aPlot );

    //     lSelf->Add( lPlot );
    // }

    // void UIPlot::UIPlot_SetAxisLimits( void *aInstance, int aAxis, double aMin, double aMax )
    // {
    //     auto lSelf = static_cast<UIPlot *>( aInstance );

    //     lSelf->mAxisConfiguration[aAxis].mSetLimitRequest = true;

    //     lSelf->mAxisConfiguration[aAxis].mMin = static_cast<float>( aMin );
    //     lSelf->mAxisConfiguration[aAxis].mMax = static_cast<float>( aMax );
    // }

    // void UIPlot::UIPlot_SetAxisTitle( void *aInstance, int aAxis, void *aTitle )
    // {
    //     auto lSelf = static_cast<UIPlot *>( aInstance );

    //     lSelf->mAxisConfiguration[aAxis].mTitle = DotNetRuntime::NewString( static_cast<MonoString *>( aTitle ) );
    // }

    // void *UIPlot::UIPlot_GetAxisTitle( void *aInstance, int aAxis )
    // {
    //     auto lSelf = static_cast<UIPlot *>( aInstance );

    //     return DotNetRuntime::NewString( lSelf->mAxisConfiguration[aAxis].mTitle );
    // }

    // void sPlotData::UIPlotData_SetLegend( void *aSelf, void *aText )
    // {
    //     auto lSelf   = static_cast<sPlotData *>( aSelf );
    //     auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

    //     lSelf->mLegend = lString;
    // }

    // void sPlotData::UIPlotData_SetThickness( void *aSelf, float aThickness )
    // {
    //     auto lSelf = static_cast<sPlotData *>( aSelf );

    //     lSelf->mThickness = aThickness;
    // }

    // void sPlotData::UIPlotData_SetColor( void *aSelf, math::vec4 aColor )
    // {
    //     auto lSelf = static_cast<sPlotData *>( aSelf );

    //     lSelf->mColor = aColor;
    // }

    // void sPlotData::UIPlotData_SetXAxis( void *aSelf, int aAxis )
    // {
    //     auto lSelf = static_cast<sPlotData *>( aSelf );

    //     lSelf->mXAxis = static_cast<UIPlotAxis>( aAxis );
    // }

    // void sPlotData::UIPlotData_SetYAxis( void *aSelf, int aAxis )
    // {
    //     auto lSelf = static_cast<sPlotData *>( aSelf );

    //     lSelf->mYAxis = static_cast<UIPlotAxis>( aAxis );
    // }

    void sFloat64LinePlot::Render( UIPlot *aParentPlot )
    {
        auto lPlotName = fmt::format( "{}##{}", mLegend, static_cast<void *>( this ) );
        auto lPlotSize = ImPlot::GetPlotSize();

        int lDownSample = static_cast<int>( static_cast<float>( mX.size() ) / lPlotSize.x );
        lDownSample     = ( lDownSample < 1 ) ? 1 : lDownSample;

        ImPlot::SetAxes( static_cast<ImAxis>( mXAxis ), static_cast<ImAxis>( mYAxis ) );

        if( mThickness != -1.0f ) ImPlot::PushStyleVar( ImPlotStyleVar_LineWeight, mThickness );
        ImPlot::PushStyleColor( ImPlotCol_Line, ImVec4{ mColor.x, mColor.y, mColor.z, mColor.w } );
        ImPlot::PlotLine( mLegend.c_str(), mX.data(), mY.data(), mX.size() / lDownSample, mOffset, sizeof( double ) * lDownSample );
        ImPlot::PopStyleColor();
        if( mThickness != -1.0f ) ImPlot::PopStyleVar();
    }

    // void *sFloat64LinePlot::UIFloat64LinePlot_Create()
    // {
    //     auto lSelf = new sFloat64LinePlot();

    //     return static_cast<sFloat64LinePlot *>( lSelf );
    // }

    // void sFloat64LinePlot::UIFloat64LinePlot_Destroy( void *aSelf ) { delete static_cast<sFloat64LinePlot *>( aSelf ); }

    // void sFloat64LinePlot::UIFloat64LinePlot_SetX( void *aSelf, void *aValue )
    // {
    //     auto lSelf = static_cast<sFloat64LinePlot *>( aSelf );

    //     lSelf->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    // }

    // void sFloat64LinePlot::UIFloat64LinePlot_SetY( void *aSelf, void *aValue )
    // {
    //     auto lSelf = static_cast<sFloat64LinePlot *>( aSelf );

    //     lSelf->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    // }

    void sFloat64ScatterPlot::Render( UIPlot *aParentPlot )
    {
        auto lPlotName = fmt::format( "{}##{}", mLegend, static_cast<void *>( this ) );
        auto lPlotSize = ImPlot::GetPlotSize();

        int lDownSample = static_cast<int>( static_cast<float>( mX.size() ) / lPlotSize.x );
        lDownSample     = ( lDownSample < 1 ) ? 1 : lDownSample;

        ImPlot::SetAxes( static_cast<ImAxis>( mXAxis ), static_cast<ImAxis>( mYAxis ) );

        if( mThickness != -1.0f ) ImPlot::PushStyleVar( ImPlotStyleVar_LineWeight, mThickness );
        ImPlot::PushStyleColor( ImPlotCol_Line, ImVec4{ mColor.x, mColor.y, mColor.z, mColor.w } );
        ImPlot::PlotScatter( mLegend.c_str(), mX.data(), mY.data(), mX.size() / lDownSample, mOffset, sizeof( double ) * lDownSample );
        ImPlot::PopStyleColor();
        if( mThickness != -1.0f ) ImPlot::PopStyleVar();
    }

    // void *sFloat64ScatterPlot::UIFloat64ScatterPlot_Create()
    // {
    //     auto lSelf = new sFloat64ScatterPlot();

    //     return static_cast<sFloat64ScatterPlot *>( lSelf );
    // }

    // void sFloat64ScatterPlot::UIFloat64ScatterPlot_Destroy( void *aSelf ) { delete static_cast<sFloat64ScatterPlot *>( aSelf ); }

    // void sFloat64ScatterPlot::UIFloat64ScatterPlot_SetX( void *aSelf, void *aValue )
    // {
    //     auto lSelf = static_cast<sFloat64ScatterPlot *>( aSelf );

    //     lSelf->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    // }

    // void sFloat64ScatterPlot::UIFloat64ScatterPlot_SetY( void *aSelf, void *aValue )
    // {
    //     auto lSelf = static_cast<sFloat64ScatterPlot *>( aSelf );

    //     lSelf->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    // }

    void sVLine::Render( UIPlot *aParentPlot )
    {
        auto lPlotName = fmt::format( "{}##{}", mLegend, static_cast<void *>( this ) );

        ImPlot::SetAxes( static_cast<ImAxis>( mXAxis ), static_cast<ImAxis>( mYAxis ) );

        if( mThickness != -1.0f ) ImPlot::PushStyleVar( ImPlotStyleVar_LineWeight, mThickness );
        ImPlot::PushStyleColor( ImPlotCol_Line, ImVec4{ mColor.x, mColor.y, mColor.z, mColor.w } );
        ImPlot::PlotVLines( lPlotName.c_str(), mX.data(), mX.size(), 0 );
        ImPlot::PopStyleColor();
        if( mThickness != -1.0f ) ImPlot::PopStyleVar();
    }

    // void *sVLine::UIVLinePlot_Create()
    // {
    //     auto lSelf = new sVLine();

    //     return static_cast<sVLine *>( lSelf );
    // }

    // void sVLine::UIVLinePlot_Destroy( void *aSelf ) { delete static_cast<sVLine *>( aSelf ); }

    // void sVLine::UIVLinePlot_SetX( void *aSelf, void *aValue )
    // {
    //     auto lSelf = static_cast<sVLine *>( aSelf );

    //     lSelf->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    // }

    void sHLine::Render( UIPlot *aParentPlot )
    {
        auto lPlotName = fmt::format( "{}##{}", mLegend, static_cast<void *>( this ) );

        ImPlot::SetAxes( static_cast<ImAxis>( mXAxis ), static_cast<ImAxis>( mYAxis ) );

        if( mThickness != -1.0f ) ImPlot::PushStyleVar( ImPlotStyleVar_LineWeight, mThickness );

        ImPlot::PushStyleColor( ImPlotCol_Line, ImVec4{ mColor.x, mColor.y, mColor.z, mColor.w } );
        ImPlot::PlotHLines( lPlotName.c_str(), mY.data(), mY.size(), 0 );
        ImPlot::PopStyleColor();

        if( mThickness != -1.0f ) ImPlot::PopStyleVar();
    }

    // void *sHLine::UIHLinePlot_Create()
    // {
    //     auto lSelf = new sHLine();

    //     return static_cast<sHLine *>( lSelf );
    // }

    // void sHLine::UIHLinePlot_Destroy( void *aSelf ) { delete static_cast<sHLine *>( aSelf ); }

    // void sHLine::UIHLinePlot_SetY( void *aSelf, void *aValue )
    // {
    //     auto lSelf = static_cast<sHLine *>( aSelf );

    //     lSelf->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    // }

    void sAxisTag::Render( UIPlot *aParentPlot )
    {
        auto lPlotName = fmt::format( "{}##{}", mLegend, static_cast<void *>( this ) );

        ImPlot::SetAxes( static_cast<ImAxis>( mXAxis ), static_cast<ImAxis>( mYAxis ) );
        switch( mAxis )
        {
        case UIPlotAxis::X1:
        case UIPlotAxis::X2:
        case UIPlotAxis::X3: ImPlot::TagX( mX, ImVec4{ mColor.x, mColor.y, mColor.z, mColor.w }, mText.c_str(), true ); break;

        case UIPlotAxis::Y1:
        case UIPlotAxis::Y2:
        case UIPlotAxis::Y3: ImPlot::TagY( mX, ImVec4{ mColor.x, mColor.y, mColor.z, mColor.w }, mText.c_str(), true ); break;
        }
    }

    // void *sAxisTag::UIAxisTag_Create()
    // {
    //     auto lSelf = new sAxisTag();

    //     return static_cast<sAxisTag *>( lSelf );
    // }

    // void *sAxisTag::UIAxisTag_CreateWithTextAndColor( UIPlotAxis aAxis, double aX, void *aText, math::vec4 aColor )
    // {
    //     auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

    //     auto lSelf = new sAxisTag( aAxis, aX, lString, aColor );

    //     return static_cast<sAxisTag *>( lSelf );
    // }

    // void sAxisTag::UIAxisTag_Destroy( void *aSelf ) { delete static_cast<sAxisTag *>( aSelf ); }

    // void sAxisTag::UIAxisTag_SetX( void *aSelf, double aValue )
    // {
    //     auto lSelf = static_cast<sAxisTag *>( aSelf );

    //     lSelf->mX = aValue;
    // }

    // void sAxisTag::UIAxisTag_SetText( void *aSelf, void *aText )
    // {
    //     auto lSelf   = static_cast<sAxisTag *>( aSelf );
    //     auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

    //     lSelf->mText = lString;
    // }

    // void sAxisTag::UIAxisTag_SetColor( void *aSelf, math::vec4 aColor )
    // {
    //     auto lSelf = static_cast<sAxisTag *>( aSelf );

    //     lSelf->mColor = aColor;
    // }

    // math::vec4 sAxisTag::UIAxisTag_GetColor( void *aSelf )
    // {
    //     auto lSelf = static_cast<sAxisTag *>( aSelf );

    //     return lSelf->mColor;
    // }

    // void sAxisTag::UIAxisTag_SetAxis( void *aSelf, int aAxis )
    // {
    //     auto lSelf = static_cast<sAxisTag *>( aSelf );

    //     lSelf->mAxis = static_cast<UIPlotAxis>( aAxis );
    // }

    // int sAxisTag::UIAxisTag_GetAxis( void *aSelf )
    // {
    //     auto lSelf = static_cast<sAxisTag *>( aSelf );

    //     return static_cast<int>( lSelf->mXAxis );
    // }

    void sVRange::Render( UIPlot *aParentPlot )
    {
        ImPlotPlot  *plot     = ImPlot::GetCurrentPlot();
        ImGuiWindow *Window   = ImGui::GetCurrentWindow();
        ImDrawList  *DrawList = Window->DrawList;

        auto  M   = plot->Axes[static_cast<ImAxis>( mXAxis )].LinM;
        auto  P0  = plot->Axes[static_cast<ImAxis>( mXAxis )].PixelMin;
        auto  D  = plot->Axes[static_cast<ImAxis>( mXAxis )].Range.Min;
        float lX0 = P0 + M * ( mX0 - D );
        float lX1 = P0 + M * ( mX1 - D );

        float       lY0 = plot->PlotRect.Min.y;
        float       lY1 = plot->PlotRect.Max.y;
        ImVec4      C{ mColor.x, mColor.y, mColor.z, mColor.w };
        const ImU32 lColor = ImGui::GetColorU32( C );

        ImVec2 lMin{ (float)lX0, lY0 };
        ImVec2 lMax{ (float)lX1, lY1 };

        DrawList->AddRectFilled( lMin, lMax, lColor );
    }

    // void *sVRange::UIVRangePlot_Create()
    // {
    //     auto lSelf = new sVRange();

    //     return static_cast<sVRange *>( lSelf );
    // }

    // void sVRange::UIVRangePlot_Destroy( void *aSelf ) { delete static_cast<sVRange *>( aSelf ); }

    // void sVRange::UIVRangePlot_SetMin( void *aSelf, double aValue )
    // {
    //     auto lSelf = static_cast<sVRange *>( aSelf );

    //     lSelf->mX0 = aValue;
    // }

    // double sVRange::UIVRangePlot_GetMin( void *aSelf )
    // {
    //     auto lSelf = static_cast<sVRange *>( aSelf );

    //     return (double)lSelf->mX0;
    // }

    // void sVRange::UIVRangePlot_SetMax( void *aSelf, double aValue )
    // {
    //     auto lSelf = static_cast<sVRange *>( aSelf );

    //     lSelf->mX1 = aValue;
    // }

    // double sVRange::UIVRangePlot_GetMax( void *aSelf )
    // {
    //     auto lSelf = static_cast<sVRange *>( aSelf );

    //     return (double)lSelf->mX1;
    // }



    void sHRange::Render( UIPlot *aParentPlot )
    {
        ImPlotPlot  *plot     = ImPlot::GetCurrentPlot();
        ImGuiWindow *Window   = ImGui::GetCurrentWindow();
        ImDrawList  *DrawList = Window->DrawList;

        auto  M   = plot->Axes[static_cast<ImAxis>( mXAxis )].LinM;
        auto  P0  = plot->Axes[static_cast<ImAxis>( mXAxis )].PixelMin;
        auto  D  = plot->Axes[static_cast<ImAxis>( mXAxis )].Range.Min;
        float lX0 = P0 + M * ( mY0 - D );
        float lX1 = P0 + M * ( mY1 - D );

        float       lY0 = plot->PlotRect.Min.y;
        float       lY1 = plot->PlotRect.Max.y;
        ImVec4      C{ mColor.x, mColor.y, mColor.z, mColor.w };
        const ImU32 lColor = ImGui::GetColorU32( C );

        ImVec2 lMin{ (float)lX0, lY0 };
        ImVec2 lMax{ (float)lX1, lY1 };

        DrawList->AddRectFilled( lMin, lMax, lColor );
    }

    // void *sHRange::UIHRangePlot_Create()
    // {
    //     auto lSelf = new sHRange();

    //     return static_cast<sHRange *>( lSelf );
    // }

    // void sHRange::UIHRangePlot_Destroy( void *aSelf ) { delete static_cast<sHRange *>( aSelf ); }

    // void sHRange::UIHRangePlot_SetMin( void *aSelf, double aValue )
    // {
    //     auto lSelf = static_cast<sHRange *>( aSelf );

    //     lSelf->mY0 = aValue;
    // }

    // double sHRange::UIHRangePlot_GetMin( void *aSelf )
    // {
    //     auto lSelf = static_cast<sHRange *>( aSelf );

    //     return (double)lSelf->mY0;
    // }

    // void sHRange::UIHRangePlot_SetMax( void *aSelf, double aValue )
    // {
    //     auto lSelf = static_cast<sHRange *>( aSelf );

    //     lSelf->mY1 = aValue;
    // }

    // double sHRange::UIHRangePlot_GetMax( void *aSelf )
    // {
    //     auto lSelf = static_cast<sHRange *>( aSelf );

    //     return (double)lSelf->mY1;
    // }

} // namespace SE::Core