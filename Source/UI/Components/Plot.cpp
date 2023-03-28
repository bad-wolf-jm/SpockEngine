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

    void UIPlot::Add( Ref<sPlotData> aPlot )
    {
        mElementRefs.push_back( aPlot );
        mElements.push_back( aPlot.get() );
    };
    void UIPlot::Add( sPlotData *aPlot ) { mElements.push_back( aPlot ); };
    void UIPlot::Clear()
    {
        mElements.clear();
        mElementRefs.clear();
    };
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
                    ImPlot::SetupAxis( static_cast<ImAxis>( lAxis.mAxis ), NULL, ImPlotAxisFlags_None );
                    if( ( lAxis.mMin != 0.0f ) && ( lAxis.mMax != 0.0f ) )
                    {
                        lPlot->Axes[static_cast<ImAxis>( lAxis.mAxis )].SetMin( lAxis.mMin );
                        lPlot->Axes[static_cast<ImAxis>( lAxis.mAxis )].SetMax( lAxis.mMax );

                        lAxis.mMin = 0.0f;
                        lAxis.mMax = 0.0f;
                    }
                }
            }

            ImPlotLegendFlags flags = ImPlotLegendFlags_None;
            ImPlot::SetupLegend( mLegendPosition, flags );
            uint32_t lIndex = 0;
            for( auto const &lPlotElement : mElements ) lPlotElement->Render( this );
            ImPlot::EndPlot();
        }
        ImPlot::PopStyleVar();
    }

    void *UIPlot::UIPlot_Create()
    {
        auto lNewPlot = new UIPlot();

        return static_cast<void *>( lNewPlot );
    }

    void UIPlot::UIPlot_Destroy( void *aInstance ) { delete static_cast<UIPlot *>( aInstance ); }

    void UIPlot::UIPlot_Clear( void *aInstance )
    {
        auto lSelf = static_cast<UIPlot *>( aInstance );

        lSelf->Clear();
    }

    void UIPlot::UIPlot_ConfigureLegend( void *aInstance, math::vec2 *aLegendPadding, math::vec2 *aLegendInnerPadding,
                                         math::vec2 *aLegendSpacing )
    {
        auto lSelf = static_cast<UIPlot *>( aInstance );

        lSelf->ConfigureLegend( *aLegendPadding, *aLegendInnerPadding, *aLegendSpacing );
    }

    void UIPlot::UIPlot_Add( void *aInstance, void *aPlot )
    {
        auto lSelf = static_cast<UIPlot *>( aInstance );
        auto lPlot = static_cast<sPlotData *>( aPlot );

        lSelf->Add( lPlot );
    }

    void sPlotData::UIPlotData_SetLegend( void *aSelf, void *aText )
    {
        auto lSelf   = static_cast<sPlotData *>( aSelf );
        auto lString = DotNetRuntime::NewString( static_cast<MonoString *>( aText ) );

        lSelf->mLegend = lString;
    }

    void sPlotData::UIPlotData_SetColor( void *aSelf, math::vec4 *aColor )
    {
        auto lSelf = static_cast<sPlotData *>( aSelf );

        lSelf->mColor = *aColor;
    }

    void sPlotData::UIPlotData_SetXAxis( void *aSelf, int aAxis )
    {
        auto lSelf = static_cast<sPlotData *>( aSelf );

        lSelf->mXAxis = static_cast<UIPlotAxis>( aAxis );
    }

    void sPlotData::UIPlotData_SetYAxis( void *aSelf, int aAxis )
    {
        auto lSelf = static_cast<sPlotData *>( aSelf );

        lSelf->mYAxis = static_cast<UIPlotAxis>( aAxis );
    }

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

    void *sFloat64LinePlot::UIFloat64LinePlot_Create( )
    {
        auto lSelf = new sFloat64LinePlot();

        return static_cast<sFloat64LinePlot *>( lSelf );
    }

    void sFloat64LinePlot::UIFloat64LinePlot_Destroy( void *aSelf ) { delete static_cast<sFloat64LinePlot *>( aSelf ); }

    void sFloat64LinePlot::UIFloat64LinePlot_SetX( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sFloat64LinePlot *>( aSelf );

        lSelf->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void sFloat64LinePlot::UIFloat64LinePlot_SetY( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sFloat64LinePlot *>( aSelf );

        lSelf->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

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

    void *sFloat64ScatterPlot::UIFloat64ScatterPlot_Create( )
    {
        auto lSelf = new sFloat64ScatterPlot();

        return static_cast<sFloat64ScatterPlot *>( lSelf );
    }

    void sFloat64ScatterPlot::UIFloat64ScatterPlot_Destroy( void *aSelf ) { delete static_cast<sFloat64ScatterPlot *>( aSelf ); }

    void sFloat64ScatterPlot::UIFloat64ScatterPlot_SetX( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sFloat64ScatterPlot *>( aSelf );

        lSelf->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

    void sFloat64ScatterPlot::UIFloat64ScatterPlot_SetY( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sFloat64ScatterPlot *>( aSelf );

        lSelf->mY = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

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

    void *sVLine::UIVLinePlot_Create( )
    {
        auto lSelf = new sVLine();

        return static_cast<sVLine *>( lSelf );
    }

    void sVLine::UIVLinePlot_Destroy( void *aSelf ) { delete static_cast<sVLine *>( aSelf ); }

    void sVLine::UIVLinePlot_SetX( void *aSelf, void *aValue )
    {
        auto lSelf = static_cast<sVLine *>( aSelf );

        lSelf->mX = DotNetRuntime::AsVector<double>( static_cast<MonoObject *>( aValue ) );
    }

} // namespace SE::Core