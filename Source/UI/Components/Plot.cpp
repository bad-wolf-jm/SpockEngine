#include "Plot.h"

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

    void UIPlot::Add( Ref<sFloat64LinePlot> aPlot ) { mElements.push_back( aPlot ); };
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

        if( ImPlot::BeginPlot( "##", aSize, ImPlotFlags_Crosshairs | ImPlotFlags_NoChild ) )
        {
            for( auto const &lAxis : mAxisConfiguration )
            {
                if( lAxis.mInUse ) ImPlot::SetupAxis( static_cast<ImAxis>( lAxis.mAxis ), NULL, ImPlotAxisFlags_None );
            }

            ImPlotLegendFlags flags = ImPlotLegendFlags_None;
            ImPlot::SetupLegend( mLegendPosition, flags );
            uint32_t lIndex = 0;
            for( auto const &lPlotElement : mElements ) lPlotElement->Render( this );
            ImPlot::EndPlot();
        }
        ImPlot::PopStyleVar();
    }

    void sFloat64LinePlot::Render( UIPlot *aParentPlot )
    {
        auto lPlotName = fmt::format( "{}##{}", mLegend, static_cast<void *>( this ) );
        auto lPlotSize = ImPlot::GetPlotSize();

        int lDownSample = static_cast<int>( static_cast<float>( mX.size() ) / lPlotSize.x );
        lDownSample     = ( lDownSample < 1 ) ? 1 : lDownSample;

        ImPlot::SetAxes( static_cast<ImAxis>( mXAxis ), static_cast<ImAxis>( mYAxis ) );
        ImPlot::PlotLine( mLegend.c_str(), mX.data(), mY.data(), mX.size() / lDownSample, mOffset, sizeof( double ) * lDownSample );
    }
} // namespace SE::Core