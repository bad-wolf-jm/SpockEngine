#include "Plot.h"
#include "implot_internal.h"



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

    void UIPlot::Add( Ref<UIPlotData> aPlot ) { mElements.push_back( aPlot.get() ); };
    void UIPlot::Add( UIPlotData *aPlot ) { mElements.push_back( aPlot ); };
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
    void UIFloat64LinePlot::Render( UIPlot *aParentPlot )
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

    void UIFloat64ScatterPlot::Render( UIPlot *aParentPlot )
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

    void UIVLinePlot::Render( UIPlot *aParentPlot )
    {
        auto lPlotName = fmt::format( "{}##{}", mLegend, static_cast<void *>( this ) );

        ImPlot::SetAxes( static_cast<ImAxis>( mXAxis ), static_cast<ImAxis>( mYAxis ) );

        if( mThickness != -1.0f ) ImPlot::PushStyleVar( ImPlotStyleVar_LineWeight, mThickness );
        ImPlot::PushStyleColor( ImPlotCol_Line, ImVec4{ mColor.x, mColor.y, mColor.z, mColor.w } );
        ImPlot::PlotVLines( lPlotName.c_str(), mX.data(), mX.size(), 0 );
        ImPlot::PopStyleColor();
        if( mThickness != -1.0f ) ImPlot::PopStyleVar();
    }

    void UIHLinePlot::Render( UIPlot *aParentPlot )
    {
        auto lPlotName = fmt::format( "{}##{}", mLegend, static_cast<void *>( this ) );

        ImPlot::SetAxes( static_cast<ImAxis>( mXAxis ), static_cast<ImAxis>( mYAxis ) );

        if( mThickness != -1.0f ) ImPlot::PushStyleVar( ImPlotStyleVar_LineWeight, mThickness );

        ImPlot::PushStyleColor( ImPlotCol_Line, ImVec4{ mColor.x, mColor.y, mColor.z, mColor.w } );
        ImPlot::PlotHLines( lPlotName.c_str(), mY.data(), mY.size(), 0 );
        ImPlot::PopStyleColor();

        if( mThickness != -1.0f ) ImPlot::PopStyleVar();
    }

    void UIAxisTag::Render( UIPlot *aParentPlot )
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

    void UIVRangePlot::Render( UIPlot *aParentPlot )
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

    void UIHRangePlot::Render( UIPlot *aParentPlot )
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
} // namespace SE::Core