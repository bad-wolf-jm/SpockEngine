#include "LinkElementTracePlot.h"

#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>
#include <numeric>

#include "Core/Profiling/BlockTimer.h"

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/File.h"
#include "Core/Logging.h"

#include "DotNet/Runtime.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

namespace SE::OtdrEditor
{

    UILinkElementTracePlot::UILinkElementTracePlot()
        : UIBoxLayout( eBoxLayoutOrientation::VERTICAL )
    {
        mTitle    = New<UILabel>( "" );
        mSubTitle = New<UILabel>( "" );
        mPlotArea = New<UIPlot>();

        mTitle->SetAlignment(eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER);

        mSubTitle->SetTextColor( math::vec4{ 0.5, 0.5, 0.5, 1.0 } );
        mSubTitle->SetAlignment(eHorizontalAlignment::LEFT, eVerticalAlignment::CENTER);
        Add( mTitle.get(), 30, false, true );
        Add( mSubTitle.get(), 30, false, true );
        Add( mPlotArea.get(), true, true );
    }

    void UILinkElementTracePlot::Clear() { mPlotArea->Clear(); }
    void UILinkElementTracePlot::SetTitle( std::string aTitle ) { mTitle->SetText( aTitle ); }
    void UILinkElementTracePlot::SetSubTitle( std::string aTitle ) { mSubTitle->SetText( aTitle ); }

    void UILinkElementTracePlot::SetData( std::vector<MonoObject *> &lTraceDataVector )
    {
        static auto &lTraceDataStructure = DotNetRuntime::GetClassType( "Metrino.Interop.TracePlotData" );

        Clear();
        for( int i = 0; i < lTraceDataVector.size(); i++ )
        {
            auto lInstance = DotNetInstance( &lTraceDataStructure, lTraceDataStructure.Class(), lTraceDataVector[i] );
            auto lPlot     = New<sFloat64LinePlot>();
            lPlot->mX      = DotNetRuntime::AsVector<double>( lInstance.GetFieldValue<MonoObject *>( "mX" ) );
            lPlot->mY      = DotNetRuntime::AsVector<double>( lInstance.GetFieldValue<MonoObject *>( "mY" ) );
            lPlot->mLegend = fmt::format( "{:.0f} nm - {} ({} samples)", lInstance.GetFieldValue<double>( "mWavelength" ) * 1e9, i,
                                          lPlot->mX.size() );

            mPlotArea->Add( lPlot );
        }
    }

    void UILinkElementTracePlot::SetEventData( sLinkElement const &aLinkElement, bool aDisplayEventBounds, bool aDisplayLsaFit,
                                               bool aAdjustAxisScale )
    {
        SetTitle( "Single Event Trace" );
        std::vector<std::string> lSubTitle;

        static auto &lSinglePulseTraceClass = DotNetRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );

        auto lPeakPlot = New<sFloat64LinePlot>();
        {
            auto lTrace = aLinkElement.mPeakTrace;
            if( aLinkElement.mPeakTrace && *aLinkElement.mPeakTrace )
            {
                auto lSamples      = lTrace->GetPropertyValue<MonoObject *>( "Samples" );
                auto lDeltaX       = lTrace->GetPropertyValue<double>( "SamplingPeriod" );
                lPeakPlot->mY      = DotNetRuntime::AsVector<double>( lSamples );
                lPeakPlot->mX      = std::vector<double>( lPeakPlot->mY.size() );
                lPeakPlot->mLegend = fmt::format( "Peak Trace##{}", (size_t)aLinkElement.mPeakTrace.get() );

                uint32_t lFirst         = 0;
                auto     lStartPosition = lTrace->CallMethod( "ConvertSampleIndexToPosition", &lFirst );
                double   lX0            = *(double *)mono_object_unbox( lStartPosition );

                uint32_t lLast        = lPeakPlot->mY.size() - 1;
                auto     lEndPosition = lTrace->CallMethod( "ConvertSampleIndexToPosition", &lLast );
                double   lX1          = *(double *)mono_object_unbox( lEndPosition );

                for( uint32_t i = 0; i < lPeakPlot->mX.size(); i++ )
                    lPeakPlot->mX[i] = ( ( static_cast<float>( i ) / static_cast<float>( lLast ) ) * ( lX1 - lX0 ) + lX0 ) * 0.001;

                mPlotArea->Add( lPeakPlot );

                lSubTitle.push_back( fmt::format( "Peak trace: samples={}, period={} ns, length={} km", lPeakPlot->mY.size(),
                                                  lDeltaX * 1e9, ( lEndPosition - lStartPosition ) * 0.001 ) );
            }
        }

        auto lDetectionPlot = New<sFloat64LinePlot>();
        {
            auto lTrace = aLinkElement.mDetectionTrace;
            if( lTrace && *lTrace )
            {
                auto lSamples           = lTrace->GetPropertyValue<MonoObject *>( "Samples" );
                auto lDeltaX            = lTrace->GetPropertyValue<double>( "SamplingPeriod" );
                lDetectionPlot->mY      = DotNetRuntime::AsVector<double>( lSamples );
                lDetectionPlot->mX      = std::vector<double>( lDetectionPlot->mY.size() );
                lDetectionPlot->mLegend = fmt::format( "Detection Trace##{}", (size_t)aLinkElement.mDetectionTrace.get() );

                uint32_t lFirst         = 0;
                auto     lStartPosition = lTrace->CallMethod( "ConvertSampleIndexToPosition", &lFirst );
                double   lX0            = *(double *)mono_object_unbox( lStartPosition );

                uint32_t lLast        = lDetectionPlot->mY.size() - 1;
                auto     lEndPosition = lTrace->CallMethod( "ConvertSampleIndexToPosition", &lLast );
                double   lX1          = *(double *)mono_object_unbox( lEndPosition );

                for( uint32_t i = 0; i < lDetectionPlot->mX.size(); i++ )
                    lDetectionPlot->mX[i] =
                        ( ( static_cast<float>( i ) / static_cast<float>( lLast ) ) * ( lX1 - lX0 ) + lX0 ) * 0.001;

                mPlotArea->Add( lDetectionPlot );

                lSubTitle.push_back( fmt::format( "Detection trace: samples={}, period={} ns, length={} km", lPeakPlot->mY.size(),
                                                  lDeltaX * 1e9, ( lEndPosition - lStartPosition ) * 0.001 ) );
            }
        }

        auto StringJoin = []( std::vector<std::string> aList )
        {
            return aList.empty() ? "N/A"
                                 : std::accumulate( aList.begin(), aList.end(), std::string(),
                                                    []( const std::string &a, const std::string &b ) -> std::string
                                                    { return a + ( a.length() > 0 ? " - " : "" ) + b; } );
        };
        SetSubTitle( StringJoin( lSubTitle ) );

        auto &lLinkElement   = *aLinkElement.mLinkElement;
        auto &lPhysicalEvent = *aLinkElement.mPhysicalEvent;
        auto &lAttributes    = *aLinkElement.mAttributes;

        auto lOtdrPhysicalEvent = lPhysicalEvent.GetPropertyValue( "PhysicalEvent", "Metrino.Otdr.PhysicalEvent" );
        if( lOtdrPhysicalEvent && *lOtdrPhysicalEvent )
        {
            auto lEventSpanStart  = lOtdrPhysicalEvent->GetPropertyValue<double>( "CursorA" );
            auto lEventSpanEnd    = lOtdrPhysicalEvent->GetPropertyValue<double>( "CursorB" );
            auto lElementPosition = lLinkElement.GetPropertyValue<double>( "Position" );

            if( aDisplayEventBounds )
            {
                auto lEventSpanLine    = New<sVLine>( std::vector<double>{ lEventSpanStart * 0.001, lEventSpanEnd * 0.001 } );
                lEventSpanLine->mColor = math::vec4{ 1.0f, 1.0f, 1.0f, 1.0f };
                mPlotArea->Add( lEventSpanLine );

                auto lEventPositionLine    = New<sVLine>( std::vector<double>{ lElementPosition * 0.001 } );
                lEventPositionLine->mColor = math::vec4{ 1.0f, .0f, .0f, 1.0f };
                mPlotArea->Add( lEventPositionLine );
            }
        }

        if( aDisplayLsaFit && lAttributes )
        {
            auto lPreviousRbs        = lAttributes.GetPropertyValue( "PreviousRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
            auto lPreviousRbsLsaData = lPreviousRbs->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );

            auto lNextRbs        = lAttributes.GetPropertyValue( "NextRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
            auto lNextRbsLsaData = lNextRbs->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );

            if( lPreviousRbsLsaData && *lPreviousRbsLsaData )
            {
                auto lPreviousRbsSlope  = lPreviousRbsLsaData->GetPropertyValue<double>( "Slope" );
                auto lPreviousRbsOffset = lPreviousRbsLsaData->GetPropertyValue<double>( "Offset" );
                auto lPreviousX0        = lPreviousRbsLsaData->GetPropertyValue<double>( "StartPosition" );
                auto lPreviousX1        = lPreviousRbsLsaData->GetPropertyValue<double>( "EndPosition" );

                double X0 = lPreviousX0;
                double X1 = lPreviousX1;
                if( lNextRbsLsaData && *lNextRbsLsaData ) X1 = lNextRbsLsaData->GetPropertyValue<double>( "EndPosition" );
                double Y0 = ( X0 - lPreviousX0 ) * lPreviousRbsSlope + lPreviousRbsOffset;
                double Y1 = ( X1 - lPreviousX0 ) * lPreviousRbsSlope + lPreviousRbsOffset;

                auto lPreviousRbsPlot        = New<sFloat64LinePlot>();
                lPreviousRbsPlot->mX         = std::vector<double>{ X0 * 0.001, X1 * 0.001 };
                lPreviousRbsPlot->mY         = std::vector<double>{ Y0, Y1 };
                lPreviousRbsPlot->mLegend    = "Previous Section";
                lPreviousRbsPlot->mColor     = math::vec4{ 1.0f, 1.0f, .5f, 1.0f };
                lPreviousRbsPlot->mThickness = 2.0f;
                mPlotArea->Add( lPreviousRbsPlot );

                auto lPreviousRbsBoundsPlot = New<sFloat64ScatterPlot>();
                lPreviousRbsBoundsPlot->mX  = std::vector<double>{ lPreviousX0 * 0.001, lPreviousX1 * 0.001 };
                lPreviousRbsBoundsPlot->mY =
                    std::vector<double>{ lPreviousRbsOffset, ( lPreviousX1 - lPreviousX0 ) * lPreviousRbsSlope + lPreviousRbsOffset };
                lPreviousRbsBoundsPlot->mColor = math::vec4{ 1.0f, 1.0f, .5f, 1.0f };
                mPlotArea->Add( lPreviousRbsBoundsPlot );
            }

            if( lNextRbsLsaData && *lNextRbsLsaData )
            {
                auto lNextRbsSlope  = lNextRbsLsaData->GetPropertyValue<double>( "Slope" );
                auto lNextRbsOffset = lNextRbsLsaData->GetPropertyValue<double>( "Offset" );
                auto lNextRbsPlot   = New<sFloat64LinePlot>();
                auto lNextX0        = lNextRbsLsaData->GetPropertyValue<double>( "StartPosition" );
                auto lNextX1        = lNextRbsLsaData->GetPropertyValue<double>( "EndPosition" );

                double X0 = lNextX0;
                double X1 = lNextX1;
                if( lPreviousRbsLsaData && *lPreviousRbsLsaData )
                    X0 = lPreviousRbsLsaData->GetPropertyValue<double>( "StartPosition" );
                double Y0 = ( X0 - lNextX0 ) * lNextRbsSlope + lNextRbsOffset;
                double Y1 = ( X1 - lNextX0 ) * lNextRbsSlope + lNextRbsOffset;

                lNextRbsPlot->mX         = std::vector<double>{ X0 * 0.001, X1 * 0.001 };
                lNextRbsPlot->mY         = std::vector<double>{ Y0, Y1 };
                lNextRbsPlot->mLegend    = "Next Section";
                lNextRbsPlot->mColor     = math::vec4{ 1.0f, .5f, 1.0f, 1.0f };
                lNextRbsPlot->mThickness = 2.0f;
                mPlotArea->Add( lNextRbsPlot );

                auto lNextRbsBoundsPlot = New<sFloat64ScatterPlot>();
                lNextRbsBoundsPlot->mX  = std::vector<double>{ lNextX0 * 0.001, lNextX1 * 0.001 };
                lNextRbsBoundsPlot->mY = std::vector<double>{ lNextRbsOffset, ( lNextX1 - lNextX0 ) * lNextRbsSlope + lNextRbsOffset };
                lNextRbsBoundsPlot->mColor = math::vec4{ 1.0f, .5f, 1.0f, 1.0f };
                mPlotArea->Add( lNextRbsBoundsPlot );
            }
        }

        if( aAdjustAxisScale && lAttributes )
        {
            mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMin = lPeakPlot->mX[0];
            mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMax = lPeakPlot->mX[lPeakPlot->mX.size() - 1];

            auto lPreviousRbs        = lAttributes.GetPropertyValue( "PreviousRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
            auto lPreviousRbsLsaData = lPreviousRbs->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );
            if( lPreviousRbsLsaData && *lPreviousRbsLsaData )
            {
                mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMin =
                    lPreviousRbsLsaData->GetPropertyValue<double>( "StartPosition" ) * 0.001;
            }

            auto lNextRbs        = lAttributes.GetPropertyValue( "NextRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
            auto lNextRbsLsaData = lNextRbs->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );
            if( lNextRbsLsaData && *lNextRbsLsaData )
            {
                mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMax =
                    lNextRbsLsaData->GetPropertyValue<double>( "EndPosition" ) * 0.001;
            }

            mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMin = 0.0f;
            mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMax = 0.0f;

            bool lAxisMinSet = false;
            bool lAxisMaxSet = false;

            for( uint32_t i = 0; i < lPeakPlot->mX.size(); i++ )
            {
                if( ( lPeakPlot->mX[i] >= mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMin ) &&
                    ( lPeakPlot->mX[i] <= mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMax ) )
                {
                    mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMin =
                        lAxisMinSet ? std::min( static_cast<float>( lPeakPlot->mY[i] ),
                                                mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMin )
                                    : static_cast<float>( lPeakPlot->mY[i] );

                    mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMax =
                        lAxisMaxSet ? std::max( static_cast<float>( lPeakPlot->mY[i] ),
                                                mPlotArea->mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMax )
                                    : static_cast<float>( lPeakPlot->mY[i] );

                    lAxisMinSet = true;
                    lAxisMaxSet = true;
                }
            }
        }
    }

    void UILinkElementTracePlot::SetEventData( std::vector<sLinkElement> &aLinkElement )
    {
        for( int i = 0; i < aLinkElement.size(); i++ ) SetEventData( aLinkElement[i] );
        SetTitle("All traces");
        SetSubTitle("");
    }

} // namespace SE::OtdrEditor