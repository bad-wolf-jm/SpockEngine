#include "LinkElementTracePlot.h"

#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>

#include "Core/Profiling/BlockTimer.h"

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/File.h"
#include "Core/Logging.h"

#include "Mono/MonoRuntime.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

namespace SE::OtdrEditor
{
    template <typename _Ty>
    static std::vector<_Ty> AsVector( MonoObject *aObject )
    {
        uint32_t lArrayLength = static_cast<uint32_t>( mono_array_length( (MonoArray *)aObject ) );

        std::vector<_Ty> lVector( lArrayLength );
        for( uint32_t i = 0; i < lArrayLength; i++ )
        {
            auto lElement = *( mono_array_addr( (MonoArray *)aObject, _Ty, i ) );
            lVector[i]    = lElement;
        }

        return lVector;
    }

    void UILinkElementTracePlot::SetData( std::vector<MonoObject *> &lTraceDataVector )
    {
        static auto &lTraceDataStructure = MonoRuntime::GetClassType( "Metrino.Interop.TracePlotData" );

        Clear();
        for( int i = 0; i < lTraceDataVector.size(); i++ )
        {
            auto lInstance = MonoScriptInstance( &lTraceDataStructure, lTraceDataStructure.Class(), lTraceDataVector[i] );
            auto lPlot     = New<sFloat64LinePlot>();
            lPlot->mX      = AsVector<double>( lInstance.GetFieldValue<MonoObject *>( "mX" ) );
            lPlot->mY      = AsVector<double>( lInstance.GetFieldValue<MonoObject *>( "mY" ) );
            lPlot->mLegend = fmt::format( "{:.0f} nm - {} ({} samples)", lInstance.GetFieldValue<double>( "mWavelength" ) * 1e9, i,
                                          lPlot->mX.size() );

            Add( lPlot );
        }
    }

    void UILinkElementTracePlot::SetEventData( sLinkElement const &aLinkElement, bool aDisplayEventBounds, bool aDisplayLsaFit,
                                               bool aAdjustAxisScale )
    {
        static auto &lSinglePulseTraceClass = MonoRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );

        auto lTrace = MonoScriptInstance( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), aLinkElement.mPeakTrace );
        if( !lTrace ) return;

        auto lSamples = lTrace.GetPropertyValue<MonoObject *>( "Samples" );
        auto lDeltaX  = lTrace.GetPropertyValue<double>( "SamplingPeriod" );

        auto lPlot = New<sFloat64LinePlot>();

        static const double lSpeedOfLight = 299792458.0;

        lPlot->mY      = AsVector<double>( lSamples );
        lPlot->mX      = std::vector<double>( lPlot->mY.size() );
        lPlot->mLegend = fmt::format( "Trace##{}", (size_t)aLinkElement.mPeakTrace );

        uint32_t lFirst         = 0;
        auto     lStartPosition = lTrace.CallMethod( "ConvertSampleIndexToPosition", &lFirst );
        double   lX0            = *(double *)mono_object_unbox( lStartPosition );

        uint32_t lLast        = lPlot->mY.size() - 1;
        auto     lEndPosition = lTrace.CallMethod( "ConvertSampleIndexToPosition", &lLast );
        double   lX1          = *(double *)mono_object_unbox( lEndPosition );

        for( uint32_t i = 0; i < lPlot->mX.size(); i++ )
            lPlot->mX[i] = ( ( static_cast<float>( i ) / static_cast<float>( lLast ) ) * ( lX1 - lX0 ) + lX0 ) * 0.001;

        Add( lPlot );

        static auto &lBaseLinkElementClass  = MonoRuntime::GetClassType( "Metrino.Olm.BaseLinkElement" );
        static auto &lOlmPhysicalEventClass = MonoRuntime::GetClassType( "Metrino.Olm.OlmPhysicalEvent" );
        static auto &lOlmAttributeClass     = MonoRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );

        auto lLinkElement = MonoScriptInstance( &lBaseLinkElementClass, lBaseLinkElementClass.Class(), aLinkElement.mLinkElement );
        auto lPhysicalEvent =
            MonoScriptInstance( &lOlmPhysicalEventClass, lOlmPhysicalEventClass.Class(), aLinkElement.mPhysicalEvent );
        auto lAttributes = MonoScriptInstance( &lOlmAttributeClass, lOlmAttributeClass.Class(), aLinkElement.mAttributes );

        auto lOtdrPhysicalEvent = lPhysicalEvent.GetPropertyValue( "PhysicalEvent", "Metrino.Otdr.PhysicalEvent" );
        auto lEventSpanStart    = lOtdrPhysicalEvent->GetPropertyValue<double>( "CursorA" );
        auto lEventSpanEnd      = lOtdrPhysicalEvent->GetPropertyValue<double>( "CursorB" );
        auto lElementPosition   = lLinkElement.GetPropertyValue<double>( "Position" );

        if( aDisplayEventBounds )
        {
            auto lEventSpanLine    = New<sVLine>( std::vector<double>{ lEventSpanStart * 0.001, lEventSpanEnd * 0.001 } );
            lEventSpanLine->mColor = math::vec4{ 1.0f, 1.0f, 1.0f, 1.0f };
            Add( lEventSpanLine );

            auto lEventPositionLine    = New<sVLine>( std::vector<double>{ lElementPosition * 0.001 } );
            lEventPositionLine->mColor = math::vec4{ 1.0f, .0f, .0f, 1.0f };
            Add( lEventPositionLine );
        }

        if( aDisplayLsaFit )
        {
            auto lPreviousRbs        = lAttributes.GetPropertyValue( "PreviousRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
            auto lPreviousRbsLsaData = lPreviousRbs->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );
            if( *lPreviousRbsLsaData )
            {
                auto lPreviousRbsSlope  = lPreviousRbsLsaData->GetPropertyValue<double>( "Slope" );
                auto lPreviousRbsOffset = lPreviousRbsLsaData->GetPropertyValue<double>( "Offset" );
                auto lPreviousRbsPlot   = New<sFloat64LinePlot>();
                auto lPreviousX0        = lPreviousRbsLsaData->GetPropertyValue<double>( "StartPosition" );
                auto lPreviousX1        = lPreviousRbsLsaData->GetPropertyValue<double>( "EndPosition" );

                lPreviousRbsPlot->mX = std::vector<double>{ lPreviousX0 * 0.001, lPreviousX1 * 0.001 };
                lPreviousRbsPlot->mY =
                    std::vector<double>{ lPreviousRbsOffset, ( lPreviousX1 - lPreviousX0 ) * lPreviousRbsSlope + lPreviousRbsOffset };
                lPreviousRbsPlot->mColor     = math::vec4{ 1.0f, 1.0f, 1.0f, 1.0f };
                lPreviousRbsPlot->mThickness = 2.0f;
                Add( lPreviousRbsPlot );
            }

            auto lNextRbs        = lAttributes.GetPropertyValue( "NextRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
            auto lNextRbsLsaData = lNextRbs->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );
            if( *lNextRbsLsaData )
            {
                auto lNextRbsSlope  = lNextRbsLsaData->GetPropertyValue<double>( "Slope" );
                auto lNextRbsOffset = lNextRbsLsaData->GetPropertyValue<double>( "Offset" );
                auto lNextRbsPlot   = New<sFloat64LinePlot>();
                auto lNextX0        = lNextRbsLsaData->GetPropertyValue<double>( "StartPosition" );
                auto lNextX1        = lNextRbsLsaData->GetPropertyValue<double>( "EndPosition" );

                lNextRbsPlot->mX     = std::vector<double>{ lNextX0 * 0.001, lNextX1 * 0.001 };
                lNextRbsPlot->mY     = std::vector<double>{ lNextRbsOffset, ( lNextX1 - lNextX0 ) * lNextRbsSlope + lNextRbsOffset };
                lNextRbsPlot->mColor = math::vec4{ 1.0f, 1.0f, 1.0f, 1.0f };
                lNextRbsPlot->mThickness = 2.0f;
                Add( lNextRbsPlot );
            }
        }

        if( aAdjustAxisScale )
        {
            mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMin = lPlot->mX[0];
            mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMax = lPlot->mX[lPlot->mX.size() - 1];

            auto lPreviousRbs        = lAttributes.GetPropertyValue( "PreviousRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
            auto lPreviousRbsLsaData = lPreviousRbs->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );
            if( *lPreviousRbsLsaData )
            {
                mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMin =
                    lPreviousRbsLsaData->GetPropertyValue<double>( "StartPosition" ) * 0.001;
            }

            auto lNextRbs        = lAttributes.GetPropertyValue( "NextRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
            auto lNextRbsLsaData = lNextRbs->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );
            if( *lNextRbsLsaData )
            {
                mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMax =
                    lNextRbsLsaData->GetPropertyValue<double>( "EndPosition" ) * 0.001;
            }

            mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMin = 0.0f;
            mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMax = 0.0f;

            bool lAxisMinSet = false;
            bool lAxisMaxSet = false;

            for( uint32_t i = 0; i < lPlot->mX.size(); i++ )
            {
                if( ( lPlot->mX[i] >= mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMin ) &&
                    ( lPlot->mX[i] <= mAxisConfiguration[static_cast<int>( UIPlotAxis::X1 )].mMax ) )
                {
                    mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMin =
                        lAxisMinSet ? std::min( static_cast<float>( lPlot->mY[i] ),
                                                mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMin )
                                    : static_cast<float>( lPlot->mY[i] );

                    mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMax =
                        lAxisMaxSet ? std::max( static_cast<float>( lPlot->mY[i] ),
                                                mAxisConfiguration[static_cast<int>( UIPlotAxis::Y1 )].mMax )
                                    : static_cast<float>( lPlot->mY[i] );

                    lAxisMinSet = true;
                    lAxisMaxSet = true;
                }
            }
        }
    }

    void UILinkElementTracePlot::SetEventData( std::vector<sLinkElement> &aLinkElement )
    {
        static auto &lSinglePulseTraceClass = MonoRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );

        for( int i = 0; i < aLinkElement.size(); i++ )
        {
            SetEventData( aLinkElement[i] );
        }
    }

} // namespace SE::OtdrEditor