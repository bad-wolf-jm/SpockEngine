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

    void UILinkElementTracePlot::SetEventData( sLinkElement const &aLinkElement )
    {
        static auto &lSinglePulseTraceClass = MonoRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );

        Clear();

        auto lTrace   = MonoScriptInstance( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), aLinkElement.mPeakTrace );
        auto lSamples = lTrace.GetPropertyValue<MonoObject *>( "Samples" );
        auto lDeltaX  = lTrace.GetPropertyValue<double>( "SamplingPeriod" );
        auto lPlot    = New<sFloat64LinePlot>();

        static const double lSpeedOfLight = 299792458.0;
        lPlot->mY                         = AsVector<double>( lSamples );
        lPlot->mX                         = std::vector<double>( lPlot->mY.size() );
        for( uint32_t i = 0; i < lPlot->mX.size(); i++ )
        {
            lPlot->mX[i] = ( i * lDeltaX ) * lSpeedOfLight * 0.5 * 0.001;
        }

        Add( lPlot );

        static auto &lBaseLinkElementClass  = MonoRuntime::GetClassType( "Metrino.Olm.BaseLinkElement" );
        static auto &lOlmPhysicalEventClass = MonoRuntime::GetClassType( "Metrino.Olm.OlmPhysicalEvent" );
        static auto &lOlmAttributeClass     = MonoRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );

        auto lLinkElement = MonoScriptInstance( &lBaseLinkElementClass, lBaseLinkElementClass.Class(), aLinkElement.mLinkElement );
        auto lPhysicalEvent =
            MonoScriptInstance( &lOlmPhysicalEventClass, lOlmPhysicalEventClass.Class(), aLinkElement.mPhysicalEvent );
        auto lAttributes = MonoScriptInstance( &lOlmAttributeClass, lOlmAttributeClass.Class(), aLinkElement.mAttributes );

        auto lOtdrPhysicalEvent = lPhysicalEvent.GetPropertyValue( "PhysicalEvent", "Metrino.Otdr.PhysicalEvent" );
        auto lEventSpanStart    = lOtdrPhysicalEvent->GetPropertyValue<double>( "CursorA" ) * 0.001f;
        auto lEventSpanEnd      = lOtdrPhysicalEvent->GetPropertyValue<double>( "CursorB" ) * 0.001f;
        auto lElementPosition   = lLinkElement.GetPropertyValue<double>( "Position" ) * 0.001f;

        Add( New<sVLine>( std::vector<double>{ lEventSpanStart, lEventSpanEnd } ) );
        Add( New<sVLine>( std::vector<double>{ lElementPosition } ) );

        auto lPreviousRbs        = lAttributes.GetPropertyValue( "PreviousRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
        auto lPreviousRbsLsaData = lPreviousRbs->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );
        auto lPreviousRbsSlope   = lPreviousRbsLsaData->GetPropertyValue<double>( "Slope" );
        auto lPreviousRbsOffset  = lPreviousRbsLsaData->GetPropertyValue<double>( "Offset" );
        auto lPreviousRbsPlot    = New<sFloat64LinePlot>();
        auto lPreviousX0         = lPreviousRbsLsaData->GetPropertyValue<double>( "StartPosition" );
        auto lPreviousX1         = lPreviousRbsLsaData->GetPropertyValue<double>( "EndPosition" );
        lPreviousRbsPlot->mX     = std::vector<double>{ lPreviousX0 * 0.001, lPreviousX1 * 0.001 };
        lPreviousRbsPlot->mY =
            std::vector<double>{ lPreviousRbsOffset, ( lPreviousX1 - lPreviousX0 ) * lPreviousRbsSlope + lPreviousRbsOffset };
        Add( lPreviousRbsPlot );

        auto lNextRbs        = lAttributes.GetPropertyValue( "NextRbs", "Metrino.Olm.SignalProcessing.RbsAttribute" );
        auto lNextRbsLsaData = lNextRbs->GetPropertyValue( "Lsa", "Metrino.Olm.SignalProcessing.RbsLsa" );
        auto lNextRbsSlope   = lNextRbsLsaData->GetPropertyValue<double>( "Slope" );
        auto lNextRbsOffset  = lNextRbsLsaData->GetPropertyValue<double>( "Offset" );
        auto lNextRbsPlot    = New<sFloat64LinePlot>();
        auto lNextX0         = lNextRbsLsaData->GetPropertyValue<double>( "StartPosition" );
        auto lNextX1         = lNextRbsLsaData->GetPropertyValue<double>( "EndPosition" );
        lNextRbsPlot->mX     = std::vector<double>{ lNextX0 * 0.001, lNextX1 * 0.001 };
        lNextRbsPlot->mY     = std::vector<double>{ lNextRbsOffset, ( lNextX1 - lNextX0 ) * lNextRbsSlope + lNextRbsOffset };
        Add( lNextRbsPlot );
    }

    void UILinkElementTracePlot::SetEventData( std::vector<sLinkElement> &aLinkElement )
    {
        static auto &lSinglePulseTraceClass = MonoRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );

        Clear();

        for( int i = 0; i < aLinkElement.size(); i++ )
        {
            auto lTrace = MonoScriptInstance( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), aLinkElement[i].mPeakTrace );
            if( lTrace.GetInstance() == nullptr ) continue;

            auto lSamples = lTrace.GetPropertyValue<MonoObject *>( "Samples" );
            auto lDeltaX  = lTrace.GetPropertyValue<double>( "SamplingPeriod" );
            auto lPlot    = New<sFloat64LinePlot>();

            lPlot->mY      = AsVector<double>( lSamples );
            lPlot->mX      = std::vector<double>( lPlot->mY.size() );
            lPlot->mLegend = fmt::format( "{:.0f} nm - {} ({} samples)", lTrace.GetPropertyValue<double>( "Wavelength" ) * 1e9, i,
                                          lPlot->mX.size() );

            for( uint32_t i = 0; i < lPlot->mX.size(); i++ ) lPlot->mX[i] = i * lDeltaX;

            Add( lPlot );
        }
    }

} // namespace SE::OtdrEditor