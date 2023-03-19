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

    void UILinkElementTracePlot::SetEventData( sLinkElement &lEventDataVector )
    {
        static auto &lTraceDataStructure = MonoRuntime::GetClassType( "Metrino.Interop.TracePlotData" );

        Clear();

        static auto &lSinglePulseTraceClass = MonoRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );
        auto lTrace   = MonoScriptInstance( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), lEventDataVector.mPeakTrace );
        auto lSamples = lTrace.GetPropertyValue<MonoObject *>( "Samples" );
        auto lDeltaX  = lTrace.GetPropertyValue<double>( "SamplingPeriod" );
        auto lPlot    = New<sFloat64LinePlot>();

        lPlot->mY = AsVector<double>( lSamples );
        lPlot->mX = std::vector<double>( lPlot->mY.size() );
        for( uint32_t i = 0; i < lPlot->mX.size(); i++ ) lPlot->mX[i] = i * lDeltaX;

        Add( lPlot );
    }

    void UILinkElementTracePlot::SetEventData( std::vector<sLinkElement> &lEventDataVector )
    {
        static auto &lTraceDataStructure    = MonoRuntime::GetClassType( "Metrino.Interop.TracePlotData" );
        static auto &lSinglePulseTraceClass = MonoRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );

        Clear();

        for( int i = 0; i < lEventDataVector.size(); i++ )
        {
            auto lTrace =
                MonoScriptInstance( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), lEventDataVector[i].mPeakTrace );
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

            // auto lTrace        = lEventDataVector[i].mPhysicalEvent->GetPropertyValue( "PeakTrace", "Metrino.Otdr.SinglePulseTrace"
            // ); auto lSamples      = lTrace->GetPropertyValue<MonoObject *>( "Samples" ); auto lSampleVector = AsVector<double>(
            // lSamples );

            // // auto lInstance = MonoScriptInstance( &lTraceDataStructure, lTraceDataStructure.Class(), lTraceDataVector[i] );
            // auto lPlot = New<sFloat64LinePlot>();
            // lPlot->mX      = AsVector<double>( lInstance.GetFieldValue<MonoObject *>( "mX" ) );
            // lPlot->mY      = AsVector<double>( lInstance.GetFieldValue<MonoObject *>( "mY" ) );
            // lPlot->mLegend = fmt::format( "{:.0f} nm - {} ({} samples)", lInstance.GetFieldValue<double>( "mWavelength" ) * 1e9, i,
            //                               lPlot->mX.size() );

            // Add( lPlot );
        }
    }

} // namespace SE::OtdrEditor