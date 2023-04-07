#include "Enums.h"

namespace SE::OtdrEditor
{
    std::string ToString( eEventType aType )
    {
        switch( aType )
        {
        case eEventType::Unknown: return "Unknown";
        case eEventType::PositiveSplice: return "Positive Splice";
        case eEventType::NegativeSplice: return "Negative Splice";
        case eEventType::Reflection: return "Reflection";
        case eEventType::EndOfAnalysis: return "End Of Analysis";
        case eEventType::ContinuousFiber: return "Continuous Fiber";
        default: return "N/A";
        }
    }

    std::vector<std::string> ToString( int32_t aStatus )
    {
        std::vector<std::string> lResult;

        if( aStatus & static_cast<int32_t>( eEventStatus::None ) ) lResult.emplace_back( "None" );
        if( aStatus & static_cast<int32_t>( eEventStatus::Echo ) ) lResult.emplace_back( "Echo" );
        if( aStatus & static_cast<int32_t>( eEventStatus::PossibleEcho ) ) lResult.emplace_back( "Possible Echo" );
        if( aStatus & static_cast<int32_t>( eEventStatus::EndOfFiber ) ) lResult.emplace_back( "EOF" );
        if( aStatus & static_cast<int32_t>( eEventStatus::LaunchLevel ) ) lResult.emplace_back( "Launch Level" );
        if( aStatus & static_cast<int32_t>( eEventStatus::Saturated ) ) lResult.emplace_back( "Saturated" );
        if( aStatus & static_cast<int32_t>( eEventStatus::AddedByUser ) ) lResult.emplace_back( "AddedByUser" );
        if( aStatus & static_cast<int32_t>( eEventStatus::SpanStart ) ) lResult.emplace_back( "SpanStart" );
        if( aStatus & static_cast<int32_t>( eEventStatus::SpanEnd ) ) lResult.emplace_back( "SpanEnd" );
        if( aStatus & static_cast<int32_t>( eEventStatus::NewWhileTemplating ) ) lResult.emplace_back( "NewWhileTemplating" );
        if( aStatus & static_cast<int32_t>( eEventStatus::AddedForSpan ) ) lResult.emplace_back( "AddedForSpan" );
        if( aStatus & static_cast<int32_t>( eEventStatus::AddedFromReference ) ) lResult.emplace_back( "AddedFromReference" );
        if( aStatus & static_cast<int32_t>( eEventStatus::Bidir ) ) lResult.emplace_back( "Bidir" );
        if( aStatus & static_cast<int32_t>( eEventStatus::Splitter ) ) lResult.emplace_back( "Splitter" );
        if( aStatus & static_cast<int32_t>( eEventStatus::PreviousSectionEcho ) ) lResult.emplace_back( "PreviousSectionEcho" );
        if( aStatus & static_cast<int32_t>( eEventStatus::UnderEstimatedLoss ) ) lResult.emplace_back( "UnderEstimatedLoss" );
        if( aStatus & static_cast<int32_t>( eEventStatus::UnderEstimatedReflectance ) )
            lResult.emplace_back( "UnderEstimatedReflectance" );
        if( aStatus & static_cast<int32_t>( eEventStatus::LoopStart ) ) lResult.emplace_back( "LoopStart" );
        if( aStatus & static_cast<int32_t>( eEventStatus::LoopEnd ) ) lResult.emplace_back( "LoopEnd" );
        if( aStatus & static_cast<int32_t>( eEventStatus::CouplerPort ) ) lResult.emplace_back( "CouplerPort" );
        if( aStatus & static_cast<int32_t>( eEventStatus::Reference ) ) lResult.emplace_back( "Reference" );
        if( aStatus & static_cast<int32_t>( eEventStatus::OverEstimatedReflectance ) )
            lResult.emplace_back( "OverEstimatedReflectance" );
        if( aStatus & static_cast<int32_t>( eEventStatus::InjectionReference ) ) lResult.emplace_back( "InjectionReference" );
        if( aStatus & static_cast<int32_t>( eEventStatus::OverEstimatedLoss ) ) lResult.emplace_back( "OverEstimatedLoss" );

        return lResult;
    }

    std::string ToString( eReflectanceType aReflectanceType )
    {
        switch( aReflectanceType )
        {
        case eReflectanceType::Bidirectional: return "Bidirectional"; break;
        case eReflectanceType::UnidirectionalForward: return "Unidirectional Forward"; break;
        case eReflectanceType::UnidirectionalBackward: return "Unidirectional Backward"; break;
        default: return "N/A"; break;
        }
    }

    std::vector<std::string> LinkStatusToString( int32_t aStatus )
    {
        std::vector<std::string> lResult;

        if( aStatus & static_cast<int32_t>( eLinkElementStatus::None ) ) lResult.emplace_back( "None" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::OtdrConnectorFail ) ) lResult.emplace_back( "OtdrConnectorFail" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::LinkStart ) ) lResult.emplace_back( "LinkStart" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::LinkEnd ) ) lResult.emplace_back( "LinkEnd" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::EndOfFiber ) ) lResult.emplace_back( "EndOfFiber" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::NotOnLink ) ) lResult.emplace_back( "NotOnLink" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::AddedByUser ) ) lResult.emplace_back( "AddedByUser" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::SpecifiedSplitterRatio ) )
            lResult.emplace_back( "SpecifiedSplitterRatio" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::AddedForMacrobend ) ) lResult.emplace_back( "AddedForMacrobend" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::LoopStart ) ) lResult.emplace_back( "LoopStart" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::LoopEnd ) ) lResult.emplace_back( "LoopEnd" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::SignatureStart ) ) lResult.emplace_back( "SignatureStart" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::SignatureEnd ) ) lResult.emplace_back( "SignatureEnd" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::NewMonitoringEvent ) ) lResult.emplace_back( "NewMonitoringEvent" );
        if( aStatus & static_cast<int32_t>( eLinkElementStatus::InternalEvent ) ) lResult.emplace_back( "InternalEvent" );

        return lResult;
    }

    std::string ToString( eLinkElementType aReflectanceType )
    {
        switch( aReflectanceType )
        {
        case eLinkElementType::Unknown: return "Unknown"; break;
        case eLinkElementType::Connector: return "Connector"; break;
        case eLinkElementType::Splice: return "Splice"; break;
        case eLinkElementType::Splitter: return "Splitter"; break;
        case eLinkElementType::OutOfRange: return "OutOfRange"; break;
        case eLinkElementType::Macrobend: return "Macrobend"; break;
        case eLinkElementType::CouplerPort: return "CouplerPort"; break;
        case eLinkElementType::Switch: return "Switch"; break;
        case eLinkElementType::Tam: return "Tam"; break;
        case eLinkElementType::Unexpected: return "Unexpected"; break;
        case eLinkElementType::Coupler: return "Coupler"; break;
        case eLinkElementType::TestEquipment: return "TestEquipment"; break;
        case eLinkElementType::UnbalancedSplitter: return "UnbalancedSplitter"; break;
        default: return "N/A"; break;
        }
    }

} // namespace SE::OtdrEditor