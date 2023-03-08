#pragma once

#include <string>
#include <vector>

namespace SE::OtdrEditor
{
    enum class eEventType : int32_t
    {
        Unknown         = 0,
        PositiveSplice  = 1,
        NegativeSplice  = 2,
        Reflection      = 3,
        EndOfAnalysis   = 4,
        ContinuousFiber = 5
    };
    std::string ToString( eEventType aType );

    enum class eEventStatus : int32_t
    {
        None                      = 0,
        Echo                      = 1,
        PossibleEcho              = 2,
        EndOfFiber                = 4,
        LaunchLevel               = 8,
        Saturated                 = 16,
        AddedByUser               = 32,
        SpanStart                 = 64,
        SpanEnd                   = 128,
        NewWhileTemplating        = 256,
        AddedForSpan              = 512,
        AddedFromReference        = 1024,
        Bidir                     = 2048,
        Splitter                  = 1 << 12, // 4096
        PreviousSectionEcho       = 1 << 13, // 8192
        UnderEstimatedLoss        = 1 << 14, // 16384
        UnderEstimatedReflectance = 1 << 15, // 32768
        LoopStart                 = 1 << 16, // 65536
        LoopEnd                   = 1 << 17, // 131072
        CouplerPort               = 1 << 18, // 262144
        Reference                 = 1 << 19,
        OverEstimatedReflectance  = 1 << 20,
        InjectionReference        = 1 << 21,
        OverEstimatedLoss         = 1 << 22 //
    };
    std::vector<std::string> ToString( int32_t aStatus );

    enum class eReflectanceType : int32_t
    {
        Bidirectional          = 0,
        UnidirectionalForward  = 1,
        UnidirectionalBackward = 2
    };
    std::string ToString( eReflectanceType aStatus );

    enum class eLinkElementStatus : int32_t
    {
        None                   = 0,
        OtdrConnectorFail      = 1,
        LinkStart              = 2,
        LinkEnd                = 4,
        EndOfFiber             = 8,
        NotOnLink              = 16,
        AddedByUser            = 32,
        SpecifiedSplitterRatio = 64,
        AddedForMacrobend      = 128,
        LoopStart              = 256,
        LoopEnd                = 512,
        SignatureStart         = 1024,
        SignatureEnd           = 2048,
        NewMonitoringEvent     = 4096,
        InternalEvent          = 8192
    };

    std::vector<std::string> LinkStatusToString( int32_t aStatus );

    enum class eLinkElementType : int32_t
    {
        Unknown            = 0,
        Connector          = 1,
        Splice             = 2,
        Splitter           = 3,
        OutOfRange         = 4,
        Macrobend          = 5,
        CouplerPort        = 6,
        Switch             = 7,
        Tam                = 8,
        Unexpected         = 9,
        Coupler            = 10,
        TestEquipment      = 11,
        UnbalancedSplitter = 12
    };
    std::string ToString( eLinkElementType aStatus );

    enum class ePassFail : int32_t
    {
        NONE,
        RISK,
        PASS,
        PASS_WITH_RISK,
        FAIL,
        FAIL_WITH_RISK
    };

} // namespace SE::OtdrEditor