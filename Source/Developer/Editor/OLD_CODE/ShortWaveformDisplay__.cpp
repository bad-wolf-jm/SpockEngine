#include "ShortWaveformDisplay.h"

#include "Developer/GraphicContext/GraphicContext.h"
#include "Developer/UI/UI.h"

using namespace LTSE::Core;

namespace LTSE::Editor
{
    // clang-format off
    static const char* lHexCodes[] = {
        "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "0A", "00", "0C", "0D", "0E", "0F",
        "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "1A", "11", "1C", "1D", "1E", "1F",
        "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "2A", "22", "2C", "2D", "2E", "2F",
        "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "3A", "33", "3C", "3D", "3E", "3F",
        "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "4A", "44", "4C", "4D", "4E", "4F",
        "50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "5A", "55", "5C", "5D", "5E", "5F",
        "60", "61", "62", "63", "64", "65", "66", "67", "68", "69", "6A", "66", "6C", "6D", "6E", "6F",
        "70", "71", "72", "73", "74", "75", "76", "77", "78", "79", "7A", "77", "7C", "7D", "7E", "7F",
        "80", "81", "82", "83", "84", "85", "86", "87", "88", "89", "8A", "88", "8C", "8D", "8E", "8F",
        "90", "91", "92", "93", "94", "95", "96", "97", "98", "99", "9A", "99", "9C", "9D", "9E", "9F",
        "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "AA", "AA", "AC", "AD", "AE", "AF",
        "B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "BA", "BB", "BC", "BD", "BE", "BF",
        "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CA", "CC", "CC", "CD", "CE", "CF",
        "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "CA", "CC", "CC", "CD", "CE", "CF",
        "E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "EA", "EE", "EC", "ED", "EE", "EF",
        "F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "FA", "FF", "FC", "FD", "FE", "FF",
    };
    // clang-format on

    inline void DisplayU32( uint8_t *aData )
    {
        char lString[9];
        sprintf( lString, "%s%s%s%s", lHexCodes[aData[0]], lHexCodes[aData[1]], lHexCodes[aData[2]], lHexCodes[aData[3]] );
        UI::Text( lString );
    }
    inline void DisplayU16( uint8_t *aData )
    {
        char lString[9];
        sprintf( lString, "%s%s", lHexCodes[aData[0]], lHexCodes[aData[1]] );
        UI::Text( lString );
    }

    inline void UnpackHeader( uint32_t aHeader, uint32_t &aID, uint32_t &aVersion, uint32_t &aDetectionCount, uint32_t &aMaxDetectionCount, uint32_t &aSampleCount )
    {
        aID                = ( aHeader >> 0 ) & 0x0000000F;
        aVersion           = ( aHeader >> 4 ) & 0x0000000F;
        aMaxDetectionCount = ( aHeader >> 8 ) & 0x0000003F;
        aSampleCount       = ( aHeader >> 14 ) & 0x00000FFF;
        aDetectionCount    = ( aHeader >> 26 ) & 0x0000003F;
    }

    inline void UnpackIDValues( uint32_t aHeader, uint32_t &aConfigurationID, uint32_t &aFrameID, uint32_t &aOpticalID, uint32_t &aAcquisitionID )
    {
        aConfigurationID = ( aHeader & 0xF0000000 ) >> 28;
        aFrameID         = ( aHeader & 0x0F000000 ) >> 24;
        aOpticalID       = ( aHeader & 0x00FC0000 ) >> 18;
        aAcquisitionID   = ( aHeader & 0x0003FFC0 ) >> 6;
    }

    inline void UnpackPDLaserAngle( uint32_t aHeader, uint32_t &aPD, uint32_t &aLaserAngle, uint32_t &aIsRemoved, uint32_t &aFrameNumber )
    {
        aIsRemoved   = ( aHeader >> 31 ) & 0x00000001;
        aLaserAngle  = ( aHeader >> 22 ) & 0x000001FF;
        aPD          = ( aHeader >> 16 ) & 0x0000003F;
        aFrameNumber = ( aHeader >> 0 ) & 0x0000FFFF;
    }

    inline void UnpackBaselineNoise( uint32_t aHeader, uint32_t &aBaseline, uint32_t &aNoise )
    {
        aBaseline = ( aHeader >> 16 ) & 0xFFFF;
        aNoise    = ( aHeader >> 0 ) & 0xFFFF;
    }

    inline void UnpackDetectionDistance( uint32_t aHeader, float &aDistance, int32_t &aLastUnsaturatedSample )
    {
        aDistance              = static_cast<float>( aHeader & 0x7FFF ) / std::pow( 2.0f, 6.0f );
        aLastUnsaturatedSample = static_cast<int32_t>( ( aHeader >> 16 ) & 0xFFFF );
    }

    inline void UnpackBaselines( uint32_t aHeader, int32_t &aBaselineBeforeSaturation, int32_t &aBaselineAfterSaturation )
    {
        aBaselineBeforeSaturation = static_cast<int32_t>( static_cast<int16_t>( aHeader & 0x0000FFFF ) );
        aBaselineAfterSaturation  = static_cast<int32_t>( static_cast<int16_t>( ( aHeader >> 16 ) & 0x0000FFFF ) );
    }

    inline void UnpackDetectionInfo( uint32_t aHeader, uint8_t &aPulseIsSaturated, uint32_t &aDetectionIndex, uint32_t &aTraceOffset )
    {
        aTraceOffset      = static_cast<int32_t>( ( aHeader >> 16 ) & 0xFFFF );
        aPulseIsSaturated = static_cast<int32_t>( ( aHeader >> 15 ) & 0x0001 );
        aDetectionIndex   = static_cast<int32_t>( ( aHeader >> 0 ) & 0x7FFF );
    }

    void DisplayShortWaveforms( UIContext &aUiContext, std::vector<uint8_t> &aTileData )
    {
        auto l_DrawList   = ImGui::GetWindowDrawList();
        auto l_WindowSize = UI::GetAvailableContentSpace();
        auto l_TopLeft    = ImGui::GetCursorScreenPos();

        aUiContext.PushFontFamily( { FontFamilyFlags::MONO } );

        uint32_t lPacketSize = 320;

        uint32_t lByteIndex = 0;
        uint32_t lLine      = 1;

        auto lByteSize       = ImGui::CalcTextSize( lHexCodes[0] ).x;
        auto lU32Size        = lByteSize * 4.0f + ( 3.0f * 7.5f );
        auto lLineNumberSize = ImGui::CalcTextSize( "9999" ).x;

        uint32_t lRows = aTileData.size() / lPacketSize;

        ImGuiListClipper lClipper;
        lClipper.Begin( lRows );
        while( lClipper.Step() )
        {
            for( uint32_t lR = lClipper.DisplayStart; lR < lClipper.DisplayEnd; lR++ )
            {
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4( .3f, .3f, .3f, 1.0f ) );
                UI::Text( "{:>4}", lR + 1 );
                ImGui::PopStyleColor();
                UI::SameLine( 35.0f );

                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.8f, 0.1f, 0.25f, 1.0f } );
                uint32_t lShortWaveformStart = lR * lPacketSize;
                auto lHeader0                = *(uint32_t *)&( aTileData.data()[lShortWaveformStart + 0] );
                auto lHeader1                = *(uint32_t *)&( aTileData.data()[lShortWaveformStart + 4] );
                auto lHeader2                = *(uint32_t *)&( aTileData.data()[lShortWaveformStart + 8] );
                auto lHeader3                = *(uint32_t *)&( aTileData.data()[lShortWaveformStart + 12] );
                auto lHeader4                = *(uint32_t *)&( aTileData.data()[lShortWaveformStart + 16] );
                uint32_t lID, lVersion, lDetectionCount, lMaxDetectionCount, lSampleCount;
                UnpackHeader( lHeader0, lID, lVersion, lDetectionCount, lMaxDetectionCount, lSampleCount );
                uint32_t lPD, lLaserAngle, lHasLeftovers, lFrameIndex;
                UnpackPDLaserAngle( lHeader1, lPD, lLaserAngle, lHasLeftovers, lFrameIndex );
                uint32_t lConfigurationID, lFrameID, lOpticalID, lAcquisitionID;
                UnpackIDValues( lHeader3, lConfigurationID, lFrameID, lOpticalID, lAcquisitionID );
                uint32_t lBaseline, lNoise;
                UnpackBaselineNoise( lHeader4, lBaseline, lNoise );

                UI::Text( "{}", ICON_FA_INFO_CIRCLE );
                if( ImGui::IsItemHovered() )
                {
                    ImGui::BeginTooltip();
                    UI::Text( "ID: {}", lID );
                    UI::Text( "Version: {}", lVersion );
                    UI::Text( "Detection count: {}", lDetectionCount );
                    UI::Text( "Max detection count: {}", lMaxDetectionCount );
                    UI::Text( "Sample count: {}", lSampleCount );
                    UI::Text( "LCA3 channel: {}", lPD );
                    UI::Text( "LaserAngle: {}", lLaserAngle );
                    UI::Text( "Leftover detections: {}", lHasLeftovers );
                    UI::Text( "Frame number: {}", lFrameIndex );
                    UI::Text( "Timestamp: {}", lHeader2 );
                    UI::Text( "Configuration ID: {}", lConfigurationID );
                    UI::Text( "Frame ID: {}", lFrameID );
                    UI::Text( "Optical ID: {}", lOpticalID );
                    UI::Text( "Acquisition ID: {}", lAcquisitionID );
                    UI::Text( "Baseline: {}", lBaseline );
                    UI::Text( "Noise: {}", lNoise );
                    ImGui::EndTooltip();
                }
                UI::SameLine();

                DisplayU32( &( aTileData.data()[lShortWaveformStart + 0] ) );
                if( ImGui::IsItemHovered() )
                {
                    ImGui::BeginTooltip();
                    UI::Text( "ID: {}", lID );
                    UI::Text( "Version: {}", lVersion );
                    UI::Text( "Detection count: {}", lDetectionCount );
                    UI::Text( "Max detection count: {}", lMaxDetectionCount );
                    UI::Text( "sample count: {}", lSampleCount );
                    ImGui::EndTooltip();
                }
                UI::SameLine();

                DisplayU32( &( aTileData.data()[lShortWaveformStart + 4] ) );
                if( ImGui::IsItemHovered() )
                {
                    ImGui::BeginTooltip();
                    UI::Text( "PD: {}", lPD );
                    UI::Text( "LaserAngle: {}", lLaserAngle );
                    UI::Text( "Leftover detections: {}", lHasLeftovers );
                    UI::Text( "Frame number: {}", lFrameIndex );
                    ImGui::EndTooltip();
                }
                UI::SameLine();

                DisplayU32( &( aTileData.data()[lShortWaveformStart + 8] ) );
                if( ImGui::IsItemHovered() )
                {
                    ImGui::BeginTooltip();
                    UI::Text( "Timestamp: {}", lHeader2 );
                    ImGui::EndTooltip();
                }
                UI::SameLine();

                DisplayU32( &( aTileData.data()[lShortWaveformStart + 12] ) );
                if( ImGui::IsItemHovered() )
                {
                    ImGui::BeginTooltip();
                    UI::Text( "Configuration ID: {}", lConfigurationID );
                    UI::Text( "Frame ID: {}", lFrameID );
                    UI::Text( "Optical ID: {}", lOpticalID );
                    UI::Text( "Acquisition ID: {}", lAcquisitionID );
                    ImGui::EndTooltip();
                }
                UI::SameLine();

                DisplayU32( &( aTileData.data()[lShortWaveformStart + 16] ) );
                if( ImGui::IsItemHovered() )
                {
                    ImGui::BeginTooltip();
                    UI::Text( "Baseline: {}", lBaseline );
                    UI::Text( "Noise: {}", lNoise );
                    ImGui::EndTooltip();
                }
                UI::SameLine( 20.0f );
                ImGui::PopStyleColor();

                uint32_t lDetectionStart = lShortWaveformStart + 20;
                auto lPos                = ImGui::GetCursorPosX();
                for( uint32_t i = 0; i < 5; i++ )
                {
                    // Stack detections vertically
                    auto lDetectionPos = ImGui::GetCursorPosY();
                    ImGui::SetCursorPos( ImVec2{ lPos, lDetectionPos } );

                    auto lValue0 = *(uint32_t *)&( aTileData.data()[lDetectionStart + 0] );
                    auto lValue1 = *(uint32_t *)&( aTileData.data()[lDetectionStart + 4] );
                    auto lValue2 = *(uint32_t *)&( aTileData.data()[lDetectionStart + 8] );
                    auto lValue3 = *(uint32_t *)&( aTileData.data()[lDetectionStart + 12] );

                    float lDistance;
                    int32_t lLastUnsaturatedSample;
                    UnpackDetectionDistance( lValue0, lDistance, lLastUnsaturatedSample );

                    int32_t lBaselineBefore;
                    int32_t lBaselineAfter;
                    UnpackBaselines( lValue1, lBaselineBefore, lBaselineAfter );

                    uint8_t lPulseIsSaturated;
                    uint32_t lDetectionIndex;
                    uint32_t lDetectionOffset;
                    UnpackDetectionInfo( lValue3, lPulseIsSaturated, lDetectionIndex, lDetectionOffset );

                    float lDimmedColor = 1.0f;
                    if( *(uint32_t *)&( aTileData.data()[lDetectionStart + 0] ) == 0 )
                        lDimmedColor = .1f;

                    if( lPulseIsSaturated )
                        ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 1.0f, 0.0f, 0.1f, 1.0f } );
                    else
                        ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.10f, 0.82f, 0.10f, lDimmedColor } );

                    UI::Text( "{}", ICON_FA_INFO_CIRCLE );
                    if( ImGui::IsItemHovered() )
                    {
                        ImGui::BeginTooltip();
                        UI::Text( "Distance: {}", lDistance );
                        UI::Text( "Last unsaturated sample: {}", lLastUnsaturatedSample );
                        UI::Text( "Baseline before saturation: {}", lBaselineBefore );
                        UI::Text( "Baseline after saturation: {}", lBaselineAfter );
                        if( lPulseIsSaturated )
                            UI::Text( "Saturation length: {}", lValue2 );
                        else
                            UI::Text( "Amplitude: {}", static_cast<int32_t>(lValue2) );
                        UI::Text( "Pulse is saturated: {}", lPulseIsSaturated );
                        UI::Text( "Detection index: {}", lDetectionIndex );
                        UI::Text( "Trace offset: {}", lDetectionOffset );

                        ImGui::EndTooltip();
                    }

                    UI::SameLine();
                    ImGui::PopStyleColor();

                    ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.10f, 0.82f, 0.10f, lDimmedColor } );

                    DisplayU32( &( aTileData.data()[lDetectionStart + 0] ) );
                    if( ImGui::IsItemHovered() )
                    {
                        ImGui::BeginTooltip();
                        UI::Text( "Distance: {}", lDistance );
                        UI::Text( "Last unsaturated sample: {}", lLastUnsaturatedSample );
                        ImGui::EndTooltip();
                    }
                    UI::SameLine();

                    DisplayU32( &( aTileData.data()[lDetectionStart + 4] ) );
                    if( ImGui::IsItemHovered() )
                    {
                        ImGui::BeginTooltip();
                        UI::Text( "Baseline before saturation: {}", lBaselineBefore );
                        UI::Text( "Baseline after saturation: {}", lBaselineAfter );
                        ImGui::EndTooltip();
                    }
                    UI::SameLine();

                    DisplayU32( &( aTileData.data()[lDetectionStart + 8] ) );
                    if( ImGui::IsItemHovered() )
                    {
                        ImGui::BeginTooltip();
                        if( lPulseIsSaturated )
                            UI::Text( "Saturation length: {}", lValue2 );
                        else
                            UI::Text( "Amplitude: {}", static_cast<int32_t>(lValue2) );
                        ImGui::EndTooltip();
                    }
                    UI::SameLine();

                    DisplayU32( &( aTileData.data()[lDetectionStart + 12] ) );
                    if( ImGui::IsItemHovered() )
                    {
                        ImGui::BeginTooltip();
                        UI::Text( "Pulse is saturated: {}", lPulseIsSaturated );
                        UI::Text( "Detection index: {}", lDetectionIndex );
                        UI::Text( "Trace offset: {}", lDetectionOffset );
                        ImGui::EndTooltip();
                    }
                    UI::SameLine( 20.0f );
                    ImGui::PopStyleColor();

                    ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.30f, 0.32f, 0.10f, lDimmedColor } );
                    uint32_t lTraceStart = lDetectionStart + 16;
                    UI::Text( "{}", ICON_FA_LINE_CHART );
                    if( ImGui::IsItemHovered() )
                    {
                        ImGui::BeginTooltip();
                        float lValues0[11];
                        float lMinValue  = 0;
                        float lMaxValue  = 100;
                        int16_t *lValues = (int16_t *)&( aTileData.data()[lTraceStart] );
                        for( uint32_t j = 0; j < 11; j++ )
                        {
                            lValues0[j] = static_cast<float>( lValues[j] );
                            lMinValue   = std::min( lMinValue, lValues0[j] );
                            lMaxValue   = std::max( lMaxValue, lValues0[j] );
                        }
                        lMaxValue += ( lMaxValue - lMinValue ) * 0.05f;
                        lMinValue -= ( lMaxValue - lMinValue ) * 0.05f;

                        ImGui::PushID( i );
                        ImPlot::PushStyleVar( ImPlotStyleVar_PlotPadding, ImVec2( 0, 0 ) );
                        if( ImPlot::BeginPlot( "##spark_0", ImVec2( 500.0f, 150.0f ), ImPlotFlags_Crosshairs | ImPlotFlags_NoChild ) )
                        {
                            ImPlot::SetupAxes( 0, 0, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_None );
                            ImPlot::SetupAxesLimits( 0, 11 - 1, lMinValue, lMaxValue, ImGuiCond_Always );
                            ImPlot::PushStyleColor( ImPlotCol_Line, ImPlot::GetColormapColor( 0 ) );
                            ImPlot::PlotLine( "##spark_0", lValues0, 11, 1, 0, 0 );
                            ImPlot::PopStyleColor();
                            ImPlot::EndPlot();
                        }
                        ImPlot::PopStyleVar();
                        ImGui::PopID();
                        ImGui::EndTooltip();
                    }
                    UI::SameLine();

                    for( uint32_t j = 0; j < 11; j++ )
                    {
                        DisplayU16( &( aTileData.data()[lTraceStart + j * sizeof( uint16_t )] ) );
                        if( j < 10 )
                            UI::SameLine();
                        else
                            UI::SameLine( 20.0f );
                    }
                    ImGui::PopStyleColor();

                    ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 0.22f, 0.22f, 0.30f, lDimmedColor } );
                    uint32_t lThresholdStart = lTraceStart + 11 * sizeof( uint16_t );
                    UI::Text( "{}", ICON_FA_LINE_CHART );
                    if( ImGui::IsItemHovered() )
                    {
                        ImGui::BeginTooltip();
                        float lValues0[11];
                        float lMinValue  = 0;
                        float lMaxValue  = 100;
                        int16_t *lValues = (int16_t *)&( aTileData.data()[lThresholdStart] );
                        for( uint32_t j = 0; j < 11; j++ )
                        {
                            lValues0[j] = static_cast<float>( lValues[j] );
                            lMinValue   = std::min( lMinValue, lValues0[j] );
                            lMaxValue   = std::max( lMaxValue, lValues0[j] );
                        }
                        lMaxValue += ( lMaxValue - lMinValue ) * 0.05f;
                        lMinValue -= ( lMaxValue - lMinValue ) * 0.05f;

                        ImGui::PushID( i );
                        ImPlot::PushStyleVar( ImPlotStyleVar_PlotPadding, ImVec2( 0, 0 ) );
                        if( ImPlot::BeginPlot( "##spark_0", ImVec2( 500.0f, 150.0f ), ImPlotFlags_Crosshairs | ImPlotFlags_NoChild ) )
                        {
                            ImPlot::SetupAxes( 0, 0, ImPlotAxisFlags_NoDecorations, ImPlotAxisFlags_None );
                            ImPlot::SetupAxesLimits( 0, 11 - 1, lMinValue, lMaxValue, ImGuiCond_Always );
                            ImPlot::PushStyleColor( ImPlotCol_Line, ImPlot::GetColormapColor( 0 ) );
                            ImPlot::PlotLine( "##spark_0", lValues0, 11, 1, 0, 0 );
                            ImPlot::PopStyleColor();
                            ImPlot::EndPlot();
                        }
                        ImPlot::PopStyleVar();
                        ImGui::PopID();
                        ImGui::EndTooltip();
                    }
                    UI::SameLine();
                    for( uint32_t j = 0; j < 11; j++ )
                    {
                        DisplayU16( &( aTileData.data()[lThresholdStart + j * sizeof( uint16_t )] ) );

                        if( j < 10 )
                            UI::SameLine();
                    }
                    ImGui::PopStyleColor();
                    lDetectionStart += ( 4 * sizeof( uint32_t ) + 22 * sizeof( uint16_t ) );
                }
            }
        }
        aUiContext.PopFont();
    }

} // namespace LTSE::Editor