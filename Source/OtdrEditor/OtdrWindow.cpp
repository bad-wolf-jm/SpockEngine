#include "OtdrWindow.h"

#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>

#include "Core/Profiling/BlockTimer.h"

#include "UI/Widgets.h"

#include "Scene/Components.h"
#include "Scene/Importer/ObjImporter.h"
#include "Scene/Importer/glTFImporter.h"

#include "Core/CUDA/Texture/TextureData.h"
#include "Core/File.h"
#include "Core/Logging.h"

#include "Scene/Components.h"
#include "Scene/Serialize/AssetFile.h"
#include "Scene/Serialize/FileIO.h"

#include "Mono/MonoRuntime.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

namespace SE::OtdrEditor
{

    using namespace SE::Core;
    using namespace SE::Core::EntityComponentSystem::Components;

    static std::string UTF16ToAscii( const char *aPayloadData, size_t aSize )
    {
        size_t      lPayloadSize = static_cast<size_t>( aSize / 2 );
        std::string lItemPathStr( lPayloadSize - 1, '\0' );
        for( uint32_t i = 0; i < lPayloadSize - 1; i++ ) lItemPathStr[i] = aPayloadData[2 * i];

        return lItemPathStr;
    }

    void OtdrWindow::ConfigureUI()
    {
        mWorkspaceArea.ConfigureUI();

        mEventTable.SetRowHeight( 20.0f );

        mPositionColumn               = New<sFloat64Column>();
        mPositionColumn->mHeader      = "Position";
        mPositionColumn->mInitialSize = 75.0f;
        mPositionColumn->mFormat      = "{:.3f} km";
        mPositionColumn->mNaNFormat   = "\xe2\x88\x9e km";
        mEventTable.AddColumn( mPositionColumn );

        mLossColumn               = New<sFloat64Column>();
        mLossColumn->mHeader      = "Loss";
        mLossColumn->mInitialSize = 75.0f;
        mLossColumn->mFormat      = "{:.1f} dB";
        mLossColumn->mNaNFormat   = "-\xe2\x88\x9e dB";
        mEventTable.AddColumn( mLossColumn );

        mEstimatedLossColumn               = New<sFloat64Column>();
        mEstimatedLossColumn->mHeader      = "Est. Loss";
        mEstimatedLossColumn->mInitialSize = 75.0f;
        mEstimatedLossColumn->mFormat      = "{:.1f} dB";
        mEstimatedLossColumn->mNaNFormat   = "-\xe2\x88\x9e dB";
        mEventTable.AddColumn( mEstimatedLossColumn );

        mReflectanceColumn               = New<sFloat64Column>();
        mReflectanceColumn->mHeader      = "Reflectance";
        mReflectanceColumn->mInitialSize = 75.0f;
        mReflectanceColumn->mFormat      = "{:.1f} dB";
        mReflectanceColumn->mNaNFormat   = "-\xe2\x88\x9e dB";
        mEventTable.AddColumn( mReflectanceColumn );

        mWavelengthColumn               = New<sFloat64Column>();
        mWavelengthColumn->mHeader      = "Wavelength";
        mWavelengthColumn->mInitialSize = 75.0f;
        mWavelengthColumn->mFormat      = "{:.1f} nm";
        mWavelengthColumn->mNaNFormat   = "\xe2\x88\x9e nm";
        mEventTable.AddColumn( mWavelengthColumn );

        mSubCursorAColumn               = New<sFloat64Column>();
        mSubCursorAColumn->mHeader      = "a";
        mSubCursorAColumn->mInitialSize = 75.0f;
        mSubCursorAColumn->mFormat      = "{:.3f} km";
        mSubCursorAColumn->mNaNFormat   = "\xe2\x88\x9e km";
        mEventTable.AddColumn( mSubCursorAColumn );

        mCursorAColumn               = New<sFloat64Column>();
        mCursorAColumn->mHeader      = "A";
        mCursorAColumn->mInitialSize = 75.0f;
        mCursorAColumn->mFormat      = "{:.3f} km";
        mCursorAColumn->mNaNFormat   = "\xe2\x88\x9e km";
        mEventTable.AddColumn( mCursorAColumn );

        mCursorBColumn               = New<sFloat64Column>();
        mCursorBColumn->mHeader      = "B";
        mCursorBColumn->mInitialSize = 75.0f;
        mCursorBColumn->mFormat      = "{:.3f} km";
        mCursorBColumn->mNaNFormat   = "\xe2\x88\x9e km";
        mEventTable.AddColumn( mCursorBColumn );

        mSubCursorBColumn               = New<sFloat64Column>();
        mSubCursorBColumn->mHeader      = "b";
        mSubCursorBColumn->mInitialSize = 75.0f;
        mSubCursorBColumn->mFormat      = "{:.3f} km";
        mSubCursorBColumn->mNaNFormat   = "\xe2\x88\x9e km";
        mEventTable.AddColumn( mSubCursorBColumn );

        mCurveLevelColumn               = New<sFloat64Column>();
        mCurveLevelColumn->mHeader      = "Level";
        mCurveLevelColumn->mInitialSize = 75.0f;
        mCurveLevelColumn->mFormat      = "{:.1f} dB";
        mCurveLevelColumn->mNaNFormat   = "-\xe2\x88\x9e db";
        mEventTable.AddColumn( mCurveLevelColumn );

        mLossAtAColumn               = New<sFloat64Column>();
        mLossAtAColumn->mHeader      = "Loss@A";
        mLossAtAColumn->mInitialSize = 75.0f;
        mLossAtAColumn->mFormat      = "{:.1f} dB";
        mLossAtAColumn->mNaNFormat   = "-\xe2\x88\x9e db";
        mEventTable.AddColumn( mLossAtAColumn );

        mLossAtBColumn               = New<sFloat64Column>();
        mLossAtBColumn->mHeader      = "Loss@B";
        mLossAtBColumn->mInitialSize = 75.0f;
        mLossAtBColumn->mFormat      = "{:.1f} dB";
        mLossAtBColumn->mNaNFormat   = "-\xe2\x88\x9e db";
        mEventTable.AddColumn( mLossAtBColumn );

        mEstimatedCurveLevelColumn               = New<sFloat64Column>();
        mEstimatedCurveLevelColumn->mHeader      = "Est. Level";
        mEstimatedCurveLevelColumn->mInitialSize = 75.0f;
        mEstimatedCurveLevelColumn->mFormat      = "{:.1f} dB";
        mEstimatedCurveLevelColumn->mNaNFormat   = "-\xe2\x88\x9e db";
        mEventTable.AddColumn( mEstimatedCurveLevelColumn );

        mEstimatedEndLevelColumn               = New<sFloat64Column>();
        mEstimatedEndLevelColumn->mHeader      = "Est. End Level";
        mEstimatedEndLevelColumn->mInitialSize = 75.0f;
        mEstimatedEndLevelColumn->mFormat      = "{:.1f} dB";
        mEstimatedEndLevelColumn->mNaNFormat   = "-\xe2\x88\x9e db";
        mEventTable.AddColumn( mEstimatedEndLevelColumn );

        mEndNoiseLevelColumn               = New<sFloat64Column>();
        mEndNoiseLevelColumn->mHeader      = "End Noise Level";
        mEndNoiseLevelColumn->mInitialSize = 75.0f;
        mEndNoiseLevelColumn->mFormat      = "{:.1f} dB";
        mEndNoiseLevelColumn->mNaNFormat   = "-\xe2\x88\x9e db";
        mEventTable.AddColumn( mEndNoiseLevelColumn );

        mPeakPulseWidth               = New<sFloat64Column>();
        mPeakPulseWidth->mHeader      = "Pulse width";
        mPeakPulseWidth->mInitialSize = 75.0f;
        mPeakPulseWidth->mFormat      = "{:.1f} nm";
        mPeakPulseWidth->mNaNFormat   = "-\xe2\x88\x9e nm";
        mEventTable.AddColumn( mPeakPulseWidth );

        mPeakPower               = New<sFloat64Column>();
        mPeakPower->mHeader      = "Peak power";
        mPeakPower->mInitialSize = 75.0f;
        mPeakPower->mFormat      = "{:.1f} dB";
        mPeakPower->mNaNFormat   = "-\xe2\x88\x9e dB";
        mEventTable.AddColumn( mPeakPower );

        mPeakSNR               = New<sFloat64Column>();
        mPeakSNR->mHeader      = "Peak SNR";
        mPeakSNR->mInitialSize = 75.0f;
        mPeakSNR->mFormat      = "{:.1f} dB";
        mPeakSNR->mNaNFormat   = "-\xe2\x88\x9e dB";
        mEventTable.AddColumn( mPeakSNR );
    }

    OtdrWindow::OtdrWindow( Ref<VkGraphicContext> aGraphicContext, Ref<UIContext> aUIOverlay )
        : mGraphicContext{ aGraphicContext }
        , mUIOverlay{ aUIOverlay }
    {
    }

    bool OtdrWindow::Display()
    {
        ImGuizmo::SetOrthographic( false );
        static bool                  p_open          = true;
        constexpr ImGuiDockNodeFlags lDockSpaceFlags = ImGuiDockNodeFlags_PassthruCentralNode;
        constexpr ImGuiWindowFlags   lMainwindowFlags =
            ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos( viewport->WorkPos );
        ImGui::SetNextWindowSize( viewport->WorkSize - ImVec2( 0.0f, StatusBarHeight ) );
        ImGui::SetNextWindowViewport( viewport->ID );

        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0.0f, 0.0f ) );
        ImGui::Begin( "DockSpace Demo", &p_open, lMainwindowFlags );
        ImGui::PopStyleVar( 3 );

        ImGuiID dockspace_id = ImGui::GetID( "MyDockSpace" );
        ImGui::DockSpace( dockspace_id, ImVec2( 0.0f, 0.0f ), lDockSpaceFlags );

        bool lRequestQuit = false;
        ImGui::PushStyleVar( ImGuiStyleVar_FramePadding, ImVec2( 0, 8 ) );
        if( ImGui::BeginMainMenuBar() )
        {
            ImGui::PushStyleVar( ImGuiStyleVar_ItemSpacing, ImVec2( 15, 14 ) );
            lRequestQuit = RenderMainMenu();
            ImGui::PopStyleVar();
            ImGui::EndMainMenuBar();
        }
        ImGui::PopStyleVar();
        ImGui::End();

        mWorkspaceArea.Update();

        if( ImGui::Begin( "ASSEMBLIES", NULL, ImGuiWindowFlags_None ) )
        {
            MonoRuntime::DisplayAssemblies();
        }
        ImGui::End();

        if( mDataInstance )
        {
            static bool pOpen = true;
            if( ImGui::Begin( "iOlmData", &pOpen, ImGuiWindowFlags_None ) )
            {
                mTracePlot.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
            }
            ImGui::End();

            if( ImGui::Begin( "iOlmData_XXX", &pOpen, ImGuiWindowFlags_None ) )
            {
                mEventTable.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
            }
            ImGui::End();

            if( ImGui::Begin( "iOlmData_LinkElements", &pOpen, ImGuiWindowFlags_None ) )
            {
                mLinkElementTable.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
            }
            ImGui::End();

            // if( ImGui::Begin( "iOlmEvents", &pOpen, ImGuiWindowFlags_None ) )
            // {
            //     const ImGuiTableFlags flags = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
            //                                   ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
            //                                   ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;
            //     if( ImGui::BeginTable( "table_scrolly", 24, flags, ImGui::GetContentRegionAvail() ) )
            //     {
            //         ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, 15 );
            //         ImGui::TableSetupColumn( "mRowIndex", ImGuiTableColumnFlags_None, 75 );
            //         ImGui::TableSetupColumn( "mEventType", ImGuiTableColumnFlags_None, 75 );
            //         ImGui::TableSetupColumn( "mEventStatus", ImGuiTableColumnFlags_None, 75 );
            //         ImGui::TableSetupColumn( "mReflectanceType", ImGuiTableColumnFlags_None, 75 );
            //         ImGui::TableSetupColumn( "mWavelength", ImGuiTableColumnFlags_None, 75 );
            //         ImGui::TableSetupColumn( "mPosition", ImGuiTableColumnFlags_None, 75 );
            //         ImGui::TableSetupColumn( "mCursorA", ImGuiTableColumnFlags_None, 70 );
            //         ImGui::TableSetupColumn( "mCursorB", ImGuiTableColumnFlags_None, 70 );
            //         ImGui::TableSetupColumn( "mSubCursorA", ImGuiTableColumnFlags_None, 70 );
            //         ImGui::TableSetupColumn( "mSubCursorB", ImGuiTableColumnFlags_None, 70 );
            //         ImGui::TableSetupColumn( "mLoss", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mReflectance", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mCurveLevel", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mLossAtA", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mLossAtB", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mEstimatedLoss", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mCurveLevel", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mEstimatedEndLevel", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mEndNoiseLevel", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mPeakPulseWidth", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mPeakPower", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mPeakSNR", ImGuiTableColumnFlags_None, 55 );
            //         ImGui::TableSetupColumn( "mConsiderAsPossibleEcho", ImGuiTableColumnFlags_None, 55 );

            //         ImGui::TableHeadersRow();

            //         ImGuiListClipper clipper;
            //         clipper.Begin( mEventDataVector.size() );
            //         ImGui::TableNextRow();
            //         while( clipper.Step() )
            //         {
            //             for( int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++ )
            //             {
            //                 std::string lEventType;
            //                 std::string lEventStatus;
            //                 std::string lReflectanceType;
            //                 ImGui::TableNextRow();
            //                 ImGui::TableSetColumnIndex( 1 );
            //                 Text( "{}", mEventDataVector[row].mRowIndex );

            //                 ImGui::TableSetColumnIndex( 2 );
            //                 switch( mEventDataVector[row].mEventType )
            //                 {
            //                 case Unknown: lEventType = "Unknown"; break;
            //                 case PositiveSplice: lEventType = "Positive Splice"; break;
            //                 case NegativeSplice: lEventType = "Negative Splice"; break;
            //                 case Reflection: lEventType = "Reflection"; break;
            //                 case EndOfAnalysis: lEventType = "End Of Analysis"; break;
            //                 case ContinuousFiber: lEventType = "Continuous Fiber"; break;
            //                 default: lEventType = "N/A"; break;
            //                 }
            //                 Text( "{}", lEventType );
            //                 ImGui::TableSetColumnIndex( 3 );
            //                 switch( mEventDataVector[row].mEventStatus )
            //                 {
            //                 case None: lEventStatus = "None"; break;
            //                 case Echo: lEventStatus = "Echo"; break;
            //                 case PossibleEcho: lEventStatus = "Possible Echo"; break;
            //                 case EndOfFiber: lEventStatus = "End Of Fiber"; break;
            //                 case LaunchLevel: lEventStatus = "Launch Level"; break;
            //                 case Saturated: lEventStatus = "Saturated"; break;
            //                 case AddedByUser: lEventStatus = "Added B yUser"; break;
            //                 case SpanStart: lEventStatus = "Span Start"; break;
            //                 case SpanEnd: lEventStatus = "Span End"; break;
            //                 case NewWhileTemplating: lEventStatus = "New While Templating"; break;
            //                 case AddedForSpan: lEventStatus = "Added For Span"; break;
            //                 case AddedFromReference: lEventStatus = "Added From Reference"; break;
            //                 case Bidir: lEventStatus = "Bidir"; break;
            //                 case Splitter: lEventStatus = "Splitter"; break;
            //                 case PreviousSectionEcho: lEventStatus = "Previous Section Echo"; break;
            //                 case UnderEstimatedLoss: lEventStatus = "Under Estimated Loss"; break;
            //                 case UnderEstimatedReflectance: lEventStatus = "Under Estimated Reflectance"; break;
            //                 case LoopStart: lEventStatus = "Loop Start"; break;
            //                 case LoopEnd: lEventStatus = "Loop End"; break;
            //                 case CouplerPort: lEventStatus = "Coupler Port"; break;
            //                 case Reference: lEventStatus = "Reference"; break;
            //                 case OverEstimatedReflectance: lEventStatus = "Over Estimated Reflectance"; break;
            //                 case InjectionReference: lEventStatus = "Injection Reference"; break;
            //                 case OverEstimatedLoss: lEventStatus = "Over Estimated Loss"; break;
            //                 default: lEventStatus = "N/A"; break;
            //                 }
            //                 Text( "{}", lEventStatus );

            //                 ImGui::TableSetColumnIndex( 4 );

            //                 switch( mEventDataVector[row].mReflectanceType )
            //                 {
            //                 case Bidirectional: lReflectanceType = "Bidirectional"; break;
            //                 case UnidirectionalForward: lReflectanceType = "Unidirectional Forward"; break;
            //                 case UnidirectionalBackward: lReflectanceType = "Unidirectional Backward"; break;
            //                 default: lReflectanceType = "N/A"; break;
            //                 }
            //                 Text( "{}", lReflectanceType );

            //                 ImGui::TableSetColumnIndex( 5 );
            //                 Text( "{:.1f} nm", mEventDataVector[row].mWavelength );

            //                 ImGui::TableSetColumnIndex( 6 );
            //                 Text( "{:.3f} km", mEventDataVector[row].mPosition / 1000.0f );

            //                 ImGui::TableSetColumnIndex( 7 );
            //                 Text( "{:.3f} km", mEventDataVector[row].mCursorA / 1000.0f );

            //                 ImGui::TableSetColumnIndex( 8 );
            //                 Text( "{:.3f} km", mEventDataVector[row].mCursorB / 1000.0f );

            //                 ImGui::TableSetColumnIndex( 9 );
            //                 Text( "{:.3f} km", mEventDataVector[row].mSubCursorA / 1000.0f );

            //                 ImGui::TableSetColumnIndex( 10 );
            //                 Text( "{:.3f} km", mEventDataVector[row].mSubCursorB / 1000.0f );

            //                 ImGui::TableSetColumnIndex( 11 );
            //                 if( std::isnan( mEventDataVector[row].mLoss ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mLoss );

            //                 ImGui::TableSetColumnIndex( 12 );
            //                 if( std::isnan( mEventDataVector[row].mReflectance ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mReflectance );

            //                 ImGui::TableSetColumnIndex( 13 );
            //                 if( std::isnan( mEventDataVector[row].mCurveLevel ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mCurveLevel );

            //                 ImGui::TableSetColumnIndex( 14 );
            //                 if( std::isnan( mEventDataVector[row].mLossAtA ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mLossAtA );

            //                 ImGui::TableSetColumnIndex( 15 );
            //                 if( std::isnan( mEventDataVector[row].mLossAtB ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mLossAtB );

            //                 ImGui::TableSetColumnIndex( 16 );
            //                 if( std::isnan( mEventDataVector[row].mEstimatedCurveLevel ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mEstimatedCurveLevel );

            //                 ImGui::TableSetColumnIndex( 17 );
            //                 if( std::isnan( mEventDataVector[row].mEstimatedLoss ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mEstimatedLoss );

            //                 ImGui::TableSetColumnIndex( 18 );
            //                 if( std::isnan( mEventDataVector[row].mEstimatedEndLevel ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mEstimatedEndLevel );

            //                 ImGui::TableSetColumnIndex( 19 );
            //                 if( std::isnan( mEventDataVector[row].mEndNoiseLevel ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mEndNoiseLevel );

            //                 ImGui::TableSetColumnIndex( 20 );
            //                 if( std::isnan( mEventDataVector[row].mPeakPulseWidth ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mPeakPulseWidth );

            //                 ImGui::TableSetColumnIndex( 21 );
            //                 if( std::isnan( mEventDataVector[row].mPeakPower ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mPeakPower );

            //                 ImGui::TableSetColumnIndex( 22 );
            //                 if( std::isnan( mEventDataVector[row].mPeakSNR ) )
            //                     Text( "-\xe2\x88\x9e dB" );
            //                 else
            //                     Text( "{:.1f} dB", mEventDataVector[row].mPeakSNR );

            //                 ImGui::TableSetColumnIndex( 23 );
            //                 Text( "{}", mEventDataVector[row].mConsiderAsPossibleEcho );
            //             }
            //         }
            //         ImGui::EndTable();
            //     }
            // }
            // ImGui::End();

            if( !pOpen )
            {
                mDataInstance = nullptr;
                pOpen         = true;
            }
        }

        if( ImGui::Begin( "CONNECTED MODULES", NULL, ImGuiWindowFlags_None ) )
        {
            static auto lFirstRun = true;
            static auto lLastTime =
                std::chrono::time_point_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() )
                    .time_since_epoch()
                    .count();
            auto lCurrentTime = std::chrono::time_point_cast<std::chrono::microseconds>( std::chrono::high_resolution_clock::now() )
                                    .time_since_epoch()
                                    .count();

            static auto &lMonoGlue          = MonoRuntime::GetClassType( "Metrino.Mono.Instruments" );
            static auto &lModuleDescription = MonoRuntime::GetClassType( "Metrino.Mono.ModuleDescription" );

            static uint32_t                 lNumConnectedModules  = 0;
            static std::vector<std::string> lConnectedModuleNames = {};

            if( lFirstRun || ( ( lCurrentTime - lLastTime ) > 1500000 ) )
            {
                lConnectedModuleNames.clear();
                lLastTime                  = lCurrentTime;
                auto lConnectedModulesList = lMonoGlue.CallMethod( "GetConnectedModules" );

                lNumConnectedModules = static_cast<uint32_t>( mono_array_length( (MonoArray *)lConnectedModulesList ) );

                std::vector<MonoObject *> lConnectedModules( lNumConnectedModules );
                for( uint32_t i = 0; i < lNumConnectedModules; i++ )
                {
                    auto lConnectedModule = *( mono_array_addr( (MonoArray *)lConnectedModulesList, MonoObject *, i ) );
                    auto lInstance        = MonoScriptInstance( lModuleDescription.Class(), lConnectedModule );
                    lConnectedModuleNames.push_back( MonoRuntime::NewString( lInstance.GetFieldValue<MonoString *>( "mName" ) ) );
                }

                lFirstRun = false;
            }

            if( lNumConnectedModules == 0 )
            {
                UI::Text( "No connected modules", lNumConnectedModules );
            }
            else
            {
                for( uint32_t i = 0; i < lNumConnectedModules; i++ )
                {
                    Text( "{} {}", ICON_FA_USB, lConnectedModuleNames[i] );
                }
            }
        }
        ImGui::End();

        // if( ImGui::Begin( "LOGS", &p_open, ImGuiWindowFlags_None ) )
        // {
        //     math::vec2 l_WindowConsoleSize = UI::GetAvailableContentSpace();
        //     Console( l_WindowConsoleSize.x, l_WindowConsoleSize.y );
        // }
        // ImGui::End();

        if( ImGui::Begin( "PROFILING", &p_open, ImGuiWindowFlags_None ) )
        {
            math::vec2  l_WindowConsoleSize     = UI::GetAvailableContentSpace();
            static bool lIsProfiling            = false;
            static bool lProfilingDataAvailable = false;

            static std::unordered_map<std::string, float>    lProfilingResults = {};
            static std::unordered_map<std::string, uint32_t> lProfilingCount   = {};

            std::string lButtonText = lIsProfiling ? "Stop capture" : "Start capture";
            if( UI::Button( lButtonText.c_str(), { 150.0f, 50.0f } ) )
            {
                if( lIsProfiling )
                {
                    auto lResults = SE::Core::Instrumentor::Get().EndSession();
                    SE::Logging::Info( "{}", lResults->mEvents.size() );
                    lIsProfiling            = false;
                    lProfilingDataAvailable = true;
                    lProfilingResults       = {};
                    for( auto &lEvent : lResults->mEvents )
                    {
                        if( lProfilingResults.find( lEvent.mName ) == lProfilingResults.end() )
                        {
                            lProfilingResults[lEvent.mName] = lEvent.mElapsedTime;
                            lProfilingCount[lEvent.mName]   = 1;
                        }
                        else
                        {
                            lProfilingResults[lEvent.mName] += lEvent.mElapsedTime;
                            lProfilingCount[lEvent.mName] += 1;
                        }
                    }

                    for( auto &lEntry : lProfilingResults )
                    {
                        lProfilingResults[lEntry.first] /= lProfilingCount[lEntry.first];
                    }
                }
                else
                {
                    SE::Core::Instrumentor::Get().BeginSession( "Session" );
                    lIsProfiling            = true;
                    lProfilingDataAvailable = false;
                    lProfilingResults       = {};
                }
            }

            if( lProfilingDataAvailable )
            {
                for( auto &lEntry : lProfilingResults )
                {
                    UI::Text( "{} ----> {} us", lEntry.first, lEntry.second );
                }
            }
            // Console( l_WindowConsoleSize.x, l_WindowConsoleSize.y );
        }
        ImGui::End();

        if( ( ImGui::Begin( "SCENE HIERARCHY", &p_open, ImGuiWindowFlags_None ) ) )
        {
            auto lWindowPropertiesSize = UI::GetAvailableContentSpace();

            mSceneHierarchyPanel.World = mActiveWorld;
            mSceneHierarchyPanel.Display( lWindowPropertiesSize.x, lWindowPropertiesSize.y );
        }
        ImGui::End();

        ImPlot::ShowDemoWindow();
        ImGui::ShowDemoWindow();

        ImGui::PushStyleColor( ImGuiCol_WindowBg, ImVec4{ 102.0f / 255.0f, 0.0f, 204.0f / 255.0f, 1.0f } );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 5.0f, 7.0f ) );
        ImGui::SetNextWindowSize( ImVec2{ viewport->WorkSize.x, StatusBarHeight } );
        ImGui::SetNextWindowPos( ImVec2{ 0.0f, viewport->WorkSize.y } );
        constexpr ImGuiWindowFlags lStatusBarFlags =
            ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

        ImGui::Begin( "##STATUS_BAR", &p_open, lStatusBarFlags );
        ImGui::PopStyleVar( 3 );
        ImGui::PopStyleColor();
        auto l_WindowSize = UI::GetAvailableContentSpace();
        {
            if( mLastFPS > 0 )
                Text( "Render: {} fps ({:.2f} ms/frame)", mLastFPS, ( 1000.0f / mLastFPS ) );
            else
                Text( "Render: 0 fps (0 ms)" );
        }
        ImGui::End();

        return lRequestQuit;
    }

    template <typename _Ty>
    std::vector<_Ty> AsVector( MonoObject *aObject )
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

    void OtdrWindow::LoadIOlmData( fs::path aPath )
    {
        static auto &lFileLoader = MonoRuntime::GetClassType( "Metrino.Mono.FileLoader" );
        static auto &lFileClass  = MonoRuntime::GetClassType( "Metrino.Mono.OlmFile" );

        MonoString *lFilePath   = MonoRuntime::NewString( aPath.string() );
        MonoObject *lDataObject = lFileLoader.CallMethod( "LoadOlmData", lFilePath );

        mDataInstance = New<MonoScriptInstance>( lFileClass.Class(), lDataObject );

        MonoObject               *lTraceData       = mDataInstance->CallMethod( "GetAllTraces" );
        std::vector<MonoObject *> lTraceDataVector = AsVector<MonoObject *>( lTraceData );

        static auto &lTraceDataStructure = MonoRuntime::GetClassType( "Metrino.Mono.TracePlotData" );

        mTracePlot.Clear();
        for( int i = 0; i < lTraceDataVector.size(); i++ )
        {
            auto lInstance = MonoScriptInstance( lTraceDataStructure.Class(), lTraceDataVector[i] );
            auto lPlot     = New<sFloat64LinePlot>();
            lPlot->mX      = AsVector<double>( lInstance.GetFieldValue<MonoObject *>( "mX" ) );
            lPlot->mY      = AsVector<double>( lInstance.GetFieldValue<MonoObject *>( "mY" ) );
            lPlot->mLegend = fmt::format( "{:.0f} nm - {} ({} samples)", lInstance.GetFieldValue<double>( "mWavelength" ) * 1e9, i,
                                          lPlot->mX.size() );

            mTracePlot.Add( lPlot );
        }

        MonoObject *lEventData = mDataInstance->CallMethod( "GetEvents" );
        mEventDataVector       = AsVector<sEvent>( lEventData );

        mPositionColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mPositionColumn->mData.push_back( lE.mPosition * 0.001f );

        mLossColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mLossColumn->mData.push_back( lE.mLoss );

        mReflectanceColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mReflectanceColumn->mData.push_back( lE.mReflectance );

        mEstimatedLossColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mEstimatedLossColumn->mData.push_back( lE.mEstimatedLoss );

        mReflectanceColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mReflectanceColumn->mData.push_back( lE.mReflectance );

        mWavelengthColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mWavelengthColumn->mData.push_back( lE.mWavelength );

        mCursorAColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mCursorAColumn->mData.push_back( lE.mCursorA * 0.001f );

        mCursorBColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mCursorBColumn->mData.push_back( lE.mCursorB * 0.001f );

        mSubCursorAColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mSubCursorAColumn->mData.push_back( lE.mSubCursorA * 0.001f );

        mSubCursorBColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mSubCursorBColumn->mData.push_back( lE.mSubCursorB * 0.001f );

        mCurveLevelColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mCurveLevelColumn->mData.push_back( lE.mCurveLevel );

        mLossAtAColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mLossAtAColumn->mData.push_back( lE.mLossAtA );

        mLossAtBColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mLossAtBColumn->mData.push_back( lE.mLossAtB );

        mEstimatedCurveLevelColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mEstimatedCurveLevelColumn->mData.push_back( lE.mEstimatedCurveLevel );

        mEstimatedEndLevelColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mEstimatedEndLevelColumn->mData.push_back( lE.mEstimatedEndLevel );

        mEndNoiseLevelColumn->mData.clear();
        for( auto const &lE : mEventDataVector ) mEndNoiseLevelColumn->mData.push_back( lE.mEndNoiseLevel );

        mPeakPulseWidth->mData.clear();
        for( auto const &lE : mEventDataVector ) mPeakPulseWidth->mData.push_back( lE.mPeakPulseWidth );

        mPeakPower->mData.clear();
        for( auto const &lE : mEventDataVector ) mPeakPower->mData.push_back( lE.mPeakPower );

        mPeakSNR->mData.clear();
        for( auto const &lE : mEventDataVector ) mPeakSNR->mData.push_back( lE.mPeakSNR );

        MonoObject *lLinkElementData   = mDataInstance->CallMethod( "GetLinkElements" );
        auto        lLinkElementVector = AsVector<sLinkElement>( lLinkElementData );
        mLinkElementTable.SetData(lLinkElementVector);
    }

    bool OtdrWindow::RenderMainMenu()
    {
        UI::Text( ApplicationIcon.c_str() );

        bool lRequestQuit = false;

        if( ImGui::BeginMenu( "File" ) )
        {
            if( UI::MenuItem( fmt::format( "{} Load", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "YAML Files (*.yaml)\0*.yaml\0All Files (*.*)\0*.*\0" );
                mWorld->Load( fs::path( lFilePath.value() ) );
            }

            if( UI::MenuItem( fmt::format( "{} Load iOlm", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "OLM Files (*.iolm)\0*.iolm\0All Files (*.*)\0*.*\0" );

                LoadIOlmData( fs::path( lFilePath.value() ) );
                // mWorld->Load( fs::path( lFilePath.value() ) );
            }

            if( UI::MenuItem( fmt::format( "{} Save", ICON_FA_ARCHIVE ).c_str(), NULL ) )
            {
                if( mCurrentPath.empty() )
                {
                    auto lFilePath = FileDialogs::SaveFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                            "YAML Files (*.yaml)\0*.yaml\0All Files (*.*)\0*.*\0" );

                    if( lFilePath.has_value() )
                    {
                        mCurrentPath = fs::path( lFilePath.value() );
                        mWorld->SaveAs( mCurrentPath );
                    }
                }
                else
                {
                    mWorld->SaveAs( mCurrentPath );
                }
            }

            if( UI::MenuItem( fmt::format( "{} Save as...", ICON_FA_ARCHIVE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::SaveFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "YAML Files (*.yaml)\0*.yaml\0All Files (*.*)\0*.*\0" );
                mCurrentPath   = fs::path( lFilePath.value() );
                mWorld->SaveAs( mCurrentPath );
            }

            lRequestQuit = UI::MenuItem( fmt::format( "{} Exit", ICON_FA_WINDOW_CLOSE_O ).c_str(), NULL );
            ImGui::EndMenu();
        }

        return lRequestQuit;
    }

    math::ivec2 OtdrWindow::GetWorkspaceAreaSize() { return mWorkspaceAreaSize; }

    void OtdrWindow::Update( Timestep aTs )
    {
        mActiveWorld->Update( aTs );

        mWorkspaceArea.Tick();

        UpdateFramerate( aTs );
    }

    void OtdrWindow::UpdateFramerate( Timestep ts )
    {
        mFrameCounter++;
        mFpsTimer += (float)ts;
        if( mFpsTimer > 1000.0f )
        {
            mLastFPS      = static_cast<uint32_t>( (float)mFrameCounter * ( 1000.0f / mFpsTimer ) );
            mFpsTimer     = 0.0f;
            mFrameCounter = 0;
        }
    }

} // namespace SE::OtdrEditor