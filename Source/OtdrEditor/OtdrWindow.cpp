#include "OtdrWindow.h"

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

    void OtdrWindow::ConfigureUI() { mWorkspaceArea.ConfigureUI(); }

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

            if( ImGui::Begin( "iOlmEvents", &pOpen, ImGuiWindowFlags_None ) )
            {
                const ImGuiTableFlags flags = ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                                              ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_Resizable |
                                              ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable;
                if( ImGui::BeginTable( "table_scrolly", 14, flags, ImGui::GetContentRegionAvail() ) )
                {
                    ImGui::TableSetupColumn( "", ImGuiTableColumnFlags_WidthFixed, 15 );
                    ImGui::TableSetupColumn( "mRowIndex", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mEventType", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mEventStatus", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mReflectanceType", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mWavelength", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mPosition", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mCursorA", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mCursorB", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mSubCursorA", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mSubCursorB", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mLoss", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mReflectance", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableSetupColumn( "mCurveLevel", ImGuiTableColumnFlags_None, 75 );
                    ImGui::TableHeadersRow();

                    ImGuiListClipper clipper;
                    clipper.Begin( mEventDataVector.size() );
                    ImGui::TableNextRow();
                    while( clipper.Step() )
                    {
                        for( int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++ )
                        {
                            std::string lEventType;
                            std::string lEventStatus;
                            std::string lReflectanceType;
                            ImGui::TableNextRow();
                            ImGui::TableSetColumnIndex( 1 );
                            Text( "{}", mEventDataVector[row].mRowIndex );
                            ImGui::TableSetColumnIndex( 2 );
                            switch( mEventDataVector[row].mEventType )
                            {
                            case Unknown: lEventType = "Unknown"; break;
                            case PositiveSplice: lEventType = "PositiveSplice"; break;
                            case NegativeSplice: lEventType = "NegativeSplice"; break;
                            case Reflection: lEventType = "Reflection"; break;
                            case EndOfAnalysis: lEventType = "EndOfAnalysis"; break;
                            case ContinuousFiber: lEventType = "ContinuousFiber"; break;
                            default: lEventType = "N/A"; break;
                            }
                            Text( "{}", lEventType );
                            ImGui::TableSetColumnIndex( 3 );
                            switch( mEventDataVector[row].mEventStatus )
                            {
                            case None: lEventStatus = "None"; break;
                            case Echo: lEventStatus = "Echo"; break;
                            case PossibleEcho: lEventStatus = "PossibleEcho"; break;
                            case EndOfFiber: lEventStatus = "EndOfFiber"; break;
                            case LaunchLevel: lEventStatus = "LaunchLevel"; break;
                            case Saturated: lEventStatus = "Saturated"; break;
                            case AddedByUser: lEventStatus = "AddedByUser"; break;
                            case SpanStart: lEventStatus = "SpanStart"; break;
                            case SpanEnd: lEventStatus = "SpanEnd"; break;
                            case NewWhileTemplating: lEventStatus = "NewWhileTemplating"; break;
                            case AddedForSpan: lEventStatus = "AddedForSpan"; break;
                            case AddedFromReference: lEventStatus = "AddedFromReference"; break;
                            case Bidir: lEventStatus = "Bidir"; break;
                            case Splitter: lEventStatus = "Splitter"; break;
                            case PreviousSectionEcho: lEventStatus = "PreviousSectionEcho"; break;
                            case UnderEstimatedLoss: lEventStatus = "UnderEstimatedLoss"; break;
                            case UnderEstimatedReflectance: lEventStatus = "UnderEstimatedReflectance"; break;
                            case LoopStart: lEventStatus = "LoopStart"; break;
                            case LoopEnd: lEventStatus = "LoopEnd"; break;
                            case CouplerPort: lEventStatus = "CouplerPort"; break;
                            case Reference: lEventStatus = "Reference"; break;
                            case OverEstimatedReflectance: lEventStatus = "OverEstimatedReflectance"; break;
                            case InjectionReference: lEventStatus = "InjectionReference"; break;
                            case OverEstimatedLoss: lEventStatus = "OverEstimatedLoss"; break;
                            default: lEventStatus = "N/A"; break;
                            }
                            Text( "{}", lEventStatus );
                            ImGui::TableSetColumnIndex( 4 );

                            switch( mEventDataVector[row].mReflectanceType )
                            {
                            case Bidirectional: lReflectanceType = "Bidirectional"; break;
                            case UnidirectionalForward: lReflectanceType = "UnidirectionalForward"; break;
                            case UnidirectionalBackward: lReflectanceType = "UnidirectionalBackward"; break;
                            default: lReflectanceType = "N/A"; break;
                            }
                            Text( "{}", lReflectanceType );
                            ImGui::TableSetColumnIndex( 5 );
                            Text( "{:.1f} nm", mEventDataVector[row].mWavelength );
                            ImGui::TableSetColumnIndex( 6 );
                            Text( "{:.3f} km", mEventDataVector[row].mPosition / 1000.0f );
                            ImGui::TableSetColumnIndex( 7 );
                            Text( "{:.3f} km", mEventDataVector[row].mCursorA / 1000.0f );
                            ImGui::TableSetColumnIndex( 8 );
                            Text( "{:.3f} km", mEventDataVector[row].mCursorB / 1000.0f );
                            ImGui::TableSetColumnIndex( 9 );
                            Text( "{:.3f} km", mEventDataVector[row].mSubCursorA / 1000.0f );
                            ImGui::TableSetColumnIndex( 10 );
                            Text( "{:.3f} km", mEventDataVector[row].mSubCursorB / 1000.0f );
                            ImGui::TableSetColumnIndex( 11 );
                            Text( "{:.1f} dB", mEventDataVector[row].mLoss );
                            ImGui::TableSetColumnIndex( 12 );
                            Text( "{:.1f} dB", mEventDataVector[row].mReflectance );
                            ImGui::TableSetColumnIndex( 13 );
                            Text( "{:.1f} dB", mEventDataVector[row].mCurveLevel );
                        }
                    }
                    ImGui::EndTable();
                }
            }
            ImGui::End();

            if( !pOpen ) mDataInstance = nullptr;
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