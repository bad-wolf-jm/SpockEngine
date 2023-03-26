#include "OtdrWindow.h"

#include <cmath>
#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <locale>

#include "pugixml.hpp"

#include "Core/Profiling/BlockTimer.h"

#include "UI/Widgets.h"

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

    using namespace SE::Core;
    // using namespace SE::Core::EntityComponentSystem::Components;

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
        mLinkElementTable->OnElementClicked(
            [&]( sLinkElement const &aElement )
            {
                static auto &lOlmMeasurementClass = MonoRuntime::GetClassType( "Metrino.Olm.OlmPhysicalEvent" );
                static auto &lOlmAttributeClass = MonoRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );

                auto lPhysicalEvent =
                    New<MonoScriptInstance>( &lOlmMeasurementClass, lOlmMeasurementClass.Class(), aElement.mPhysicalEvent );
                auto lAttributes = New<MonoScriptInstance>( &lOlmAttributeClass, lOlmAttributeClass.Class(), aElement.mAttributes );

                // if( *lPhysicalEvent && *lAttributes ) mEventOverview.SetData( lPhysicalEvent, lAttributes );

                static auto &lAcquisitionDataClassType = MonoRuntime::GetClassType( "Metrino.Otdr.AcquisitionData" );
                auto lAcquisitionDataInstance = New<MonoScriptInstance>( &lAcquisitionDataClassType, lAcquisitionDataClassType.Class(),
                                                                         aElement.mAcquisitionData );
                if( *lAcquisitionDataInstance )
                {
                    static auto &lSinglePulseTraceClass = MonoRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );
                    static auto &lFiberInfoClassType    = MonoRuntime::GetClassType( "Metrino.Otdr.PhysicalFiberCharacteristics" );

                    auto lSinglePulseTraceInstance =
                        New<MonoScriptInstance>( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), aElement.mPeakTrace );
                    auto lFiberInfo =
                        New<MonoScriptInstance>( &lFiberInfoClassType, lFiberInfoClassType.Class(), aElement.mFiberInfo );
                    mAcquisitionDataOverview.SetData( lSinglePulseTraceInstance, lAcquisitionDataInstance, lFiberInfo );
                }
                // mTracePlot.SetEventData( mLinkElementTable->GetElementsByIndex( aElement.mLinkIndex ) );
                mTracePlot.Clear();
                mTracePlot.SetEventData( aElement, true, true, true );
            } );

        // mLinkElementTable1->OnElementClicked(
        //     [&]( sLinkElement const &aElement )
        //     {
        //         static auto &lOlmMeasurementClass = MonoRuntime::GetClassType( "Metrino.Olm.OlmPhysicalEvent" );
        //         static auto &lOlmAttributeClass = MonoRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute"
        //         );

        //         auto lPhysicalEvent =
        //             New<MonoScriptInstance>( &lOlmMeasurementClass, lOlmMeasurementClass.Class(), aElement.mPhysicalEvent );
        //         auto lAttributes = New<MonoScriptInstance>( &lOlmAttributeClass, lOlmAttributeClass.Class(), aElement.mAttributes );

        //         // mEventOverview.SetData( lPhysicalEvent, lAttributes );

        //         // mTracePlot.SetEventData( mLinkElementTable->GetElementsByIndex( aElement.mLinkIndex ) );
        //         mTracePlot.Clear();
        //         mTracePlot.SetEventData( aElement, true, true );
        //     } );

        mTestFailResultTable->OnElementClicked( [&]( sTestFailElement const &aElement ) { LoadIOlmData( aElement.mFilename ); } );
    }

    OtdrWindow::OtdrWindow( Ref<VkGraphicContext> aGraphicContext, Ref<UIContext> aUIOverlay )
        : mGraphicContext{ aGraphicContext }
        , mUIOverlay{ aUIOverlay }
    {
        mLinkElementTable    = New<UILinkElementTable>();
        mLinkElementTable1   = New<UILinkElementTable>();
        mEventTable          = New<UIMultiPulseEventTable>();
        mEventTable1         = New<UIMultiPulseEventTable>();
        mTestFailResultTable = New<UITestFailResultTable>();
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

        if( ImGui::Begin( "CLASSES", NULL, ImGuiWindowFlags_None ) )
        {
            math::vec2 l_WindowConsoleSize = UI::GetAvailableContentSpace();
            mMonoClasses.Display( l_WindowConsoleSize.x, l_WindowConsoleSize.y );
        }
        ImGui::End();

        if( mDataInstance )
        {
            static bool pOpen = true;
            if( ImGui::Begin( "iOlmData", &pOpen, ImGuiWindowFlags_None ) )
            {
                if( ImGui::Button( "All Traces" ) )
                {
                    mTracePlot.Clear();
                    mTracePlot.SetEventData( mLinkElementVector );
                }
                mTracePlot.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
            }
            ImGui::End();

            if( ImGui::Begin( "iOlmData_Overview", &pOpen, ImGuiWindowFlags_None ) )
            {
                mMeasurementOverview.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
            }
            ImGui::End();

            if( ImGui::Begin( "iOlmData_EventOverview", &pOpen, ImGuiWindowFlags_None ) )
            {
                mEventOverview.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
            }
            ImGui::End();

            if( ImGui::Begin( "iOlmData_AcquisitionDataOverview", &pOpen, ImGuiWindowFlags_None ) )
            {
                mAcquisitionDataOverview.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
            }
            ImGui::End();

            if( ImGui::Begin( "iOlmData_LinkElements", &pOpen, ImGuiWindowFlags_None ) )
            {
                mLinkElementTable->Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
            }
            ImGui::End();

            if( ImGui::Begin( "iOlmData_Events", &pOpen, ImGuiWindowFlags_None ) )
            {
                mEventTable->Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
            }
            ImGui::End();

            if( !pOpen )
            {
                mDataInstance = nullptr;
                pOpen         = true;
            }
        }

        if( ImGui::Begin( "Test results", NULL, ImGuiWindowFlags_None ) )
        {
            mTestFailResultTable->Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
        }
        ImGui::End();

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

            static auto &lMonoGlue          = MonoRuntime::GetClassType( "Metrino.Interop.Instruments" );
            static auto &lModuleDescription = MonoRuntime::GetClassType( "Metrino.Interop.ModuleDescription" );

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

    void OtdrWindow::LoadIOlmData( fs::path aPath, bool aReanalyse )
    {
        static auto &lFileLoader = MonoRuntime::GetClassType( "Metrino.Interop.FileLoader" );
        static auto &lFileClass  = MonoRuntime::GetClassType( "Metrino.Interop.OlmFile" );

        MonoString *lFilePath   = MonoRuntime::NewString( aPath.string() );
        MonoObject *lDataObject = lFileLoader.CallMethod( "LoadOlmData", lFilePath );

        mDataInstance = New<MonoScriptInstance>( &lFileClass, lFileClass.Class(), lDataObject );

        MonoObject               *lTraceData       = mDataInstance->CallMethod( "GetAllTraces" );
        std::vector<MonoObject *> lTraceDataVector = MonoRuntime::AsVector<MonoObject *>( lTraceData );
        if( lTraceDataVector.size() != 0 )
        {
            static auto &lTraceDataStructure        = MonoRuntime::GetClassType( "Metrino.Interop.TracePlotData" );
            static auto &lSinglePulseTraceClassType = MonoRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );
            static auto &lAcquisitionDataClassType  = MonoRuntime::GetClassType( "Metrino.Otdr.AcquisitionData" );
            static auto &lFiberInfoClassType        = MonoRuntime::GetClassType( "Metrino.Otdr.PhysicalFiberCharacteristics" );

            auto &lTraceDataInstance = MonoScriptInstance( &lTraceDataStructure, lTraceDataStructure.Class(), lTraceDataVector[0] );

            auto lSinglePulseTrace = lTraceDataInstance.GetFieldValue<MonoObject *>( "mTrace" );
            auto lSinglePulseTraceInstance =
                New<MonoScriptInstance>( &lSinglePulseTraceClassType, lSinglePulseTraceClassType.Class(), lSinglePulseTrace );

            auto lAcquisitionData = lTraceDataInstance.GetFieldValue<MonoObject *>( "mAcquisitionData" );
            auto lAcquisitionDataInstance =
                New<MonoScriptInstance>( &lAcquisitionDataClassType, lAcquisitionDataClassType.Class(), lAcquisitionData );

            auto lFiberInfo = lTraceDataInstance.GetPropertyValue( "FiberInfo", "Metrino.Otdr.PhysicalFiberCharacteristics" );
            mAcquisitionDataOverview.SetData( lSinglePulseTraceInstance, lAcquisitionDataInstance, lFiberInfo );
        }

        {
            bool        lX         = false;
            MonoObject *lEventData = mDataInstance->CallMethod( "GetEvents", &lX );
            mEventVector           = MonoRuntime::AsVector<sMultiPulseEvent>( lEventData );
            mEventTable->SetData( mEventVector );
        }

        {
            bool        lX               = false;
            MonoObject *lLinkElementData = mDataInstance->CallMethod( "GetLinkElements", &lX );
            mLinkElementVector           = MonoRuntime::AsVector<sLinkElement>( lLinkElementData );

            mLinkElementTable->SetData( mLinkElementVector );
            mTracePlot.Clear();
            mTracePlot.SetEventData( mLinkElementVector );

            mMeasurementOverview.SetData( mDataInstance );
        }
    }

    void OtdrWindow::LoadTestReport( fs::path aPath )
    {
        mTestFailResultTable->Clear();

        std::vector<sTestFailElement> lTableRows;

        for( auto const &lFile : std::filesystem::directory_iterator( aPath ) )
        {
            pugi::xml_document     doc;
            pugi::xml_parse_result result = doc.load_file( lFile.path().c_str() );

            if( !result ) continue;

            auto lRoot           = doc.child( "TestDataFailInfo" );
            auto lTestName       = lRoot.child( "TestName" ).child_value();
            auto lTestDate       = lRoot.child( "DateString" ).child_value();
            auto lFailesFileList = lRoot.child( "FailedFiles" );

            for( pugi::xml_node lInfo : lFailesFileList.children( "FailedFileInfo" ) )
            {
                auto *lFileName = lInfo.child( "Filename" ).child_value();

                for( pugi::xml_node lFailInfo : lInfo.children( "FailInfos" ) )
                {
                    for( pugi::xml_node lFail : lFailInfo.children( "FailInfo" ) )
                    {
                        auto &lNewData = lTableRows.emplace_back();

                        lNewData.mTestName              = std::string( lTestName );
                        lNewData.mTestDate              = std::string( lTestDate );
                        lNewData.mFilename              = std::string( lFileName );
                        lNewData.mLinkElementIndex      = std::string( lFail.child( "LinkElementIndex" ).child_value() );
                        lNewData.mSubLinkElementIndex   = std::string( lFail.child( "SubLinkElementIndex" ).child_value() );
                        lNewData.mPhysicalEventIndex    = std::string( lFail.child( "PhysicalEventIndex" ).child_value() );
                        lNewData.mLinkElementPosition   = std::stod( lFail.child( "LinkElementPosition" ).child_value() );
                        lNewData.mIsSubElement          = std::string( lFail.child( "IsSubElement" ).child_value() );
                        lNewData.mWavelength            = std::stod( lFail.child( "Wavelength" ).child_value() );
                        lNewData.mPhysicalEventPosition = std::stod( lFail.child( "PhysicalEventPosition" ).child_value() );
                        lNewData.mSinglePulseTraceIndex = std::string( lFail.child( "SinglePulseTraceIndex" ).child_value() );
                        lNewData.mMessage               = std::string( lFail.child( "Message" ).child_value() );
                    }
                }
            }
        }

        mTestFailResultTable->SetData( lTableRows );
    }

    bool OtdrWindow::RenderMainMenu()
    {
        UI::Text( ApplicationIcon.c_str() );

        bool lRequestQuit = false;

        if( ImGui::BeginMenu( "File" ) )
        {
            if( UI::MenuItem( fmt::format( "{} Load iOlm", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "OLM Files (*.iolm)\0*.iolm\0All Files (*.*)\0*.*\0" );

                if( lFilePath.has_value() ) LoadIOlmData( fs::path( lFilePath.value() ) );
            }

            if( UI::MenuItem( fmt::format( "{} Load test report", ICON_FA_PLUS_CIRCLE ).c_str(), NULL ) )
            {
                auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                        "XML Files (*.xml)\0*.xml\0All Files (*.*)\0*.*\0" );

                if( lFilePath.has_value() ) LoadTestReport( fs::path( lFilePath.value() ).parent_path() );
            }
            lRequestQuit = UI::MenuItem( fmt::format( "{} Exit", ICON_FA_WINDOW_CLOSE_O ).c_str(), NULL );
            ImGui::EndMenu();
        }

        return lRequestQuit;
    }

    math::ivec2 OtdrWindow::GetWorkspaceAreaSize() { return mWorkspaceAreaSize; }

    void OtdrWindow::Update( Timestep aTs )
    {
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