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

#include "DotNet/Runtime.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include "IOlmDiffDocument.h"
#include "IOlmDocument.h"

namespace SE::OtdrEditor
{
    using namespace SE::Core;
    // using namespace SE::Core::EntityComponentSystem::Components;

    // struct sDotNetLinkElement
    // {
    //     int       mRowIndex;
    //     int       mLinkIndex;
    //     int       mSubLinkIndex;
    //     int       mEventIndex;
    //     int       mSubEventIndex;
    //     bool      mIsSubElement;
    //     int       mDiagnosicCount;
    //     ePassFail mLossPassFail;
    //     ePassFail mReflectancePassFail;
    //     void     *mLinkElement;
    //     void     *mPhysicalEvent;
    //     void     *mPeakTrace;
    //     void     *mDetectionTrace;
    //     void     *mAttributes;
    //     void     *mAcquisitionData;
    //     void     *mFiberInfo;
    // };

    static std::string UTF16ToAscii( const char *aPayloadData, size_t aSize )
    {
        size_t      lPayloadSize = static_cast<size_t>( aSize / 2 );
        std::string lItemPathStr( lPayloadSize - 1, '\0' );
        for( uint32_t i = 0; i < lPayloadSize - 1; i++ ) lItemPathStr[i] = aPayloadData[2 * i];

        return lItemPathStr;
    }

    void OtdrWindow::ConfigureUI()
    {
        mTestDialog = New<UIDialog>( "Test", math::vec2{ 640, 480 } );
        mWorkspaceArea.ConfigureUI();
        // mLinkElementTable->OnElementClicked(
        //     [&]( sLinkElement const &aElement )
        //     {
        //         static auto &lOlmMeasurementClass = DotNetRuntime::GetClassType( "Metrino.Olm.OlmPhysicalEvent" );
        //         static auto &lOlmAttributeClass =
        //             DotNetRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );

        //         auto lPhysicalEvent = aElement.mPhysicalEvent;
        //         auto lAttributes    = aElement.mAttributes;

        //         if( *lPhysicalEvent && *lAttributes ) mEventOverview.SetData( lPhysicalEvent, lAttributes );

        //         auto lAcquisitionDataInstance = aElement.mAcquisitionData;
        //         if( aElement.mAcquisitionData && *aElement.mAcquisitionData )
        //             mAcquisitionDataOverview.SetData( aElement.mPeakTrace, aElement.mAcquisitionData, aElement.mFiberInfo );

        //         mTracePlot.Clear();
        //         mTracePlot.SetEventData( aElement, true, true, true );
        //     } );

        mTestFailResultTable->OnElementClicked( [&]( sTestFailElement const &aElement )
                                                { LoadIOlmDiffData( aElement.mFilename, true ); } );

        mMainMenu = UIMenu( "File" );
        // mMainMenu.AddAction( fmt::format( "{} Load iOlm", ICON_FA_PLUS_CIRCLE ), "" )
        //     ->OnTrigger(
        //         [&]()
        //         {
        //             auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
        //                                                     "OLM Files (*.iolm)\0*.iolm\0All Files (*.*)\0*.*\0" );

        //             if( lFilePath.has_value() ) LoadIOlmData( fs::path( lFilePath.value() ) );
        //         } );
        // mMainMenu.AddAction( fmt::format( "{} Load iOlm Diff", ICON_FA_PLUS_CIRCLE ), "" )
        //     ->OnTrigger(
        //         [&]()
        //         {
        //             auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
        //                                                     "OLM Files (*.iolm)\0*.iolm\0All Files (*.*)\0*.*\0" );

        //             if( lFilePath.has_value() ) LoadIOlmDiffData( fs::path( lFilePath.value() ) );
        //         } );

        mMainMenu.AddSeparator();
        mMainMenu.AddAction( fmt::format( "{} Load test report", ICON_FA_PLUS_CIRCLE ), "" )
            ->OnTrigger(
                [&]()
                {
                    auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(),
                                                            "OLM Files (*.iolm)\0*.iolm\0All Files (*.*)\0*.*\0" );

                    if( lFilePath.has_value() ) LoadTestReport( fs::path( lFilePath.value() ).parent_path() );
                } );
        mMainMenu.AddSeparator();
        mMainMenu.AddAction( fmt::format( "{} Exit", ICON_FA_WINDOW_CLOSE_O ), "" )->OnTrigger( [&]() { mRequestQuit = true; } );
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

            if( !lRequestQuit && mApplicationInstance )
            {
                bool lDNRequestQuit = false;
                lDNRequestQuit      = *(bool *)mono_object_unbox( mApplicationInstance->CallMethod( "UpdateMenu" ) );
                lRequestQuit        = lRequestQuit || lDNRequestQuit;
            }

            ImGui::PopStyleVar();
            ImGui::EndMainMenuBar();
        }
        ImGui::PopStyleVar();
        ImGui::End();

        mWorkspaceArea.Update();

        // if( ImGui::Begin( "WS", NULL, ImGuiWindowFlags_None ) )
        // {
        //     mDocumentArea.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
        // }
        // ImGui::End();

        mTestDialog->Update();

        // if( mDataInstance )
        // {
        //     static bool pOpen = true;
        //     if( ImGui::Begin( "iOlmData", &pOpen, ImGuiWindowFlags_None ) )
        //     {
        //         if( ImGui::Button( "All Traces" ) )
        //         {
        //             mTracePlot.Clear();
        //             mTracePlot.SetEventData( mLinkElementVector );
        //         }
        //         mTracePlot.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
        //     }
        //     ImGui::End();

        //     if( ImGui::Begin( "iOlmData_Overview", &pOpen, ImGuiWindowFlags_None ) )
        //     {
        //         mMeasurementOverview.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
        //     }
        //     ImGui::End();

        //     if( ImGui::Begin( "iOlmData_EventOverview", &pOpen, ImGuiWindowFlags_None ) )
        //     {
        //         mEventOverview.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
        //     }
        //     ImGui::End();

        //     if( ImGui::Begin( "iOlmData_AcquisitionDataOverview", &pOpen, ImGuiWindowFlags_None ) )
        //     {
        //         mAcquisitionDataOverview.Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
        //     }
        //     ImGui::End();

        //     if( ImGui::Begin( "iOlmData_LinkElements", &pOpen, ImGuiWindowFlags_None ) )
        //     {
        //         mLinkElementTable->Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
        //     }
        //     ImGui::End();

        //     if( ImGui::Begin( "iOlmData_Events", &pOpen, ImGuiWindowFlags_None ) )
        //     {
        //         mEventTable->Update( ImGui::GetCursorPos(), ImGui::GetContentRegionAvail() );
        //     }
        //     ImGui::End();

        //     if( !pOpen )
        //     {
        //         mDataInstance = nullptr;
        //         pOpen         = true;
        //     }
        // }

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

            static auto &lMonoGlue          = DotNetRuntime::GetClassType( "Metrino.Interop.Instruments" );
            static auto &lModuleDescription = DotNetRuntime::GetClassType( "Metrino.Interop.ModuleDescription" );

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
                    auto lInstance        = DotNetInstance( lModuleDescription.Class(), lConnectedModule );
                    lConnectedModuleNames.push_back( DotNetRuntime::NewString( lInstance.GetFieldValue<MonoString *>( "mName" ) ) );
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

    void OtdrWindow::LoadIOlmDiffData( fs::path aPath, bool aReanalyse )
    {
        mDocumentArea.Add( New<UIIolmDiffDocument>( aPath, aReanalyse ) );
    }

    // void OtdrWindow::LoadIOlmData( fs::path aPath, bool aReanalyse )
    // {
    //     mDocumentArea.Add( New<UIIolmDocument>( aPath, aReanalyse ) );

    //     static auto &lFileLoader = DotNetRuntime::GetClassType( "Metrino.Interop.FileLoader" );
    //     static auto &lFileClass  = DotNetRuntime::GetClassType( "Metrino.Interop.OlmFile" );

    //     MonoString *lFilePath   = DotNetRuntime::NewString( aPath.string() );
    //     MonoObject *lDataObject = lFileLoader.CallMethod( "LoadOlmData", lFilePath );

    //     mDataInstance = New<DotNetInstance>( &lFileClass, lFileClass.Class(), lDataObject );

    //     MonoObject               *lTraceData       = mDataInstance->CallMethod( "GetAllTraces" );
    //     std::vector<MonoObject *> lTraceDataVector = DotNetRuntime::AsVector<MonoObject *>( lTraceData );
    //     if( lTraceDataVector.size() != 0 )
    //     {
    //         static auto &lTraceDataStructure        = DotNetRuntime::GetClassType( "Metrino.Interop.TracePlotData" );
    //         static auto &lSinglePulseTraceClassType = DotNetRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );
    //         static auto &lAcquisitionDataClassType  = DotNetRuntime::GetClassType( "Metrino.Otdr.AcquisitionData" );
    //         static auto &lFiberInfoClassType        = DotNetRuntime::GetClassType( "Metrino.Otdr.PhysicalFiberCharacteristics" );

    //         auto &lTraceDataInstance = DotNetInstance( &lTraceDataStructure, lTraceDataStructure.Class(), lTraceDataVector[0] );

    //         auto lSinglePulseTrace = lTraceDataInstance.GetFieldValue<MonoObject *>( "mTrace" );
    //         auto lSinglePulseTraceInstance =
    //             New<DotNetInstance>( &lSinglePulseTraceClassType, lSinglePulseTraceClassType.Class(), lSinglePulseTrace );

    //         auto lAcquisitionData = lTraceDataInstance.GetFieldValue<MonoObject *>( "mAcquisitionData" );
    //         auto lAcquisitionDataInstance =
    //             New<DotNetInstance>( &lAcquisitionDataClassType, lAcquisitionDataClassType.Class(), lAcquisitionData );

    //         auto lFiberInfo = lTraceDataInstance.GetPropertyValue( "FiberInfo", "Metrino.Otdr.PhysicalFiberCharacteristics" );
    //         mAcquisitionDataOverview.SetData( lSinglePulseTraceInstance, lAcquisitionDataInstance, lFiberInfo );
    //     }

    //     {
    //         MonoObject *lLinkElementData = mDataInstance->CallMethod( "GetLinkElements", &aReanalyse );

    //         auto lLinkElementVector = DotNetRuntime::AsVector<sDotNetLinkElement>( lLinkElementData );

    //         mLinkElementVector = std::vector<sLinkElement>();

    //         for( auto const &x : lLinkElementVector )
    //         {
    //             auto &lElement = mLinkElementVector.emplace_back();

    //             lElement.mRowIndex            = x.mRowIndex;
    //             lElement.mLinkIndex           = x.mLinkIndex;
    //             lElement.mSubLinkIndex        = x.mSubLinkIndex;
    //             lElement.mEventIndex          = x.mEventIndex;
    //             lElement.mSubEventIndex       = x.mSubEventIndex;
    //             lElement.mIsSubElement        = x.mIsSubElement;
    //             lElement.mDiagnosicCount      = x.mDiagnosicCount;
    //             lElement.mLossPassFail        = x.mLossPassFail;
    //             lElement.mReflectancePassFail = x.mReflectancePassFail;

    //             static auto &lBaseLinkElementClass  = DotNetRuntime::GetClassType( "Metrino.Olm.BaseLinkElement" );
    //             static auto &lOlmPhysicalEventClass = DotNetRuntime::GetClassType( "Metrino.Olm.OlmPhysicalEvent" );
    //             static auto &lOlmAttributeClass =
    //                 DotNetRuntime::GetClassType( "Metrino.Olm.SignalProcessing.MultiPulseEventAttribute" );
    //             static auto &lAcquisitionDataClassType = DotNetRuntime::GetClassType( "Metrino.Otdr.AcquisitionData" );
    //             static auto &lSinglePulseTraceClass    = DotNetRuntime::GetClassType( "Metrino.Otdr.SinglePulseTrace" );
    //             static auto &lFiberInfoClassType       = DotNetRuntime::GetClassType( "Metrino.Otdr.PhysicalFiberCharacteristics" );

    //             lElement.mLinkElement = New<DotNetInstance>( &lBaseLinkElementClass, lBaseLinkElementClass.Class(), x.mLinkElement
    //             ); lElement.mPhysicalEvent =
    //                 New<DotNetInstance>( &lOlmPhysicalEventClass, lOlmPhysicalEventClass.Class(), x.mPhysicalEvent );
    //             lElement.mPeakTrace = New<DotNetInstance>( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), x.mPeakTrace );
    //             lElement.mDetectionTrace =
    //                 New<DotNetInstance>( &lSinglePulseTraceClass, lSinglePulseTraceClass.Class(), x.mDetectionTrace );
    //             lElement.mAttributes = New<DotNetInstance>( &lOlmAttributeClass, lOlmAttributeClass.Class(), x.mAttributes );
    //             lElement.mAcquisitionData =
    //                 New<DotNetInstance>( &lAcquisitionDataClassType, lAcquisitionDataClassType.Class(), x.mAcquisitionData );
    //             lElement.mFiberInfo = New<DotNetInstance>( &lFiberInfoClassType, lFiberInfoClassType.Class(), x.mFiberInfo );
    //         }

    //         mLinkElementTable->SetData( mLinkElementVector );
    //         mTracePlot.Clear();
    //         mTracePlot.SetEventData( mLinkElementVector );

    //         mMeasurementOverview.SetData( mDataInstance );
    //     }
    // }

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

        mMainMenu.Update();

        return mRequestQuit;
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