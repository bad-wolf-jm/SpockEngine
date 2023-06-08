#include "Runtime.h"

#include "Core/File.h"
#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Engine/Engine.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/mono-config.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include <unordered_map>

#ifdef WIN32_LEAN_AND_MEAN
#    undef WIN32_LEAN_AND_MEAN
#endif
#include "Core/FileWatch.hpp"

#include "InteropCalls.h"

// #include "UI/Components/BaseImage.h"
// #include "UI/Components/Button.h"
// #include "UI/Components/CheckBox.h"
// #include "UI/Components/ColorButton.h"
// #include "UI/Components/ComboBox.h"
// #include "UI/Components/Component.h"
// #include "UI/Components/DropdownButton.h"
// #include "UI/Components/Image.h"
// #include "UI/Components/ImageButton.h"
// #include "UI/Components/ImageToggleButton.h"
// #include "UI/Components/Label.h"
// #include "UI/Components/Menu.h"
// #include "UI/Components/Plot.h"
// #include "UI/Components/ProgressBar.h"
// #include "UI/Components/PropertyValue.h"
// #include "UI/Components/Slider.h"
// #include "UI/Components/Table.h"
// #include "UI/Components/TextInput.h"
// #include "UI/Components/TextOverlay.h"
// #include "UI/Components/TextToggleButton.h"
// #include "UI/Components/TreeView.h"
// #include "UI/Components/Workspace.h"
// #include "UI/Components/VectorEdit.h"
// #include "UI/Widgets/FileTree.h"
// #include "UI/UI.h"
// #include "UI/Layouts/Container.h"
// #include "UI/Layouts/Splitter.h"
// #include "UI/Layouts/StackLayout.h"
// #include "UI/Layouts/ZLayout.h"
// #include "UI/Dialog.h"
// #include "UI/Form.h"
// #include "UI/Layouts/BoxLayout.h"

#include "Utils.h"

#include "InternalCalls.h"

namespace fs = std::filesystem;
using namespace SE::MonoInternalCalls;

namespace SE::Core
{
    using PathList     = std::vector<fs::path>;
    using ClassMapping = std::map<std::string, DotNetClass>;

    struct sAssemblyData
    {
        fs::path                 mPath           = "";
        fs::path                 mFilename       = "";
        std::string              mCategory       = "";
        MonoAssembly            *mAssembly       = nullptr;
        MonoImage               *mImage          = nullptr;
        bool                     mNeedsReloading = false;
        bool                     mFileExists     = false;
        std::vector<std::string> mClasses{};

        sAssemblyData()                        = default;
        sAssemblyData( const sAssemblyData & ) = default;
    };

    using AssemblyMapping = std::map<fs::path, sAssemblyData>;

    struct sMonoRuntimeData
    {
        MonoDomain     *mRootDomain = nullptr;
        MonoDomain     *mAppDomain  = nullptr;
        sAssemblyData   mCoreAssembly{};
        PathList        mAppAssemblyFiles = {};
        AssemblyMapping mAssemblies       = {};
        ClassMapping    mClasses          = {};

        std::map<std::string, std::vector<sAssemblyData *>> mCategories;
        HINSTANCE                                           mMonoPosixHelper;
    };

    static sMonoRuntimeData *sRuntimeData = nullptr;

    MonoObject *DotNetRuntime::InstantiateClass( MonoClass *aMonoClass, bool aIsCore )
    {
        MonoObject *aInstance = mono_object_new( sRuntimeData->mAppDomain, aMonoClass );

        return aInstance;
    }

    void DotNetRuntime::LoadCoreAssembly( const fs::path &aFilepath )
    {
        sRuntimeData->mCoreAssembly.mPath     = aFilepath.parent_path();
        sRuntimeData->mCoreAssembly.mFilename = aFilepath.filename();
    }

    void DotNetRuntime::AddAppAssemblyPath( const fs::path &aFilepath, std::string const &aCategory )
    {
        if( std::find( sRuntimeData->mAppAssemblyFiles.begin(), sRuntimeData->mAppAssemblyFiles.end(), aFilepath ) !=
            sRuntimeData->mAppAssemblyFiles.end() )
            return;

        if( !fs::exists( aFilepath.parent_path() ) ) return;

        sRuntimeData->mAppAssemblyFiles.push_back( aFilepath );

        sRuntimeData->mAssemblies.emplace( aFilepath, sAssemblyData{} );
        sRuntimeData->mAssemblies[aFilepath].mPath     = aFilepath.parent_path();
        sRuntimeData->mAssemblies[aFilepath].mFilename = aFilepath.filename();

        Ref<fs::path> lAssemblyFilePath = New<fs::path>( aFilepath );

        if( sRuntimeData->mCategories.find( aCategory ) == sRuntimeData->mCategories.end() )
            sRuntimeData->mCategories[aCategory] = std::vector<sAssemblyData *>{};
        sRuntimeData->mCategories[aCategory].push_back( &sRuntimeData->mAssemblies[aFilepath] );
    }

    void DotNetRuntime::Initialize( fs::path &aMonoPath, const fs::path &aCoreAssemblyPath )
    {
        if( sRuntimeData != nullptr ) return;

        sRuntimeData = new sMonoRuntimeData();

        sRuntimeData->mMonoPosixHelper = LoadLibrary( "C:\\GitLab\\SpockEngine\\ThirdParty\\mono\\bin\\Debug\\MonoPosixHelper.dll" );

        InitMono( aMonoPath );
        RegisterInternalCppFunctions();
        LoadCoreAssembly( aCoreAssemblyPath );
    }

    void DotNetRuntime::Shutdown()
    {
        ShutdownMono();

        delete sRuntimeData;

        sRuntimeData = nullptr;
    }

    void DotNetRuntime::InitMono( fs::path &aMonoPath )
    {
        mono_set_assemblies_path( aMonoPath.string().c_str() );
        mono_config_parse( NULL );

        sRuntimeData->mRootDomain = mono_jit_init( "SpockEngineRuntime" );
    }

    void DotNetRuntime::ShutdownMono()
    {
        mono_domain_set( mono_get_root_domain(), false );
        mono_domain_unload( sRuntimeData->mAppDomain );
        mono_jit_cleanup( sRuntimeData->mRootDomain );

        sRuntimeData->mAppDomain  = nullptr;
        sRuntimeData->mRootDomain = nullptr;
    }

    MonoString *DotNetRuntime::NewString( std::string const &aString )
    {
        return mono_string_new( sRuntimeData->mAppDomain, aString.c_str() );
    }

    std::string DotNetRuntime::NewString( MonoString *aString )
    {
        auto *lCharacters = mono_string_to_utf8( aString );
        auto  lString     = std::string( mono_string_to_utf8( aString ) );
        mono_free( lCharacters );

        return lString;
    }

    void DotNetRuntime::ReloadAssemblies()
    {
        mono_domain_set( mono_get_root_domain(), true );
        if( sRuntimeData->mAppDomain != nullptr ) mono_domain_unload( sRuntimeData->mAppDomain );

        sRuntimeData->mAppDomain = mono_domain_create_appdomain( "SE_Runtime", nullptr );
        mono_domain_set_config( sRuntimeData->mAppDomain, ".", "XXX" );
        mono_domain_set( sRuntimeData->mAppDomain, true );
        sRuntimeData->mCoreAssembly.mAssembly =
            Mono::Utils::LoadMonoAssembly( sRuntimeData->mCoreAssembly.mPath / sRuntimeData->mCoreAssembly.mFilename );
        sRuntimeData->mCoreAssembly.mImage = mono_assembly_get_image( sRuntimeData->mCoreAssembly.mAssembly );

        for( auto &[lFile, lData] : sRuntimeData->mAssemblies )
        {
            if( !fs::exists( lFile ) ) continue;

            lData.mAssembly = Mono::Utils::LoadMonoAssembly( lData.mPath / lData.mFilename );
            lData.mImage    = mono_assembly_get_image( lData.mAssembly );
        }
    }

    DotNetClass &DotNetRuntime::GetClassType( const std::string &aClassName )
    {
        if( sRuntimeData->mClasses.find( aClassName ) != sRuntimeData->mClasses.end() ) return sRuntimeData->mClasses[aClassName];

        for( auto const &[lPath, lAssembly] : sRuntimeData->mAssemblies )
        {
            std::size_t lPos       = aClassName.find_last_of( "." );
            std::string lNameSpace = aClassName.substr( 0, lPos );
            std::string lClassName = aClassName.substr( lPos + 1 );

            MonoClass *lClass = mono_class_from_name( lAssembly.mImage, lNameSpace.c_str(), lClassName.c_str() );
            if( lClass != nullptr )
            {
                sRuntimeData->mClasses[aClassName] = DotNetClass( lClass, aClassName, lClassName, lAssembly.mImage, lPath, true );

                return sRuntimeData->mClasses[aClassName];
            }
        }
        return DotNetClass{};
    }

    MonoType *DotNetRuntime::GetCoreTypeFromName( std::string &aName )
    {
        MonoType *lMonoType = mono_reflection_type_from_name( aName.data(), sRuntimeData->mCoreAssembly.mImage );
        if( !lMonoType )
        {
            SE::Logging::Info( "Could not find type '{}'", aName );

            return nullptr;
        }

        return lMonoType;
    }

    static MonoString *OpenFile( MonoString *aFilter )
    {
        auto  lFilter     = DotNetRuntime::NewString( aFilter );
        char *lCharacters = lFilter.data();

        for( uint32_t i = 0; i < lFilter.size(); i++ ) lCharacters[i] = ( lCharacters[i] == '|' ) ? '\0' : lCharacters[i];
        auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(), lFilter.c_str() );

        if( lFilePath.has_value() ) return DotNetRuntime::NewString( lFilePath.value() );

        return DotNetRuntime::NewString( "" );
    }

    static void ICall( std::string const &aName, void *aFunction )
    {
        auto lFullName = fmt::format( "SpockEngine.{}", aName );

        mono_add_internal_call( lFullName.c_str(), aFunction );
    }

    void DotNetRuntime::RegisterInternalCppFunctions()
    {
        using namespace SE::Core::Interop;

        ICall( "UIColor::GetStyleColor", SE::Core::UI::GetStyleColor );

        ICall( "CppCall::OpenFile", OpenFile );
        ICall( "CppCall::Entity_Create", Entity_Create );
        ICall( "CppCall::Entity_IsValid", Entity_IsValid );
        ICall( "CppCall::Entity_Has", Entity_Has );
        ICall( "CppCall::Entity_Get", Entity_Get );
        ICall( "CppCall::Entity_Add", Entity_Add );
        ICall( "CppCall::Entity_Replace", Entity_Replace );

        ICall( "Interop::UIComponent_SetIsVisible", UIComponent_SetIsVisible );
        ICall( "Interop::UIComponent_SetIsEnabled", UIComponent_SetIsEnabled );
        ICall( "Interop::UIComponent_SetAllowDragDrop", UIComponent_SetAllowDragDrop );
        ICall( "Interop::UIComponent_SetPaddingAll", UIComponent_SetPaddingAll );
        ICall( "Interop::UIComponent_SetPaddingPairs", UIComponent_SetPaddingPairs );
        ICall( "Interop::UIComponent_SetPaddingIndividual", UIComponent_SetPaddingIndividual );
        ICall( "Interop::UIComponent_SetAlignment", UIComponent_SetAlignment );
        ICall( "Interop::UIComponent_SetHorizontalAlignment", UIComponent_SetHorizontalAlignment );
        ICall( "Interop::UIComponent_SetVerticalAlignment", UIComponent_SetVerticalAlignment );
        ICall( "Interop::UIComponent_SetBackgroundColor", UIComponent_SetBackgroundColor );
        ICall( "Interop::UIComponent_SetFont", UIComponent_SetFont );
        ICall( "Interop::UIComponent_SetTooltip", UIComponent_SetTooltip );

        ICall( "Interop::UIForm_Create", UIForm_Create );
        ICall( "Interop::UIForm_Destroy", UIForm_Destroy );
        ICall( "Interop::UIForm_SetTitle", UIForm_SetTitle );
        ICall( "Interop::UIForm_SetContent", UIForm_SetContent );
        ICall( "Interop::UIForm_Update", UIForm_Update );
        ICall( "Interop::UIForm_SetSize", UIForm_SetSize );

        ICall( "Interop::UIDialog_Create", UIDialog_Create );
        ICall( "Interop::UIDialog_CreateWithTitleAndSize", UIDialog_CreateWithTitleAndSize );
        ICall( "Interop::UIDialog_Destroy", UIDialog_Destroy );
        ICall( "Interop::UIDialog_SetTitle", UIDialog_SetTitle );
        ICall( "Interop::UIDialog_SetSize", UIDialog_SetSize );
        ICall( "Interop::UIDialog_SetContent", UIDialog_SetContent );
        ICall( "Interop::UIDialog_Open", UIDialog_Open );
        ICall( "Interop::UIDialog_Close", UIDialog_Close );
        ICall( "Interop::UIDialog_Update", UIDialog_Update );

        ICall( "Interop::UILabel_Create", UILabel_Create );
        ICall( "Interop::UILabel_CreateWithText", UILabel_CreateWithText );
        ICall( "Interop::UILabel_Destroy", UILabel_Destroy );
        ICall( "Interop::UILabel_SetText", UILabel_SetText );
        ICall( "Interop::UILabel_SetTextColor", UILabel_SetTextColor );

        ICall( "Interop::UIBaseImage_Create", UIBaseImage_Create );
        ICall( "Interop::UIBaseImage_CreateWithPath", UIBaseImage_CreateWithPath );
        ICall( "Interop::UIBaseImage_Destroy", UIBaseImage_Destroy );
        ICall( "Interop::UIBaseImage_SetImage", UIBaseImage_SetImage );
        ICall( "Interop::UIBaseImage_SetSize", UIBaseImage_SetSize );
        ICall( "Interop::UIBaseImage_GetSize", UIBaseImage_GetSize );
        ICall( "Interop::UIBaseImage_SetTopLeft", UIBaseImage_SetTopLeft );
        ICall( "Interop::UIBaseImage_GetTopLeft", UIBaseImage_GetTopLeft );
        ICall( "Interop::UIBaseImage_SetBottomRight", UIBaseImage_SetBottomRight );
        ICall( "Interop::UIBaseImage_GetBottomRight", UIBaseImage_GetBottomRight );
        ICall( "Interop::UIBaseImage_SetTintColor", UIBaseImage_SetTintColor );
        ICall( "Interop::UIBaseImage_GetTintColor", UIBaseImage_GetTintColor );

        ICall( "Interop::UIImage_Create", UIImage_Create );
        ICall( "Interop::UIImage_CreateWithPath", UIImage_CreateWithPath );
        ICall( "Interop::UIImage_Destroy", UIImage_Destroy );

        ICall( "Interop::UIImageButton_Create", UIImageButton_Create );
        ICall( "Interop::UIImageButton_CreateWithPath", UIImageButton_CreateWithPath );
        ICall( "Interop::UIImageButton_Destroy", UIImageButton_Destroy );
        ICall( "Interop::UIImageButton_OnClick", UIImageButton_OnClick );

        ICall( "Interop::UIImageToggleButton_Create", UIImageToggleButton_Create );
        ICall( "Interop::UIImageToggleButton_Destroy", UIImageToggleButton_Destroy );
        ICall( "Interop::UIImageToggleButton_OnClicked", UIImageToggleButton_OnClicked );
        ICall( "Interop::UIImageToggleButton_OnChanged", UIImageToggleButton_OnChanged );
        ICall( "Interop::UIImageToggleButton_IsActive", UIImageToggleButton_IsActive );
        ICall( "Interop::UIImageToggleButton_SetActive", UIImageToggleButton_SetActive );
        ICall( "Interop::UIImageToggleButton_SetActiveImage", UIImageToggleButton_SetActiveImage );
        ICall( "Interop::UIImageToggleButton_SetInactiveImage", UIImageToggleButton_SetInactiveImage );

        ICall( "Interop::UIButton_Create", UIButton_Create );
        ICall( "Interop::UIButton_CreateWithText", UIButton_CreateWithText );
        ICall( "Interop::UIButton_Destroy", UIButton_Destroy );
        ICall( "Interop::UIButton_SetText", UIButton_SetText );
        ICall( "Interop::UIButton_OnClick", UIButton_OnClick );

        ICall( "Interop::UITextToggleButton_Create", UITextToggleButton_Create );
        ICall( "Interop::UITextToggleButton_CreateWithText", UITextToggleButton_CreateWithText );
        ICall( "Interop::UITextToggleButton_Destroy", UITextToggleButton_Destroy );
        ICall( "Interop::UITextToggleButton_OnClicked", UITextToggleButton_OnClicked );
        ICall( "Interop::UITextToggleButton_OnChanged", UITextToggleButton_OnChanged );

        ICall( "Interop::UITextToggleButton_IsActive", UITextToggleButton_IsActive );
        ICall( "Interop::UITextToggleButton_SetActive", UITextToggleButton_SetActive );
        ICall( "Interop::UITextToggleButton_SetActiveColor", UITextToggleButton_SetActiveColor );
        ICall( "Interop::UITextToggleButton_SetInactiveColor", UITextToggleButton_SetInactiveColor );

        ICall( "Interop::UICheckBox_Create", UICheckBox_Create );
        ICall( "Interop::UICheckBox_Destroy", UICheckBox_Destroy );
        ICall( "Interop::UICheckBox_OnClick", UICheckBox_OnClick );
        ICall( "Interop::UICheckBox_IsChecked", UICheckBox_IsChecked );
        ICall( "Interop::UICheckBox_SetIsChecked", UICheckBox_SetIsChecked );

        ICall( "Interop::UIComboBox_Create", UIComboBox_Create );
        ICall( "Interop::UIComboBox_CreateWithItems", UIComboBox_CreateWithItems );
        ICall( "Interop::UIComboBox_Destroy", UIComboBox_Destroy );
        ICall( "Interop::UIComboBox_GetCurrent", UIComboBox_GetCurrent );
        ICall( "Interop::UIComboBox_SetCurrent", UIComboBox_SetCurrent );
        ICall( "Interop::UIComboBox_SetItemList", UIComboBox_SetItemList );
        ICall( "Interop::UIComboBox_OnChanged", UIComboBox_OnChanged );

        ICall( "Interop::UIBoxLayout_CreateWithOrientation", UIBoxLayout_CreateWithOrientation );
        ICall( "Interop::UIBoxLayout_Destroy", UIBoxLayout_Destroy );
        ICall( "Interop::UIBoxLayout_AddAlignedNonFixed", UIBoxLayout_AddAlignedNonFixed );
        ICall( "Interop::UIBoxLayout_AddNonAlignedNonFixed", UIBoxLayout_AddNonAlignedNonFixed );
        ICall( "Interop::UIBoxLayout_AddAlignedFixed", UIBoxLayout_AddAlignedFixed );
        ICall( "Interop::UIBoxLayout_AddNonAlignedFixed", UIBoxLayout_AddNonAlignedFixed );
        ICall( "Interop::UIBoxLayout_AddSeparator", UIBoxLayout_AddSeparator );
        ICall( "Interop::UIBoxLayout_SetItemSpacing", UIBoxLayout_SetItemSpacing );
        ICall( "Interop::UIBoxLayout_Clear", UIBoxLayout_Clear );

        ICall( "Interop::UIZLayout_Create", UIZLayout_Create );
        ICall( "Interop::UIZLayout_Destroy", UIZLayout_Destroy );
        ICall( "Interop::UIZLayout_AddAlignedNonFixed", UIZLayout_AddAlignedNonFixed );
        ICall( "Interop::UIZLayout_AddNonAlignedNonFixed", UIZLayout_AddNonAlignedNonFixed );
        ICall( "Interop::UIZLayout_AddAlignedFixed", UIZLayout_AddAlignedFixed );
        ICall( "Interop::UIZLayout_AddNonAlignedFixed", UIZLayout_AddNonAlignedFixed );

        ICall( "Interop::UIStackLayout_Create", UIStackLayout_Create );
        ICall( "Interop::UIStackLayout_Destroy", UIStackLayout_Destroy );
        ICall( "Interop::UIStackLayout_Add", UIStackLayout_Add );
        ICall( "Interop::UIStackLayout_SetCurrent", UIStackLayout_SetCurrent );

        ICall( "Interop::UISplitter_Create", UISplitter_Create );
        ICall( "Interop::UISplitter_CreateWithOrientation", UISplitter_CreateWithOrientation );
        ICall( "Interop::UISplitter_Destroy", UISplitter_Destroy );
        ICall( "Interop::UISplitter_Add1", UISplitter_Add1 );
        ICall( "Interop::UISplitter_Add2", UISplitter_Add2 );
        ICall( "Interop::UISplitter_SetItemSpacing", UISplitter_SetItemSpacing );

        ICall( "Interop::UITableColumn_SetTooltip", UITableColumn_SetTooltip );
        ICall( "Interop::UITableColumn_SetForegroundColor", UITableColumn_SetForegroundColor );
        ICall( "Interop::UITableColumn_SetBackgroundColor", UITableColumn_SetBackgroundColor );

        ICall( "Interop::UIFloat64Column_Create", UIFloat64Column_Create );
        ICall( "Interop::UIFloat64Column_CreateFull", UIFloat64Column_CreateFull );
        ICall( "Interop::UIFloat64Column_Destroy", UIFloat64Column_Destroy );
        ICall( "Interop::UIFloat64Column_Clear", UIFloat64Column_Clear );
        ICall( "Interop::UIFloat64Column_SetData", UIFloat64Column_SetData );

        ICall( "Interop::UIUint32Column_Create", UIUint32Column_Create );
        ICall( "Interop::UIUint32Column_CreateFull", UIUint32Column_CreateFull );
        ICall( "Interop::UIUint32Column_Destroy", UIUint32Column_Destroy );
        ICall( "Interop::UIUint32Column_Clear", UIUint32Column_Clear );
        ICall( "Interop::UIUint32Column_SetData", UIUint32Column_SetData );

        ICall( "Interop::UIStringColumn_Create", UIStringColumn_Create );
        ICall( "Interop::UIStringColumn_CreateFull", UIStringColumn_CreateFull );
        ICall( "Interop::UIStringColumn_Destroy", UIStringColumn_Destroy );
        ICall( "Interop::UIStringColumn_Clear", UIStringColumn_Clear );
        ICall( "Interop::UIStringColumn_SetData", UIStringColumn_SetData );

        ICall( "Interop::UITable_Create", UITable_Create );
        ICall( "Interop::UITable_Destroy", UITable_Destroy );
        ICall( "Interop::UITable_OnRowClicked", UITable_OnRowClicked );
        ICall( "Interop::UITable_AddColumn", UITable_AddColumn );
        ICall( "Interop::UITable_SetRowHeight", UITable_SetRowHeight );
        ICall( "Interop::UITable_SetRowBackgroundColor", UITable_SetRowBackgroundColor );
        ICall( "Interop::UITable_ClearRowBackgroundColor", UITable_ClearRowBackgroundColor );
        ICall( "Interop::UITable_SetDisplayedRowIndices", UITable_SetDisplayedRowIndices );

        ICall( "Interop::UIPlot_Create", UIPlot_Create );
        ICall( "Interop::UIPlot_Destroy", UIPlot_Destroy );
        ICall( "Interop::UIPlot_Clear", UIPlot_Clear );
        ICall( "Interop::UIPlot_ConfigureLegend", UIPlot_ConfigureLegend );
        ICall( "Interop::UIPlot_Add", UIPlot_Add );
        ICall( "Interop::UIPlot_SetAxisLimits", UIPlot_SetAxisLimits );
        ICall( "Interop::UIPlot_GetAxisTitle", UIPlot_GetAxisTitle );
        ICall( "Interop::UIPlot_SetAxisTitle", UIPlot_SetAxisTitle );

        ICall( "Interop::UIPlotData_SetThickness", UIPlotData_SetThickness );
        ICall( "Interop::UIPlotData_SetLegend", UIPlotData_SetLegend );
        ICall( "Interop::UIPlotData_SetColor", UIPlotData_SetColor );
        ICall( "Interop::UIPlotData_SetXAxis", UIPlotData_SetXAxis );
        ICall( "Interop::UIPlotData_SetYAxis", UIPlotData_SetYAxis );

        ICall( "Interop::UIFloat64LinePlot_Create", UIFloat64LinePlot_Create );
        ICall( "Interop::UIFloat64LinePlot_Destroy", UIFloat64LinePlot_Destroy );
        ICall( "Interop::UIFloat64LinePlot_SetX", UIFloat64LinePlot_SetX );
        ICall( "Interop::UIFloat64LinePlot_SetY", UIFloat64LinePlot_SetY );

        ICall( "Interop::UIFloat64ScatterPlot_Create", UIFloat64ScatterPlot_Create );
        ICall( "Interop::UIFloat64ScatterPlot_Destroy", UIFloat64ScatterPlot_Destroy );
        ICall( "Interop::UIFloat64ScatterPlot_SetX", UIFloat64ScatterPlot_SetX );
        ICall( "Interop::UIFloat64ScatterPlot_SetY", UIFloat64ScatterPlot_SetY );

        ICall( "Interop::UIVLinePlot_Create", UIVLinePlot_Create );
        ICall( "Interop::UIVLinePlot_Destroy", UIVLinePlot_Destroy );
        ICall( "Interop::UIVLinePlot_SetX", UIVLinePlot_SetX );

        ICall( "Interop::UIHLinePlot_Create", UIHLinePlot_Create );
        ICall( "Interop::UIHLinePlot_Destroy", UIHLinePlot_Destroy );
        ICall( "Interop::UIHLinePlot_SetY", UIHLinePlot_SetY );

        ICall( "Interop::UIAxisTag_Create", UIAxisTag_Create );
        ICall( "Interop::UIAxisTag_CreateWithTextAndColor", UIAxisTag_CreateWithTextAndColor );
        ICall( "Interop::UIAxisTag_Destroy", UIAxisTag_Destroy );
        ICall( "Interop::UIAxisTag_SetX", UIAxisTag_SetX );
        ICall( "Interop::UIAxisTag_SetText", UIAxisTag_SetText );
        ICall( "Interop::UIAxisTag_GetColor", UIAxisTag_GetColor );
        ICall( "Interop::UIAxisTag_SetColor", UIAxisTag_SetColor );

        ICall( "Interop::UIVRangePlot_Create", UIVRangePlot_Create );
        ICall( "Interop::UIVRangePlot_Destroy", UIVRangePlot_Destroy );
        ICall( "Interop::UIVRangePlot_GetMin", UIVRangePlot_GetMin );
        ICall( "Interop::UIVRangePlot_SetMin", UIVRangePlot_SetMin );
        ICall( "Interop::UIVRangePlot_GetMax", UIVRangePlot_GetMax );
        ICall( "Interop::UIVRangePlot_SetMax", UIVRangePlot_SetMax );

        ICall( "Interop::UIHRangePlot_Create", UIHRangePlot_Create );
        ICall( "Interop::UIHRangePlot_Destroy", UIHRangePlot_Destroy );
        ICall( "Interop::UIHRangePlot_GetMin", UIHRangePlot_GetMin );
        ICall( "Interop::UIHRangePlot_SetMin", UIHRangePlot_SetMin );
        ICall( "Interop::UIHRangePlot_GetMax", UIHRangePlot_GetMax );
        ICall( "Interop::UIHRangePlot_SetMax", UIHRangePlot_SetMax );

        ICall( "Interop::UITextOverlay_Create", UITextOverlay_Create );
        ICall( "Interop::UITextOverlay_Destroy", UITextOverlay_Destroy );
        ICall( "Interop::UITextOverlay_AddText", UITextOverlay_AddText );
        ICall( "Interop::UITextOverlay_Clear", UITextOverlay_Clear );

        ICall( "Interop::UIWorkspace_Create", UIWorkspace_Create );
        ICall( "Interop::UIWorkspace_Destroy", UIWorkspace_Destroy );
        ICall( "Interop::UIWorkspace_Add", UIWorkspace_Add );
        ICall( "Interop::UIWorkspace_RegisterCloseDocumentDelegate", UIWorkspace_RegisterCloseDocumentDelegate );

        ICall( "Interop::UIWorkspaceDocument_Create", UIWorkspaceDocument_Create );
        ICall( "Interop::UIWorkspaceDocument_Destroy", UIWorkspaceDocument_Destroy );
        ICall( "Interop::UIWorkspaceDocument_SetContent", UIWorkspaceDocument_SetContent );
        ICall( "Interop::UIWorkspaceDocument_Update", UIWorkspaceDocument_Update );
        ICall( "Interop::UIWorkspaceDocument_SetName", UIWorkspaceDocument_SetName );
        ICall( "Interop::UIWorkspaceDocument_IsDirty", UIWorkspaceDocument_IsDirty );
        ICall( "Interop::UIWorkspaceDocument_MarkAsDirty", UIWorkspaceDocument_MarkAsDirty );
        ICall( "Interop::UIWorkspaceDocument_Open", UIWorkspaceDocument_Open );
        ICall( "Interop::UIWorkspaceDocument_RequestClose", UIWorkspaceDocument_RequestClose );
        ICall( "Interop::UIWorkspaceDocument_ForceClose", UIWorkspaceDocument_ForceClose );
        ICall( "Interop::UIWorkspaceDocument_RegisterSaveDelegate", UIWorkspaceDocument_RegisterSaveDelegate );

        ICall( "Interop::UIMenuItem_Create", UIMenuItem_Create );
        ICall( "Interop::UIMenuItem_CreateWithText", UIMenuItem_CreateWithText );
        ICall( "Interop::UIMenuItem_CreateWithTextAndShortcut", UIMenuItem_CreateWithTextAndShortcut );
        ICall( "Interop::UIMenuItem_Destroy", UIMenuItem_Destroy );
        ICall( "Interop::UIMenuItem_SetText", UIMenuItem_SetText );
        ICall( "Interop::UIMenuItem_SetShortcut", UIMenuItem_SetShortcut );
        ICall( "Interop::UIMenuItem_SetTextColor", UIMenuItem_SetTextColor );
        ICall( "Interop::UIMenuItem_OnTrigger", UIMenuItem_OnTrigger );

        ICall( "Interop::UIMenuSeparator_Create", UIMenuSeparator_Create );
        ICall( "Interop::UIMenuSeparator_Destroy", UIMenuSeparator_Destroy );

        ICall( "Interop::UIMenu_Create", UIMenu_Create );
        ICall( "Interop::UIMenu_CreateWithText", UIMenu_CreateWithText );
        ICall( "Interop::UIMenu_Destroy", UIMenu_Destroy );
        ICall( "Interop::UIMenu_AddAction", UIMenu_AddAction );
        ICall( "Interop::UIMenu_AddMenu", UIMenu_AddMenu );
        ICall( "Interop::UIMenu_AddSeparator", UIMenu_AddSeparator );
        ICall( "Interop::UIMenu_Update", UIMenu_Update );

        ICall( "Interop::UIPropertyValue_Create", UIPropertyValue_Create );
        ICall( "Interop::UIPropertyValue_CreateWithText", UIPropertyValue_CreateWithText );
        ICall( "Interop::UIPropertyValue_CreateWithTextAndOrientation", UIPropertyValue_CreateWithTextAndOrientation );
        ICall( "Interop::UIPropertyValue_Destroy", UIPropertyValue_Destroy );
        ICall( "Interop::UIPropertyValue_SetValue", UIPropertyValue_SetValue );
        ICall( "Interop::UIPropertyValue_SetValueFont", UIPropertyValue_SetValueFont );
        ICall( "Interop::UIPropertyValue_SetNameFont", UIPropertyValue_SetNameFont );

        ICall( "Interop::UITextInput_Create", UITextInput_Create );
        ICall( "Interop::UITextInput_CreateWithText", UITextInput_CreateWithText );
        ICall( "Interop::UITextInput_Destroy", UITextInput_Destroy );
        ICall( "Interop::UITextInput_GetText", UITextInput_GetText );
        ICall( "Interop::UITextInput_SetHintText", UITextInput_SetHintText );
        ICall( "Interop::UITextInput_SetTextColor", UITextInput_SetTextColor );
        ICall( "Interop::UITextInput_SetBufferSize", UITextInput_SetBufferSize );
        ICall( "Interop::UITextInput_OnTextChanged", UITextInput_OnTextChanged );

        ICall( "Interop::UIProgressBar_Create", UIProgressBar_Create );
        ICall( "Interop::UIProgressBar_Destroy", UIProgressBar_Destroy );
        ICall( "Interop::UIProgressBar_SetProgressValue", UIProgressBar_SetProgressValue );
        ICall( "Interop::UIProgressBar_SetProgressColor", UIProgressBar_SetProgressColor );
        ICall( "Interop::UIProgressBar_SetText", UIProgressBar_SetText );
        ICall( "Interop::UIProgressBar_SetTextColor", UIProgressBar_SetTextColor );
        ICall( "Interop::UIProgressBar_SetThickness", UIProgressBar_SetThickness );

        ICall( "Interop::UIDropdownButton_Create", UIDropdownButton_Create );
        ICall( "Interop::UIDropdownButton_Destroy", UIDropdownButton_Destroy );
        ICall( "Interop::UIDropdownButton_SetContent", UIDropdownButton_SetContent );
        ICall( "Interop::UIDropdownButton_SetContentSize", UIDropdownButton_SetContentSize );
        ICall( "Interop::UIDropdownButton_SetImage", UIDropdownButton_SetImage );
        ICall( "Interop::UIDropdownButton_SetText", UIDropdownButton_SetText );
        ICall( "Interop::UIDropdownButton_SetTextColor", UIDropdownButton_SetTextColor );

        ICall( "Interop::UIContainer_Create", UIContainer_Create );
        ICall( "Interop::UIContainer_Destroy", UIContainer_Destroy );
        ICall( "Interop::UIContainer_SetContent", UIContainer_SetContent );

        ICall( "Interop::UITreeView_Create", UITreeView_Create );
        ICall( "Interop::UITreeView_Destroy", UITreeView_Destroy );
        ICall( "Interop::UITreeView_SetIndent", UITreeView_SetIndent );
        ICall( "Interop::UITreeView_SetIconSpacing", UITreeView_SetIconSpacing );
        ICall( "Interop::UITreeView_Add", UITreeView_Add );

        ICall( "Interop::UITreeViewNode_Create", UITreeViewNode_Create );
        ICall( "Interop::UITreeViewNode_Destroy", UITreeViewNode_Destroy );
        ICall( "Interop::UITreeViewNode_SetIcon", UITreeViewNode_SetIcon );
        ICall( "Interop::UITreeViewNode_SetIndicator", UITreeViewNode_SetIndicator );
        ICall( "Interop::UITreeViewNode_SetText", UITreeViewNode_SetText );
        ICall( "Interop::UITreeViewNode_SetTextColor", UITreeViewNode_SetTextColor );
        ICall( "Interop::UITreeViewNode_Add", UITreeViewNode_Add );

        ICall( "Interop::UIFileTree_Create", UIFileTree_Create );
        ICall( "Interop::UIFileTree_Destroy", UIFileTree_Destroy );
        ICall( "Interop::UIFileTree_Add", UIFileTree_Add );

        ICall( "Interop::UIColorButton_Create", UIColorButton_Create );
        ICall( "Interop::UIColorButton_Destroy", UIColorButton_Destroy );

        ICall( "Interop::UISlider_Create", UISlider_Create );
        ICall( "Interop::UISlider_Destroy", UISlider_Destroy );

        ICall( "Interop::UIVec2Input_Create", UIVec2Input_Create );
        ICall( "Interop::UIVec2Input_Destroy", UIVec2Input_Destroy );
        ICall( "Interop::UIVec2Input_OnChanged", UIVec2Input_OnChanged );
        ICall( "Interop::UIVec2Input_SetValue", UIVec2Input_SetValue );
        ICall( "Interop::UIVec2Input_GetValue", UIVec2Input_GetValue );
        ICall( "Interop::UIVec2Input_SetFormat", UIVec2Input_SetFormat );

        ICall( "Interop::UIVec3Input_Create", UIVec3Input_Create );
        ICall( "Interop::UIVec3Input_Destroy", UIVec3Input_Destroy );
        ICall( "Interop::UIVec3Input_OnChanged", UIVec3Input_OnChanged );
        ICall( "Interop::UIVec3Input_SetValue", UIVec3Input_SetValue );
        ICall( "Interop::UIVec3Input_GetValue", UIVec3Input_GetValue );
        ICall( "Interop::UIVec3Input_SetFormat", UIVec3Input_SetFormat );

        ICall( "Interop::UIVec4Input_Create", UIVec4Input_Create );
        ICall( "Interop::UIVec4Input_Destroy", UIVec4Input_Destroy );
        ICall( "Interop::UIVec4Input_OnChanged", UIVec4Input_OnChanged );
        ICall( "Interop::UIVec4Input_SetValue", UIVec4Input_SetValue );
        ICall( "Interop::UIVec4Input_GetValue", UIVec4Input_GetValue );
        ICall( "Interop::UIVec4Input_GetValue", UIVec4Input_SetFormat );
    }

} // namespace SE::Core