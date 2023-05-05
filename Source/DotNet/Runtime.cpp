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

#include "UI/Components/BaseImage.h"
#include "UI/Components/Button.h"
#include "UI/Components/CheckBox.h"
#include "UI/Components/ComboBox.h"
#include "UI/Components/Component.h"
#include "UI/Components/Image.h"
#include "UI/Components/ImageButton.h"
#include "UI/Components/ImageToggleButton.h"
#include "UI/Components/Label.h"
#include "UI/Components/Menu.h"
#include "UI/Components/Plot.h"
#include "UI/Components/ProgressBar.h"
#include "UI/Components/PropertyValue.h"
#include "UI/Components/Table.h"
#include "UI/Components/TextInput.h"
#include "UI/Components/TextOverlay.h"
#include "UI/Components/TextToggleButton.h"
#include "UI/Components/Workspace.h"
#include "UI/UI.h"

#include "UI/Layouts/Splitter.h"
#include "UI/Layouts/StackLayout.h"
#include "UI/Layouts/ZLayout.h"

#include "UI/Dialog.h"
#include "UI/Form.h"
#include "UI/Layouts/BoxLayout.h"

#include "Utils.h"

namespace fs = std::filesystem;

namespace SE::Core
{
    using PathList     = std::vector<fs::path>;
    using ClassMapping = std::map<std::string, DotNetClass>;

    struct sAssemblyData
    {
        fs::path      mPath           = "";
        fs::path      mFilename       = "";
        std::string   mCategory       = "";
        MonoAssembly *mAssembly       = nullptr;
        MonoImage    *mImage          = nullptr;
        bool          mNeedsReloading = false;
        bool          mFileExists     = false;

        std::vector<std::string> mClasses{};

        std::shared_ptr<filewatch::FileWatch<std::string>> mWatcher{};

        sAssemblyData()                        = default;
        sAssemblyData( const sAssemblyData & ) = default;
    };

    using AssemblyMapping = std::map<fs::path, sAssemblyData>;

    struct sMonoRuntimeData
    {
        MonoDomain *mRootDomain = nullptr;
        MonoDomain *mAppDomain  = nullptr;

        sAssemblyData mCoreAssembly{};
        // ClassMapping  mCoreClasses = {};

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

    static void OnAppAssemblyFileSystemEvent( const fs::path &path, const filewatch::Event change_type )
    {
        switch( change_type )
        {
        case filewatch::Event::modified: sRuntimeData->mAssemblies[path].mNeedsReloading = true; break;
        case filewatch::Event::removed: sRuntimeData->mAssemblies[path].mFileExists = false; break;
        case filewatch::Event::added: sRuntimeData->mAssemblies[path].mFileExists = true; break;
        default: break;
        }
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

        sRuntimeData->mAssemblies[aFilepath].mWatcher = std::make_shared<filewatch::FileWatch<std::string>>(
            aFilepath.parent_path().string(),
            [lAssemblyFilePath]( const std::string &path, const filewatch::Event change_type )
            {
                if( lAssemblyFilePath->filename().string() == path )
                {
                    OnAppAssemblyFileSystemEvent( *lAssemblyFilePath, change_type );
                }
            } );

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

    void DotNetRuntime::RegisterInternalCppFunctions()
    {
        mono_add_internal_call( "SpockEngine.CppCall::OpenFile", OpenFile );
        mono_add_internal_call( "SpockEngine.UIColor::GetStyleColor", SE::Core::UI::GetStyleColor );

        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetIsVisible", UIComponent::UIComponent_SetIsVisible );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetIsEnabled", UIComponent::UIComponent_SetIsEnabled );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetAllowDragDrop", UIComponent::UIComponent_SetAllowDragDrop );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetPaddingAll", UIComponent::UIComponent_SetPaddingAll );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetPaddingPairs", UIComponent::UIComponent_SetPaddingPairs );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetPaddingIndividual",
                                UIComponent::UIComponent_SetPaddingIndividual );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetAlignment", UIComponent::UIComponent_SetAlignment );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetHorizontalAlignment",
                                UIComponent::UIComponent_SetHorizontalAlignment );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetVerticalAlignment",
                                UIComponent::UIComponent_SetVerticalAlignment );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetBackgroundColor",
                                UIComponent::UIComponent_SetBackgroundColor );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetFont", UIComponent::UIComponent_SetFont );
        mono_add_internal_call( "SpockEngine.UIComponent::UIComponent_SetTooltip", UIComponent::UIComponent_SetTooltip );

        mono_add_internal_call( "SpockEngine.UIForm::UIForm_Create", UIForm::UIForm_Create );
        mono_add_internal_call( "SpockEngine.UIForm::UIForm_Destroy", UIForm::UIForm_Destroy );
        mono_add_internal_call( "SpockEngine.UIForm::UIForm_SetTitle", UIForm::UIForm_SetTitle );
        mono_add_internal_call( "SpockEngine.UIForm::UIForm_SetContent", UIForm::UIForm_SetContent );
        mono_add_internal_call( "SpockEngine.UIForm::UIForm_Update", UIForm::UIForm_Update );

        mono_add_internal_call( "SpockEngine.UIDialog::UIDialog_Create", UIDialog::UIDialog_Create );
        mono_add_internal_call( "SpockEngine.UIDialog::UIDialog_CreateWithTitleAndSize", UIDialog::UIDialog_CreateWithTitleAndSize );
        mono_add_internal_call( "SpockEngine.UIDialog::UIDialog_Destroy", UIDialog::UIDialog_Destroy );
        mono_add_internal_call( "SpockEngine.UIDialog::UIDialog_SetTitle", UIDialog::UIDialog_SetTitle );
        mono_add_internal_call( "SpockEngine.UIDialog::UIDialog_SetSize", UIDialog::UIDialog_SetSize );
        mono_add_internal_call( "SpockEngine.UIDialog::UIDialog_SetContent", UIDialog::UIDialog_SetContent );
        mono_add_internal_call( "SpockEngine.UIDialog::UIDialog_Open", UIDialog::UIDialog_Open );
        mono_add_internal_call( "SpockEngine.UIDialog::UIDialog_Close", UIDialog::UIDialog_Close );
        mono_add_internal_call( "SpockEngine.UIDialog::UIDialog_Update", UIDialog::UIDialog_Update );

        mono_add_internal_call( "SpockEngine.UILabel::UILabel_Create", UILabel::UILabel_Create );
        mono_add_internal_call( "SpockEngine.UILabel::UILabel_CreateWithText", UILabel::UILabel_CreateWithText );
        mono_add_internal_call( "SpockEngine.UILabel::UILabel_Destroy", UILabel::UILabel_Destroy );
        mono_add_internal_call( "SpockEngine.UILabel::UILabel_SetText", UILabel::UILabel_SetText );
        mono_add_internal_call( "SpockEngine.UILabel::UILabel_SetTextColor", UILabel::UILabel_SetTextColor );

        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_Create", UIBaseImage::UIBaseImage_Create );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_CreateWithPath", UIBaseImage::UIBaseImage_CreateWithPath );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_Destroy", UIBaseImage::UIBaseImage_Destroy );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_SetImage", UIBaseImage::UIBaseImage_SetImage );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_SetSize", UIBaseImage::UIBaseImage_SetSize );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_GetSize", UIBaseImage::UIBaseImage_GetSize );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_SetTopLeft", UIBaseImage::UIBaseImage_SetTopLeft );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_GetTopLeft", UIBaseImage::UIBaseImage_GetTopLeft );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_SetBottomRight", UIBaseImage::UIBaseImage_SetBottomRight );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_GetBottomRight", UIBaseImage::UIBaseImage_GetBottomRight );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_SetTintColor", UIBaseImage::UIBaseImage_SetTintColor );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_GetTintColor", UIBaseImage::UIBaseImage_GetTintColor );

        mono_add_internal_call( "SpockEngine.UIImage::UIImage_Create", UIImage::UIImage_Create );
        mono_add_internal_call( "SpockEngine.UIImage::UIImage_CreateWithPath", UIImage::UIImage_CreateWithPath );
        mono_add_internal_call( "SpockEngine.UIImage::UIImage_Destroy", UIImage::UIImage_Destroy );

        mono_add_internal_call( "SpockEngine.UIImageButton::UIImageButton_Create", UIImageButton::UIImageButton_Create );
        mono_add_internal_call( "SpockEngine.UIImageButton::UIImageButton_CreateWithPath",
                                UIImageButton::UIImageButton_CreateWithPath );
        mono_add_internal_call( "SpockEngine.UIImageButton::UIImageButton_Destroy", UIImageButton::UIImageButton_Destroy );
        mono_add_internal_call( "SpockEngine.UIImageButton::UIImageButton_OnClick", UIImageButton::UIImageButton_OnClick );

        mono_add_internal_call( "SpockEngine.UIImageToggleButton::UIImageToggleButton_Create",
                                UIImageToggleButton::UIImageToggleButton_Create );
        mono_add_internal_call( "SpockEngine.UIImageToggleButton::UIImageToggleButton_Destroy",
                                UIImageToggleButton::UIImageToggleButton_Destroy );
        mono_add_internal_call( "SpockEngine.UIImageToggleButton::UIImageToggleButton_OnClicked",
                                UIImageToggleButton::UIImageToggleButton_OnClicked );
        mono_add_internal_call( "SpockEngine.UIImageToggleButton::UIImageToggleButton_OnChanged",
                                UIImageToggleButton::UIImageToggleButton_OnChanged );
        mono_add_internal_call( "SpockEngine.UIImageToggleButton::UIImageToggleButton_IsActive",
                                UIImageToggleButton::UIImageToggleButton_IsActive );
        mono_add_internal_call( "SpockEngine.UIImageToggleButton::UIImageToggleButton_SetActive",
                                UIImageToggleButton::UIImageToggleButton_SetActive );
        mono_add_internal_call( "SpockEngine.UIImageToggleButton::UIImageToggleButton_SetActiveImage",
                                UIImageToggleButton::UIImageToggleButton_SetActiveImage );
        mono_add_internal_call( "SpockEngine.UIImageToggleButton::UIImageToggleButton_SetInactiveImage",
                                UIImageToggleButton::UIImageToggleButton_SetInactiveImage );

        mono_add_internal_call( "SpockEngine.UIButton::UIButton_Create", UIButton::UIButton_Create );
        mono_add_internal_call( "SpockEngine.UIButton::UIButton_CreateWithText", UIButton::UIButton_CreateWithText );
        mono_add_internal_call( "SpockEngine.UIButton::UIButton_Destroy", UIButton::UIButton_Destroy );
        mono_add_internal_call( "SpockEngine.UIButton::UIButton_SetText", UIButton::UIButton_SetText );
        mono_add_internal_call( "SpockEngine.UIButton::UIButton_OnClick", UIButton::UIButton_OnClick );

        mono_add_internal_call( "SpockEngine.UITextToggleButton::UITextToggleButton_Create",
                                UITextToggleButton::UITextToggleButton_Create );
        mono_add_internal_call( "SpockEngine.UITextToggleButton::UITextToggleButton_CreateWithText",
                                UITextToggleButton::UITextToggleButton_CreateWithText );
        mono_add_internal_call( "SpockEngine.UITextToggleButton::UITextToggleButton_Destroy",
                                UITextToggleButton::UITextToggleButton_Destroy );
        mono_add_internal_call( "SpockEngine.UITextToggleButton::UITextToggleButton_OnClicked",
                                UITextToggleButton::UITextToggleButton_OnClicked );
        mono_add_internal_call( "SpockEngine.UITextToggleButton::UITextToggleButton_OnChanged",
                                UITextToggleButton::UITextToggleButton_OnChanged );

        mono_add_internal_call( "SpockEngine.UITextToggleButton::UITextToggleButton_IsActive",
                                UITextToggleButton::UITextToggleButton_IsActive );
        mono_add_internal_call( "SpockEngine.UITextToggleButton::UITextToggleButton_SetActive",
                                UITextToggleButton::UITextToggleButton_SetActive );
        mono_add_internal_call( "SpockEngine.UITextToggleButton::UITextToggleButton_SetActiveColor",
                                UITextToggleButton::UITextToggleButton_SetActiveColor );
        mono_add_internal_call( "SpockEngine.UITextToggleButton::UITextToggleButton_SetInactiveColor",
                                UITextToggleButton::UITextToggleButton_SetInactiveColor );

        mono_add_internal_call( "SpockEngine.UICheckBox::UICheckBox_Create", UICheckBox::UICheckBox_Create );
        mono_add_internal_call( "SpockEngine.UICheckBox::UICheckBox_Destroy", UICheckBox::UICheckBox_Destroy );
        mono_add_internal_call( "SpockEngine.UICheckBox::UICheckBox_OnClick", UICheckBox::UICheckBox_OnClick );
        mono_add_internal_call( "SpockEngine.UICheckBox::UICheckBox_IsChecked", UICheckBox::UICheckBox_IsChecked );
        mono_add_internal_call( "SpockEngine.UICheckBox::UICheckBox_SetIsChecked", UICheckBox::UICheckBox_SetIsChecked );

        mono_add_internal_call( "SpockEngine.UIComboBox::UIComboBox_Create", UIComboBox::UIComboBox_Create );
        mono_add_internal_call( "SpockEngine.UIComboBox::UIComboBox_CreateWithItems", UIComboBox::UIComboBox_CreateWithItems );
        mono_add_internal_call( "SpockEngine.UIComboBox::UIComboBox_Destroy", UIComboBox::UIComboBox_Destroy );
        mono_add_internal_call( "SpockEngine.UIComboBox::UIComboBox_GetCurrent", UIComboBox::UIComboBox_GetCurrent );
        mono_add_internal_call( "SpockEngine.UIComboBox::UIComboBox_SetCurrent", UIComboBox::UIComboBox_SetCurrent );
        mono_add_internal_call( "SpockEngine.UIComboBox::UIComboBox_SetItemList", UIComboBox::UIComboBox_SetItemList );
        mono_add_internal_call( "SpockEngine.UIComboBox::UIComboBox_OnChanged", UIComboBox::UIComboBox_OnChanged );

        mono_add_internal_call( "SpockEngine.UIBoxLayout::UIBoxLayout_CreateWithOrientation",
                                UIBoxLayout::UIBoxLayout_CreateWithOrientation );
        mono_add_internal_call( "SpockEngine.UIBoxLayout::UIBoxLayout_Destroy", UIBoxLayout::UIBoxLayout_Destroy );
        mono_add_internal_call( "SpockEngine.UIBoxLayout::UIBoxLayout_AddAlignedNonFixed",
                                UIBoxLayout::UIBoxLayout_AddAlignedNonFixed );
        mono_add_internal_call( "SpockEngine.UIBoxLayout::UIBoxLayout_AddNonAlignedNonFixed",
                                UIBoxLayout::UIBoxLayout_AddNonAlignedNonFixed );
        mono_add_internal_call( "SpockEngine.UIBoxLayout::UIBoxLayout_AddAlignedFixed", UIBoxLayout::UIBoxLayout_AddAlignedFixed );
        mono_add_internal_call( "SpockEngine.UIBoxLayout::UIBoxLayout_AddNonAlignedFixed",
                                UIBoxLayout::UIBoxLayout_AddNonAlignedFixed );
        mono_add_internal_call( "SpockEngine.UIBoxLayout::UIBoxLayout_SetItemSpacing", UIBoxLayout::UIBoxLayout_SetItemSpacing );
        mono_add_internal_call( "SpockEngine.UIBoxLayout::UIBoxLayout_Clear", UIBoxLayout::UIBoxLayout_Clear );

        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_Create", UIZLayout::UIZLayout_Create );
        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_Destroy", UIZLayout::UIZLayout_Destroy );
        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_AddAlignedNonFixed", UIZLayout::UIZLayout_AddAlignedNonFixed );
        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_AddNonAlignedNonFixed", UIZLayout::UIZLayout_AddNonAlignedNonFixed );
        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_AddAlignedFixed", UIZLayout::UIZLayout_AddAlignedFixed );
        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_AddNonAlignedFixed", UIZLayout::UIZLayout_AddNonAlignedFixed );

        mono_add_internal_call( "SpockEngine.UIStackLayout::UIStackLayout_Create", UIStackLayout::UIStackLayout_Create );
        mono_add_internal_call( "SpockEngine.UIStackLayout::UIStackLayout_Destroy", UIStackLayout::UIStackLayout_Destroy );
        mono_add_internal_call( "SpockEngine.UIStackLayout::UIStackLayout_Add", UIStackLayout::UIStackLayout_Add );
        mono_add_internal_call( "SpockEngine.UIStackLayout::UIStackLayout_SetCurrent", UIStackLayout::UIStackLayout_SetCurrent );

        mono_add_internal_call( "SpockEngine.UISplitter::UISplitter_Create", UISplitter::UISplitter_Create );
        mono_add_internal_call( "SpockEngine.UISplitter::UISplitter_CreateWithOrientation",
                                UISplitter::UISplitter_CreateWithOrientation );
        mono_add_internal_call( "SpockEngine.UISplitter::UISplitter_Destroy", UISplitter::UISplitter_Destroy );
        mono_add_internal_call( "SpockEngine.UISplitter::UISplitter_Add1", UISplitter::UISplitter_Add1 );
        mono_add_internal_call( "SpockEngine.UISplitter::UISplitter_Add2", UISplitter::UISplitter_Add2 );
        mono_add_internal_call( "SpockEngine.UISplitter::UISplitter_SetItemSpacing", UISplitter::UISplitter_SetItemSpacing );

        mono_add_internal_call( "SpockEngine.UITableColumn::UITableColumn_SetTooltip", sTableColumn::UITableColumn_SetTooltip );
        mono_add_internal_call( "SpockEngine.UITableColumn::UITableColumn_SetForegroundColor",
                                sTableColumn::UITableColumn_SetForegroundColor );
        mono_add_internal_call( "SpockEngine.UITableColumn::UITableColumn_SetBackgroundColor",
                                sTableColumn::UITableColumn_SetBackgroundColor );

        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_Create", sFloat64Column::UIFloat64Column_Create );
        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_CreateFull",
                                sFloat64Column::UIFloat64Column_CreateFull );
        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_Destroy", sFloat64Column::UIFloat64Column_Destroy );
        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_Clear", sFloat64Column::UIFloat64Column_Clear );
        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_SetData", sFloat64Column::UIFloat64Column_SetData );

        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_Create", sUint32Column::UIUint32Column_Create );
        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_CreateFull", sUint32Column::UIUint32Column_CreateFull );
        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_Destroy", sUint32Column::UIUint32Column_Destroy );
        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_Clear", sUint32Column::UIUint32Column_Clear );
        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_SetData", sUint32Column::UIUint32Column_SetData );

        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_Create", sStringColumn::UIStringColumn_Create );
        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_CreateFull", sStringColumn::UIStringColumn_CreateFull );
        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_Destroy", sStringColumn::UIStringColumn_Destroy );
        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_Clear", sStringColumn::UIStringColumn_Clear );
        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_SetData", sStringColumn::UIStringColumn_SetData );

        mono_add_internal_call( "SpockEngine.UITable::UITable_Create", UITable::UITable_Create );
        mono_add_internal_call( "SpockEngine.UITable::UITable_Destroy", UITable::UITable_Destroy );
        mono_add_internal_call( "SpockEngine.UITable::UITable_OnRowClicked", UITable::UITable_OnRowClicked );
        mono_add_internal_call( "SpockEngine.UITable::UITable_AddColumn", UITable::UITable_AddColumn );
        mono_add_internal_call( "SpockEngine.UITable::UITable_SetRowHeight", UITable::UITable_SetRowHeight );
        mono_add_internal_call( "SpockEngine.UITable::UITable_SetRowBackgroundColor", UITable::UITable_SetRowBackgroundColor );
        mono_add_internal_call( "SpockEngine.UITable::UITable_ClearRowBackgroundColor", UITable::UITable_ClearRowBackgroundColor );
        mono_add_internal_call( "SpockEngine.UITable::UITable_SetDisplayedRowIndices", UITable::UITable_SetDisplayedRowIndices );

        mono_add_internal_call( "SpockEngine.UIPlot::UIPlot_Create", UIPlot::UIPlot_Create );
        mono_add_internal_call( "SpockEngine.UIPlot::UIPlot_Destroy", UIPlot::UIPlot_Destroy );
        mono_add_internal_call( "SpockEngine.UIPlot::UIPlot_Clear", UIPlot::UIPlot_Clear );
        mono_add_internal_call( "SpockEngine.UIPlot::UIPlot_ConfigureLegend", UIPlot::UIPlot_ConfigureLegend );
        mono_add_internal_call( "SpockEngine.UIPlot::UIPlot_Add", UIPlot::UIPlot_Add );
        mono_add_internal_call( "SpockEngine.UIPlotAxis::UIPlot_SetAxisLimits", UIPlot::UIPlot_SetAxisLimits );
        mono_add_internal_call( "SpockEngine.UIPlotAxis::UIPlot_GetAxisTitle", UIPlot::UIPlot_GetAxisTitle );
        mono_add_internal_call( "SpockEngine.UIPlotAxis::UIPlot_SetAxisTitle", UIPlot::UIPlot_SetAxisTitle );

        mono_add_internal_call( "SpockEngine.UIPlotData::UIPlotData_SetThickness", sPlotData::UIPlotData_SetThickness );
        mono_add_internal_call( "SpockEngine.UIPlotData::UIPlotData_SetLegend", sPlotData::UIPlotData_SetLegend );
        mono_add_internal_call( "SpockEngine.UIPlotData::UIPlotData_SetColor", sPlotData::UIPlotData_SetColor );
        mono_add_internal_call( "SpockEngine.UIPlotData::UIPlotData_SetXAxis", sPlotData::UIPlotData_SetXAxis );
        mono_add_internal_call( "SpockEngine.UIPlotData::UIPlotData_SetYAxis", sPlotData::UIPlotData_SetYAxis );

        mono_add_internal_call( "SpockEngine.UIFloat64LinePlot::UIFloat64LinePlot_Create",
                                sFloat64LinePlot::UIFloat64LinePlot_Create );
        mono_add_internal_call( "SpockEngine.UIFloat64LinePlot::UIFloat64LinePlot_Destroy",
                                sFloat64LinePlot::UIFloat64LinePlot_Destroy );
        mono_add_internal_call( "SpockEngine.UIFloat64LinePlot::UIFloat64LinePlot_SetX", sFloat64LinePlot::UIFloat64LinePlot_SetX );
        mono_add_internal_call( "SpockEngine.UIFloat64LinePlot::UIFloat64LinePlot_SetY", sFloat64LinePlot::UIFloat64LinePlot_SetY );

        mono_add_internal_call( "SpockEngine.UIFloat64ScatterPlot::UIFloat64ScatterPlot_Create",
                                sFloat64ScatterPlot::UIFloat64ScatterPlot_Create );
        mono_add_internal_call( "SpockEngine.UIFloat64ScatterPlot::UIFloat64ScatterPlot_Destroy",
                                sFloat64ScatterPlot::UIFloat64ScatterPlot_Destroy );
        mono_add_internal_call( "SpockEngine.UIFloat64ScatterPlot::UIFloat64ScatterPlot_SetX",
                                sFloat64ScatterPlot::UIFloat64ScatterPlot_SetX );
        mono_add_internal_call( "SpockEngine.UIFloat64ScatterPlot::UIFloat64ScatterPlot_SetY",
                                sFloat64ScatterPlot::UIFloat64ScatterPlot_SetY );

        mono_add_internal_call( "SpockEngine.UIVLinePlot::UIVLinePlot_Create", sVLine::UIVLinePlot_Create );
        mono_add_internal_call( "SpockEngine.UIVLinePlot::UIVLinePlot_Destroy", sVLine::UIVLinePlot_Destroy );
        mono_add_internal_call( "SpockEngine.UIVLinePlot::UIVLinePlot_SetX", sVLine::UIVLinePlot_SetX );

        mono_add_internal_call( "SpockEngine.UIHLinePlot::UIHLinePlot_Create", sHLine::UIHLinePlot_Create );
        mono_add_internal_call( "SpockEngine.UIHLinePlot::UIHLinePlot_Destroy", sHLine::UIHLinePlot_Destroy );
        mono_add_internal_call( "SpockEngine.UIHLinePlot::UIHLinePlot_SetY", sHLine::UIHLinePlot_SetY );

        mono_add_internal_call( "SpockEngine.UIAxisTag::UIAxisTag_Create", sAxisTag::UIAxisTag_Create );
        mono_add_internal_call( "SpockEngine.UIAxisTag::UIAxisTag_CreateWithTextAndColor",
                                sAxisTag::UIAxisTag_CreateWithTextAndColor );
        mono_add_internal_call( "SpockEngine.UIAxisTag::UIAxisTag_Destroy", sAxisTag::UIAxisTag_Destroy );
        mono_add_internal_call( "SpockEngine.UIAxisTag::UIAxisTag_SetX", sAxisTag::UIAxisTag_SetX );
        mono_add_internal_call( "SpockEngine.UIAxisTag::UIAxisTag_SetText", sAxisTag::UIAxisTag_SetText );
        mono_add_internal_call( "SpockEngine.UIAxisTag::UIAxisTag_GetColor", sAxisTag::UIAxisTag_GetColor );
        mono_add_internal_call( "SpockEngine.UIAxisTag::UIAxisTag_SetColor", sAxisTag::UIAxisTag_SetColor );

        mono_add_internal_call( "SpockEngine.UIVRange::UIVRangePlot_Create", sVRange::UIVRangePlot_Create );
        mono_add_internal_call( "SpockEngine.UIVRange::UIVRangePlot_Destroy", sVRange::UIVRangePlot_Destroy );
        mono_add_internal_call( "SpockEngine.UIVRange::UIVRangePlot_GetMin", sVRange::UIVRangePlot_GetMin );
        mono_add_internal_call( "SpockEngine.UIVRange::UIVRangePlot_SetMin", sVRange::UIVRangePlot_SetMin );
        mono_add_internal_call( "SpockEngine.UIVRange::UIVRangePlot_GetMax", sVRange::UIVRangePlot_GetMax );
        mono_add_internal_call( "SpockEngine.UIVRange::UIVRangePlot_SetMax", sVRange::UIVRangePlot_SetMax );

        mono_add_internal_call( "SpockEngine.UIHRange::UIHRangePlot_Create", sHRange::UIHRangePlot_Create );
        mono_add_internal_call( "SpockEngine.UIHRange::UIHRangePlot_Destroy", sHRange::UIHRangePlot_Destroy );
        mono_add_internal_call( "SpockEngine.UIHRange::UIHRangePlot_GetMin", sHRange::UIHRangePlot_GetMin );
        mono_add_internal_call( "SpockEngine.UIHRange::UIHRangePlot_SetMin", sHRange::UIHRangePlot_SetMin );
        mono_add_internal_call( "SpockEngine.UIHRange::UIHRangePlot_GetMax", sHRange::UIHRangePlot_GetMax );
        mono_add_internal_call( "SpockEngine.UIHRange::UIHRangePlot_SetMax", sHRange::UIHRangePlot_SetMax );

        mono_add_internal_call( "SpockEngine.UITextOverlay::UITextOverlay_Create", UITextOverlay::UITextOverlay_Create );
        mono_add_internal_call( "SpockEngine.UITextOverlay::UITextOverlay_Destroy", UITextOverlay::UITextOverlay_Destroy );
        mono_add_internal_call( "SpockEngine.UITextOverlay::UITextOverlay_AddText", UITextOverlay::UITextOverlay_AddText );
        mono_add_internal_call( "SpockEngine.UITextOverlay::UITextOverlay_Clear", UITextOverlay::UITextOverlay_Clear );

        mono_add_internal_call( "SpockEngine.UIWorkspace::UIWorkspace_Create", UIWorkspace::UIWorkspace_Create );
        mono_add_internal_call( "SpockEngine.UIWorkspace::UIWorkspace_Destroy", UIWorkspace::UIWorkspace_Destroy );
        mono_add_internal_call( "SpockEngine.UIWorkspace::UIWorkspace_Add", UIWorkspace::UIWorkspace_Add );
        mono_add_internal_call( "SpockEngine.UIWorkspace::UIWorkspace_RegisterCloseDocumentDelegate",
                                UIWorkspace::UIWorkspace_RegisterCloseDocumentDelegate );

        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_Create",
                                UIWorkspaceDocument::UIWorkspaceDocument_Create );
        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_Destroy",
                                UIWorkspaceDocument::UIWorkspaceDocument_Destroy );
        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_SetContent",
                                UIWorkspaceDocument::UIWorkspaceDocument_SetContent );
        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_Update",
                                UIWorkspaceDocument::UIWorkspaceDocument_Update );
        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_SetName",
                                UIWorkspaceDocument::UIWorkspaceDocument_SetName );
        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_IsDirty",
                                UIWorkspaceDocument::UIWorkspaceDocument_IsDirty );
        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_MarkAsDirty",
                                UIWorkspaceDocument::UIWorkspaceDocument_MarkAsDirty );
        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_Open",
                                UIWorkspaceDocument::UIWorkspaceDocument_Open );
        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_RequestClose",
                                UIWorkspaceDocument::UIWorkspaceDocument_RequestClose );
        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_ForceClose",
                                UIWorkspaceDocument::UIWorkspaceDocument_ForceClose );
        mono_add_internal_call( "SpockEngine.UIWorkspaceDocument::UIWorkspaceDocument_RegisterSaveDelegate",
                                UIWorkspaceDocument::UIWorkspaceDocument_RegisterSaveDelegate );

        mono_add_internal_call( "SpockEngine.UIMenuItem::UIMenuItem_Create", UIMenuItem::UIMenuItem_Create );
        mono_add_internal_call( "SpockEngine.UIMenuItem::UIMenuItem_CreateWithText", UIMenuItem::UIMenuItem_CreateWithText );
        mono_add_internal_call( "SpockEngine.UIMenuItem::UIMenuItem_CreateWithTextAndShortcut",
                                UIMenuItem::UIMenuItem_CreateWithTextAndShortcut );
        mono_add_internal_call( "SpockEngine.UIMenuItem::UIMenuItem_Destroy", UIMenuItem::UIMenuItem_Destroy );
        mono_add_internal_call( "SpockEngine.UIMenuItem::UIMenuItem_SetText", UIMenuItem::UIMenuItem_SetText );
        mono_add_internal_call( "SpockEngine.UIMenuItem::UIMenuItem_SetShortcut", UIMenuItem::UIMenuItem_SetShortcut );
        mono_add_internal_call( "SpockEngine.UIMenuItem::UIMenuItem_SetTextColor", UIMenuItem::UIMenuItem_SetTextColor );
        mono_add_internal_call( "SpockEngine.UIMenuItem::UIMenuItem_OnTrigger", UIMenuItem::UIMenuItem_OnTrigger );

        mono_add_internal_call( "SpockEngine.UIMenuSeparator::UIMenuSeparator_Create", UIMenuSeparator::UIMenuSeparator_Create );
        mono_add_internal_call( "SpockEngine.UIMenuSeparator::UIMenuSeparator_Destroy", UIMenuSeparator::UIMenuSeparator_Destroy );

        mono_add_internal_call( "SpockEngine.UIMenu::UIMenu_Create", UIMenu::UIMenu_Create );
        mono_add_internal_call( "SpockEngine.UIMenu::UIMenu_CreateWithText", UIMenu::UIMenu_CreateWithText );
        mono_add_internal_call( "SpockEngine.UIMenu::UIMenu_Destroy", UIMenu::UIMenu_Destroy );
        mono_add_internal_call( "SpockEngine.UIMenu::UIMenu_AddAction", UIMenu::UIMenu_AddAction );
        mono_add_internal_call( "SpockEngine.UIMenu::UIMenu_AddMenu", UIMenu::UIMenu_AddMenu );
        mono_add_internal_call( "SpockEngine.UIMenu::UIMenu_AddSeparator", UIMenu::UIMenu_AddSeparator );
        mono_add_internal_call( "SpockEngine.UIMenu::UIMenu_Update", UIMenu::UIMenu_Update );

        mono_add_internal_call( "SpockEngine.UIPropertyValue::UIPropertyValue_Create", UIPropertyValue::UIPropertyValue_Create );
        mono_add_internal_call( "SpockEngine.UIPropertyValue::UIPropertyValue_CreateWithText",
                                UIPropertyValue::UIPropertyValue_CreateWithText );
        mono_add_internal_call( "SpockEngine.UIPropertyValue::UIPropertyValue_CreateWithTextAndOrientation",
                                UIPropertyValue::UIPropertyValue_CreateWithTextAndOrientation );
        mono_add_internal_call( "SpockEngine.UIPropertyValue::UIPropertyValue_Destroy", UIPropertyValue::UIPropertyValue_Destroy );
        mono_add_internal_call( "SpockEngine.UIPropertyValue::UIPropertyValue_SetValue", UIPropertyValue::UIPropertyValue_SetValue );
        mono_add_internal_call( "SpockEngine.UIPropertyValue::UIPropertyValue_SetValueFont",
                                UIPropertyValue::UIPropertyValue_SetValueFont );
        mono_add_internal_call( "SpockEngine.UIPropertyValue::UIPropertyValue_SetNameFont",
                                UIPropertyValue::UIPropertyValue_SetNameFont );

        mono_add_internal_call( "SpockEngine.UITextInput::UITextInput_Create", UITextInput::UITextInput_Create );
        mono_add_internal_call( "SpockEngine.UITextInput::UITextInput_CreateWithText", UITextInput::UITextInput_CreateWithText );
        mono_add_internal_call( "SpockEngine.UITextInput::UITextInput_Destroy", UITextInput::UITextInput_Destroy );
        mono_add_internal_call( "SpockEngine.UITextInput::UITextInput_GetText", UITextInput::UITextInput_GetText );
        mono_add_internal_call( "SpockEngine.UITextInput::UITextInput_SetHintText", UITextInput::UITextInput_SetHintText );
        mono_add_internal_call( "SpockEngine.UITextInput::UITextInput_SetTextColor", UITextInput::UITextInput_SetTextColor );
        mono_add_internal_call( "SpockEngine.UITextInput::UITextInput_SetBufferSize", UITextInput::UITextInput_SetBufferSize );
        mono_add_internal_call( "SpockEngine.UITextInput::UITextInput_OnTextChanged", UITextInput::UITextInput_OnTextChanged );

        mono_add_internal_call( "SpockEngine.UIProgressBar::UIProgressBar_Create", UIProgressBar::UIProgressBar_Create );
        mono_add_internal_call( "SpockEngine.UIProgressBar::UIProgressBar_Destroy", UIProgressBar::UIProgressBar_Destroy );
        mono_add_internal_call( "SpockEngine.UIProgressBar::UIProgressBar_SetProgressValue",
                                UIProgressBar::UIProgressBar_SetProgressValue );
        mono_add_internal_call( "SpockEngine.UIProgressBar::UIProgressBar_SetProgressColor",
                                UIProgressBar::UIProgressBar_SetProgressColor );
        mono_add_internal_call( "SpockEngine.UIProgressBar::UIProgressBar_SetText", UIProgressBar::UIProgressBar_SetText );
        mono_add_internal_call( "SpockEngine.UIProgressBar::UIProgressBar_SetTextColor", UIProgressBar::UIProgressBar_SetTextColor );
        mono_add_internal_call( "SpockEngine.UIProgressBar::UIProgressBar_SetThickness", UIProgressBar::UIProgressBar_SetThickness );
    }

} // namespace SE::Core