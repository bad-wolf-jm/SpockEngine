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
#include "UI/Components/DropdownButton.h"
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
#include "UI/Components/TreeView.h"
#include "UI/Components/Workspace.h"
#include "UI/Components/ColorButton.h"

#include "UI/Widgets/FileTree.h"

#include "UI/UI.h"

#include "UI/Layouts/Container.h"
#include "UI/Layouts/Splitter.h"
#include "UI/Layouts/StackLayout.h"
#include "UI/Layouts/ZLayout.h"

#include "UI/Dialog.h"
#include "UI/Form.h"
#include "UI/Layouts/BoxLayout.h"

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
        mono_domain_set( sRuntimeData->mRootDomain, true );
        // mono_domain_unload( sRuntimeData->mAppDomain );
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

#define SE_ADD_INTERNAL_CALL( Name ) mono_add_internal_call( "SpockEngine.CppCall::" #Name, Name )

    void DotNetRuntime::RegisterInternalCppFunctions()
    {
        ICall( "CppCall::OpenFile", OpenFile );
        ICall( "UIColor::GetStyleColor", SE::Core::UI::GetStyleColor );

        SE_ADD_INTERNAL_CALL( Entity_Create );
        SE_ADD_INTERNAL_CALL( Entity_IsValid );
        SE_ADD_INTERNAL_CALL( Entity_Has );
        SE_ADD_INTERNAL_CALL( Entity_Get );
        SE_ADD_INTERNAL_CALL( Entity_Add );
        SE_ADD_INTERNAL_CALL( Entity_Replace );

        SE_ADD_INTERNAL_CALL( OpNode_NewTensorShape );
        SE_ADD_INTERNAL_CALL( OpNode_DestroyTensorShape );
        SE_ADD_INTERNAL_CALL( OpNode_CountLayers );
        SE_ADD_INTERNAL_CALL( OpNode_GetDimension );
        SE_ADD_INTERNAL_CALL( OpNode_Trim );
        SE_ADD_INTERNAL_CALL( OpNode_Flatten );
        SE_ADD_INTERNAL_CALL( OpNode_InsertDimension );

        SE_ADD_INTERNAL_CALL( OpNode_NewScope );
        SE_ADD_INTERNAL_CALL( OpNode_DestroyScope );
        SE_ADD_INTERNAL_CALL( OpNode_CreateMultiTensor_Constant_Initializer );
        SE_ADD_INTERNAL_CALL( OpNode_CreateMultiTensor_Vector_Initializer );
        SE_ADD_INTERNAL_CALL( OpNode_CreateMultiTensor_Data_Initializer );
        SE_ADD_INTERNAL_CALL( OpNode_CreateMultiTensor_Random_Uniform_Initializer );
        SE_ADD_INTERNAL_CALL( OpNode_CreateMultiTensor_Random_Normal_Initializer );
        SE_ADD_INTERNAL_CALL( OpNode_CreateVector );
        SE_ADD_INTERNAL_CALL( OpNode_CreateScalarVector );
        SE_ADD_INTERNAL_CALL( OpNode_CreateScalarValue );

        SE_ADD_INTERNAL_CALL( OpNode_Add );
        SE_ADD_INTERNAL_CALL( OpNode_Subtract );
        SE_ADD_INTERNAL_CALL( OpNode_Divide );
        SE_ADD_INTERNAL_CALL( OpNode_Multiply );
        SE_ADD_INTERNAL_CALL( OpNode_And );
        SE_ADD_INTERNAL_CALL( OpNode_Or );
        SE_ADD_INTERNAL_CALL( OpNode_Not );
        SE_ADD_INTERNAL_CALL( OpNode_BitwiseAnd );
        SE_ADD_INTERNAL_CALL( OpNode_BitwiseOr );
        SE_ADD_INTERNAL_CALL( OpNode_BitwiseNot );
        SE_ADD_INTERNAL_CALL( OpNode_InInterval );
        SE_ADD_INTERNAL_CALL( OpNode_Equal );
        SE_ADD_INTERNAL_CALL( OpNode_LessThan );
        SE_ADD_INTERNAL_CALL( OpNode_LessThanOrEqual );
        SE_ADD_INTERNAL_CALL( OpNode_GreaterThan );
        SE_ADD_INTERNAL_CALL( OpNode_GreaterThanOrEqual );
        SE_ADD_INTERNAL_CALL( OpNode_Where );
        SE_ADD_INTERNAL_CALL( OpNode_Mix );
        SE_ADD_INTERNAL_CALL( OpNode_AffineTransform );
        SE_ADD_INTERNAL_CALL( OpNode_ARange );
        SE_ADD_INTERNAL_CALL( OpNode_LinearSpace );
        SE_ADD_INTERNAL_CALL( OpNode_Repeat );
        SE_ADD_INTERNAL_CALL( OpNode_Tile );
        SE_ADD_INTERNAL_CALL( OpNode_Sample2D );
        SE_ADD_INTERNAL_CALL( OpNode_Collapse );
        SE_ADD_INTERNAL_CALL( OpNode_Expand );
        SE_ADD_INTERNAL_CALL( OpNode_Reshape );
        SE_ADD_INTERNAL_CALL( OpNode_Relayout );
        SE_ADD_INTERNAL_CALL( OpNode_FlattenNode );
        SE_ADD_INTERNAL_CALL( OpNode_Slice );
        SE_ADD_INTERNAL_CALL( OpNode_Summation );
        SE_ADD_INTERNAL_CALL( OpNode_CountTrue );
        SE_ADD_INTERNAL_CALL( OpNode_CountNonZero );
        SE_ADD_INTERNAL_CALL( OpNode_CountZero );
        SE_ADD_INTERNAL_CALL( OpNode_Floor );
        SE_ADD_INTERNAL_CALL( OpNode_Ceil );
        SE_ADD_INTERNAL_CALL( OpNode_Abs );
        SE_ADD_INTERNAL_CALL( OpNode_Sqrt );
        SE_ADD_INTERNAL_CALL( OpNode_Round );
        SE_ADD_INTERNAL_CALL( OpNode_Diff );
        SE_ADD_INTERNAL_CALL( OpNode_Shift );
        SE_ADD_INTERNAL_CALL( OpNode_Conv1D );
        SE_ADD_INTERNAL_CALL( OpNode_HCat );

        ICall( "UIComponent::UIComponent_SetIsVisible", UIComponent::UIComponent_SetIsVisible );
        ICall( "UIComponent::UIComponent_SetIsEnabled", UIComponent::UIComponent_SetIsEnabled );
        ICall( "UIComponent::UIComponent_SetAllowDragDrop", UIComponent::UIComponent_SetAllowDragDrop );
        ICall( "UIComponent::UIComponent_SetPaddingAll", UIComponent::UIComponent_SetPaddingAll );
        ICall( "UIComponent::UIComponent_SetPaddingPairs", UIComponent::UIComponent_SetPaddingPairs );
        ICall( "UIComponent::UIComponent_SetPaddingIndividual", UIComponent::UIComponent_SetPaddingIndividual );
        ICall( "UIComponent::UIComponent_SetAlignment", UIComponent::UIComponent_SetAlignment );
        ICall( "UIComponent::UIComponent_SetHorizontalAlignment", UIComponent::UIComponent_SetHorizontalAlignment );
        ICall( "UIComponent::UIComponent_SetVerticalAlignment", UIComponent::UIComponent_SetVerticalAlignment );
        ICall( "UIComponent::UIComponent_SetBackgroundColor", UIComponent::UIComponent_SetBackgroundColor );
        ICall( "UIComponent::UIComponent_SetFont", UIComponent::UIComponent_SetFont );
        ICall( "UIComponent::UIComponent_SetTooltip", UIComponent::UIComponent_SetTooltip );

        ICall( "UIForm::UIForm_Create", UIForm::UIForm_Create );
        ICall( "UIForm::UIForm_Destroy", UIForm::UIForm_Destroy );
        ICall( "UIForm::UIForm_SetTitle", UIForm::UIForm_SetTitle );
        ICall( "UIForm::UIForm_SetContent", UIForm::UIForm_SetContent );
        ICall( "UIForm::UIForm_Update", UIForm::UIForm_Update );
        ICall( "UIForm::UIForm_SetSize", UIForm::UIForm_SetSize );

        ICall( "UIDialog::UIDialog_Create", UIDialog::UIDialog_Create );
        ICall( "UIDialog::UIDialog_CreateWithTitleAndSize", UIDialog::UIDialog_CreateWithTitleAndSize );
        ICall( "UIDialog::UIDialog_Destroy", UIDialog::UIDialog_Destroy );
        ICall( "UIDialog::UIDialog_SetTitle", UIDialog::UIDialog_SetTitle );
        ICall( "UIDialog::UIDialog_SetSize", UIDialog::UIDialog_SetSize );
        ICall( "UIDialog::UIDialog_SetContent", UIDialog::UIDialog_SetContent );
        ICall( "UIDialog::UIDialog_Open", UIDialog::UIDialog_Open );
        ICall( "UIDialog::UIDialog_Close", UIDialog::UIDialog_Close );
        ICall( "UIDialog::UIDialog_Update", UIDialog::UIDialog_Update );

        ICall( "UILabel::UILabel_Create", UILabel::UILabel_Create );
        ICall( "UILabel::UILabel_CreateWithText", UILabel::UILabel_CreateWithText );
        ICall( "UILabel::UILabel_Destroy", UILabel::UILabel_Destroy );
        ICall( "UILabel::UILabel_SetText", UILabel::UILabel_SetText );
        ICall( "UILabel::UILabel_SetTextColor", UILabel::UILabel_SetTextColor );

        ICall( "UIBaseImage::UIBaseImage_Create", UIBaseImage::UIBaseImage_Create );
        ICall( "UIBaseImage::UIBaseImage_CreateWithPath", UIBaseImage::UIBaseImage_CreateWithPath );
        ICall( "UIBaseImage::UIBaseImage_Destroy", UIBaseImage::UIBaseImage_Destroy );
        ICall( "UIBaseImage::UIBaseImage_SetImage", UIBaseImage::UIBaseImage_SetImage );
        ICall( "UIBaseImage::UIBaseImage_SetSize", UIBaseImage::UIBaseImage_SetSize );
        ICall( "UIBaseImage::UIBaseImage_GetSize", UIBaseImage::UIBaseImage_GetSize );
        ICall( "UIBaseImage::UIBaseImage_SetTopLeft", UIBaseImage::UIBaseImage_SetTopLeft );
        ICall( "UIBaseImage::UIBaseImage_GetTopLeft", UIBaseImage::UIBaseImage_GetTopLeft );
        ICall( "UIBaseImage::UIBaseImage_SetBottomRight", UIBaseImage::UIBaseImage_SetBottomRight );
        ICall( "UIBaseImage::UIBaseImage_GetBottomRight", UIBaseImage::UIBaseImage_GetBottomRight );
        ICall( "UIBaseImage::UIBaseImage_SetTintColor", UIBaseImage::UIBaseImage_SetTintColor );
        ICall( "UIBaseImage::UIBaseImage_GetTintColor", UIBaseImage::UIBaseImage_GetTintColor );

        ICall( "UIImage::UIImage_Create", UIImage::UIImage_Create );
        ICall( "UIImage::UIImage_CreateWithPath", UIImage::UIImage_CreateWithPath );
        ICall( "UIImage::UIImage_Destroy", UIImage::UIImage_Destroy );

        ICall( "UIImageButton::UIImageButton_Create", UIImageButton::UIImageButton_Create );
        ICall( "UIImageButton::UIImageButton_CreateWithPath", UIImageButton::UIImageButton_CreateWithPath );
        ICall( "UIImageButton::UIImageButton_Destroy", UIImageButton::UIImageButton_Destroy );
        ICall( "UIImageButton::UIImageButton_OnClick", UIImageButton::UIImageButton_OnClick );

        ICall( "UIImageToggleButton::UIImageToggleButton_Create", UIImageToggleButton::UIImageToggleButton_Create );
        ICall( "UIImageToggleButton::UIImageToggleButton_Destroy", UIImageToggleButton::UIImageToggleButton_Destroy );
        ICall( "UIImageToggleButton::UIImageToggleButton_OnClicked", UIImageToggleButton::UIImageToggleButton_OnClicked );
        ICall( "UIImageToggleButton::UIImageToggleButton_OnChanged", UIImageToggleButton::UIImageToggleButton_OnChanged );
        ICall( "UIImageToggleButton::UIImageToggleButton_IsActive", UIImageToggleButton::UIImageToggleButton_IsActive );
        ICall( "UIImageToggleButton::UIImageToggleButton_SetActive", UIImageToggleButton::UIImageToggleButton_SetActive );
        ICall( "UIImageToggleButton::UIImageToggleButton_SetActiveImage", UIImageToggleButton::UIImageToggleButton_SetActiveImage );
        ICall( "UIImageToggleButton::UIImageToggleButton_SetInactiveImage",
               UIImageToggleButton::UIImageToggleButton_SetInactiveImage );

        ICall( "UIButton::UIButton_Create", UIButton::UIButton_Create );
        ICall( "UIButton::UIButton_CreateWithText", UIButton::UIButton_CreateWithText );
        ICall( "UIButton::UIButton_Destroy", UIButton::UIButton_Destroy );
        ICall( "UIButton::UIButton_SetText", UIButton::UIButton_SetText );
        ICall( "UIButton::UIButton_OnClick", UIButton::UIButton_OnClick );

        ICall( "UITextToggleButton::UITextToggleButton_Create", UITextToggleButton::UITextToggleButton_Create );
        ICall( "UITextToggleButton::UITextToggleButton_CreateWithText", UITextToggleButton::UITextToggleButton_CreateWithText );
        ICall( "UITextToggleButton::UITextToggleButton_Destroy", UITextToggleButton::UITextToggleButton_Destroy );
        ICall( "UITextToggleButton::UITextToggleButton_OnClicked", UITextToggleButton::UITextToggleButton_OnClicked );
        ICall( "UITextToggleButton::UITextToggleButton_OnChanged", UITextToggleButton::UITextToggleButton_OnChanged );

        ICall( "UITextToggleButton::UITextToggleButton_IsActive", UITextToggleButton::UITextToggleButton_IsActive );
        ICall( "UITextToggleButton::UITextToggleButton_SetActive", UITextToggleButton::UITextToggleButton_SetActive );
        ICall( "UITextToggleButton::UITextToggleButton_SetActiveColor", UITextToggleButton::UITextToggleButton_SetActiveColor );
        ICall( "UITextToggleButton::UITextToggleButton_SetInactiveColor", UITextToggleButton::UITextToggleButton_SetInactiveColor );

        ICall( "UICheckBox::UICheckBox_Create", UICheckBox::UICheckBox_Create );
        ICall( "UICheckBox::UICheckBox_Destroy", UICheckBox::UICheckBox_Destroy );
        ICall( "UICheckBox::UICheckBox_OnClick", UICheckBox::UICheckBox_OnClick );
        ICall( "UICheckBox::UICheckBox_IsChecked", UICheckBox::UICheckBox_IsChecked );
        ICall( "UICheckBox::UICheckBox_SetIsChecked", UICheckBox::UICheckBox_SetIsChecked );

        ICall( "UIComboBox::UIComboBox_Create", UIComboBox::UIComboBox_Create );
        ICall( "UIComboBox::UIComboBox_CreateWithItems", UIComboBox::UIComboBox_CreateWithItems );
        ICall( "UIComboBox::UIComboBox_Destroy", UIComboBox::UIComboBox_Destroy );
        ICall( "UIComboBox::UIComboBox_GetCurrent", UIComboBox::UIComboBox_GetCurrent );
        ICall( "UIComboBox::UIComboBox_SetCurrent", UIComboBox::UIComboBox_SetCurrent );
        ICall( "UIComboBox::UIComboBox_SetItemList", UIComboBox::UIComboBox_SetItemList );
        ICall( "UIComboBox::UIComboBox_OnChanged", UIComboBox::UIComboBox_OnChanged );

        ICall( "UIBoxLayout::UIBoxLayout_CreateWithOrientation", UIBoxLayout::UIBoxLayout_CreateWithOrientation );
        ICall( "UIBoxLayout::UIBoxLayout_Destroy", UIBoxLayout::UIBoxLayout_Destroy );
        ICall( "UIBoxLayout::UIBoxLayout_AddAlignedNonFixed", UIBoxLayout::UIBoxLayout_AddAlignedNonFixed );
        ICall( "UIBoxLayout::UIBoxLayout_AddNonAlignedNonFixed", UIBoxLayout::UIBoxLayout_AddNonAlignedNonFixed );
        ICall( "UIBoxLayout::UIBoxLayout_AddAlignedFixed", UIBoxLayout::UIBoxLayout_AddAlignedFixed );
        ICall( "UIBoxLayout::UIBoxLayout_AddNonAlignedFixed", UIBoxLayout::UIBoxLayout_AddNonAlignedFixed );
        ICall( "UIBoxLayout::UIBoxLayout_AddSeparator", UIBoxLayout::UIBoxLayout_AddSeparator );
        ICall( "UIBoxLayout::UIBoxLayout_SetItemSpacing", UIBoxLayout::UIBoxLayout_SetItemSpacing );
        ICall( "UIBoxLayout::UIBoxLayout_Clear", UIBoxLayout::UIBoxLayout_Clear );

        ICall( "UIZLayout::UIZLayout_Create", UIZLayout::UIZLayout_Create );
        ICall( "UIZLayout::UIZLayout_Destroy", UIZLayout::UIZLayout_Destroy );
        ICall( "UIZLayout::UIZLayout_AddAlignedNonFixed", UIZLayout::UIZLayout_AddAlignedNonFixed );
        ICall( "UIZLayout::UIZLayout_AddNonAlignedNonFixed", UIZLayout::UIZLayout_AddNonAlignedNonFixed );
        ICall( "UIZLayout::UIZLayout_AddAlignedFixed", UIZLayout::UIZLayout_AddAlignedFixed );
        ICall( "UIZLayout::UIZLayout_AddNonAlignedFixed", UIZLayout::UIZLayout_AddNonAlignedFixed );

        ICall( "UIStackLayout::UIStackLayout_Create", UIStackLayout::UIStackLayout_Create );
        ICall( "UIStackLayout::UIStackLayout_Destroy", UIStackLayout::UIStackLayout_Destroy );
        ICall( "UIStackLayout::UIStackLayout_Add", UIStackLayout::UIStackLayout_Add );
        ICall( "UIStackLayout::UIStackLayout_SetCurrent", UIStackLayout::UIStackLayout_SetCurrent );

        ICall( "UISplitter::UISplitter_Create", UISplitter::UISplitter_Create );
        ICall( "UISplitter::UISplitter_CreateWithOrientation", UISplitter::UISplitter_CreateWithOrientation );
        ICall( "UISplitter::UISplitter_Destroy", UISplitter::UISplitter_Destroy );
        ICall( "UISplitter::UISplitter_Add1", UISplitter::UISplitter_Add1 );
        ICall( "UISplitter::UISplitter_Add2", UISplitter::UISplitter_Add2 );
        ICall( "UISplitter::UISplitter_SetItemSpacing", UISplitter::UISplitter_SetItemSpacing );

        ICall( "UITableColumn::UITableColumn_SetTooltip", sTableColumn::UITableColumn_SetTooltip );
        ICall( "UITableColumn::UITableColumn_SetForegroundColor", sTableColumn::UITableColumn_SetForegroundColor );
        ICall( "UITableColumn::UITableColumn_SetBackgroundColor", sTableColumn::UITableColumn_SetBackgroundColor );

        ICall( "UIFloat64Column::UIFloat64Column_Create", sFloat64Column::UIFloat64Column_Create );
        ICall( "UIFloat64Column::UIFloat64Column_CreateFull", sFloat64Column::UIFloat64Column_CreateFull );
        ICall( "UIFloat64Column::UIFloat64Column_Destroy", sFloat64Column::UIFloat64Column_Destroy );
        ICall( "UIFloat64Column::UIFloat64Column_Clear", sFloat64Column::UIFloat64Column_Clear );
        ICall( "UIFloat64Column::UIFloat64Column_SetData", sFloat64Column::UIFloat64Column_SetData );

        ICall( "UIUint32Column::UIUint32Column_Create", sUint32Column::UIUint32Column_Create );
        ICall( "UIUint32Column::UIUint32Column_CreateFull", sUint32Column::UIUint32Column_CreateFull );
        ICall( "UIUint32Column::UIUint32Column_Destroy", sUint32Column::UIUint32Column_Destroy );
        ICall( "UIUint32Column::UIUint32Column_Clear", sUint32Column::UIUint32Column_Clear );
        ICall( "UIUint32Column::UIUint32Column_SetData", sUint32Column::UIUint32Column_SetData );

        ICall( "UIStringColumn::UIStringColumn_Create", sStringColumn::UIStringColumn_Create );
        ICall( "UIStringColumn::UIStringColumn_CreateFull", sStringColumn::UIStringColumn_CreateFull );
        ICall( "UIStringColumn::UIStringColumn_Destroy", sStringColumn::UIStringColumn_Destroy );
        ICall( "UIStringColumn::UIStringColumn_Clear", sStringColumn::UIStringColumn_Clear );
        ICall( "UIStringColumn::UIStringColumn_SetData", sStringColumn::UIStringColumn_SetData );

        ICall( "UITable::UITable_Create", UITable::UITable_Create );
        ICall( "UITable::UITable_Destroy", UITable::UITable_Destroy );
        ICall( "UITable::UITable_OnRowClicked", UITable::UITable_OnRowClicked );
        ICall( "UITable::UITable_AddColumn", UITable::UITable_AddColumn );
        ICall( "UITable::UITable_SetRowHeight", UITable::UITable_SetRowHeight );
        ICall( "UITable::UITable_SetRowBackgroundColor", UITable::UITable_SetRowBackgroundColor );
        ICall( "UITable::UITable_ClearRowBackgroundColor", UITable::UITable_ClearRowBackgroundColor );
        ICall( "UITable::UITable_SetDisplayedRowIndices", UITable::UITable_SetDisplayedRowIndices );

        ICall( "UIPlot::UIPlot_Create", UIPlot::UIPlot_Create );
        ICall( "UIPlot::UIPlot_Destroy", UIPlot::UIPlot_Destroy );
        ICall( "UIPlot::UIPlot_Clear", UIPlot::UIPlot_Clear );
        ICall( "UIPlot::UIPlot_ConfigureLegend", UIPlot::UIPlot_ConfigureLegend );
        ICall( "UIPlot::UIPlot_Add", UIPlot::UIPlot_Add );
        ICall( "UIPlotAxis::UIPlot_SetAxisLimits", UIPlot::UIPlot_SetAxisLimits );
        ICall( "UIPlotAxis::UIPlot_GetAxisTitle", UIPlot::UIPlot_GetAxisTitle );
        ICall( "UIPlotAxis::UIPlot_SetAxisTitle", UIPlot::UIPlot_SetAxisTitle );

        ICall( "UIPlotData::UIPlotData_SetThickness", sPlotData::UIPlotData_SetThickness );
        ICall( "UIPlotData::UIPlotData_SetLegend", sPlotData::UIPlotData_SetLegend );
        ICall( "UIPlotData::UIPlotData_SetColor", sPlotData::UIPlotData_SetColor );
        ICall( "UIPlotData::UIPlotData_SetXAxis", sPlotData::UIPlotData_SetXAxis );
        ICall( "UIPlotData::UIPlotData_SetYAxis", sPlotData::UIPlotData_SetYAxis );

        ICall( "UIFloat64LinePlot::UIFloat64LinePlot_Create", sFloat64LinePlot::UIFloat64LinePlot_Create );
        ICall( "UIFloat64LinePlot::UIFloat64LinePlot_Destroy", sFloat64LinePlot::UIFloat64LinePlot_Destroy );
        ICall( "UIFloat64LinePlot::UIFloat64LinePlot_SetX", sFloat64LinePlot::UIFloat64LinePlot_SetX );
        ICall( "UIFloat64LinePlot::UIFloat64LinePlot_SetY", sFloat64LinePlot::UIFloat64LinePlot_SetY );

        ICall( "UIFloat64ScatterPlot::UIFloat64ScatterPlot_Create", sFloat64ScatterPlot::UIFloat64ScatterPlot_Create );
        ICall( "UIFloat64ScatterPlot::UIFloat64ScatterPlot_Destroy", sFloat64ScatterPlot::UIFloat64ScatterPlot_Destroy );
        ICall( "UIFloat64ScatterPlot::UIFloat64ScatterPlot_SetX", sFloat64ScatterPlot::UIFloat64ScatterPlot_SetX );
        ICall( "UIFloat64ScatterPlot::UIFloat64ScatterPlot_SetY", sFloat64ScatterPlot::UIFloat64ScatterPlot_SetY );

        ICall( "UIVLinePlot::UIVLinePlot_Create", sVLine::UIVLinePlot_Create );
        ICall( "UIVLinePlot::UIVLinePlot_Destroy", sVLine::UIVLinePlot_Destroy );
        ICall( "UIVLinePlot::UIVLinePlot_SetX", sVLine::UIVLinePlot_SetX );

        ICall( "UIHLinePlot::UIHLinePlot_Create", sHLine::UIHLinePlot_Create );
        ICall( "UIHLinePlot::UIHLinePlot_Destroy", sHLine::UIHLinePlot_Destroy );
        ICall( "UIHLinePlot::UIHLinePlot_SetY", sHLine::UIHLinePlot_SetY );

        ICall( "UIAxisTag::UIAxisTag_Create", sAxisTag::UIAxisTag_Create );
        ICall( "UIAxisTag::UIAxisTag_CreateWithTextAndColor", sAxisTag::UIAxisTag_CreateWithTextAndColor );
        ICall( "UIAxisTag::UIAxisTag_Destroy", sAxisTag::UIAxisTag_Destroy );
        ICall( "UIAxisTag::UIAxisTag_SetX", sAxisTag::UIAxisTag_SetX );
        ICall( "UIAxisTag::UIAxisTag_SetText", sAxisTag::UIAxisTag_SetText );
        ICall( "UIAxisTag::UIAxisTag_GetColor", sAxisTag::UIAxisTag_GetColor );
        ICall( "UIAxisTag::UIAxisTag_SetColor", sAxisTag::UIAxisTag_SetColor );

        ICall( "UIVRange::UIVRangePlot_Create", sVRange::UIVRangePlot_Create );
        ICall( "UIVRange::UIVRangePlot_Destroy", sVRange::UIVRangePlot_Destroy );
        ICall( "UIVRange::UIVRangePlot_GetMin", sVRange::UIVRangePlot_GetMin );
        ICall( "UIVRange::UIVRangePlot_SetMin", sVRange::UIVRangePlot_SetMin );
        ICall( "UIVRange::UIVRangePlot_GetMax", sVRange::UIVRangePlot_GetMax );
        ICall( "UIVRange::UIVRangePlot_SetMax", sVRange::UIVRangePlot_SetMax );

        ICall( "UIHRange::UIHRangePlot_Create", sHRange::UIHRangePlot_Create );
        ICall( "UIHRange::UIHRangePlot_Destroy", sHRange::UIHRangePlot_Destroy );
        ICall( "UIHRange::UIHRangePlot_GetMin", sHRange::UIHRangePlot_GetMin );
        ICall( "UIHRange::UIHRangePlot_SetMin", sHRange::UIHRangePlot_SetMin );
        ICall( "UIHRange::UIHRangePlot_GetMax", sHRange::UIHRangePlot_GetMax );
        ICall( "UIHRange::UIHRangePlot_SetMax", sHRange::UIHRangePlot_SetMax );

        ICall( "UITextOverlay::UITextOverlay_Create", UITextOverlay::UITextOverlay_Create );
        ICall( "UITextOverlay::UITextOverlay_Destroy", UITextOverlay::UITextOverlay_Destroy );
        ICall( "UITextOverlay::UITextOverlay_AddText", UITextOverlay::UITextOverlay_AddText );
        ICall( "UITextOverlay::UITextOverlay_Clear", UITextOverlay::UITextOverlay_Clear );

        ICall( "UIWorkspace::UIWorkspace_Create", UIWorkspace::UIWorkspace_Create );
        ICall( "UIWorkspace::UIWorkspace_Destroy", UIWorkspace::UIWorkspace_Destroy );
        ICall( "UIWorkspace::UIWorkspace_Add", UIWorkspace::UIWorkspace_Add );
        ICall( "UIWorkspace::UIWorkspace_RegisterCloseDocumentDelegate", UIWorkspace::UIWorkspace_RegisterCloseDocumentDelegate );

        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_Create", UIWorkspaceDocument::UIWorkspaceDocument_Create );
        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_Destroy", UIWorkspaceDocument::UIWorkspaceDocument_Destroy );
        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_SetContent", UIWorkspaceDocument::UIWorkspaceDocument_SetContent );
        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_Update", UIWorkspaceDocument::UIWorkspaceDocument_Update );
        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_SetName", UIWorkspaceDocument::UIWorkspaceDocument_SetName );
        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_IsDirty", UIWorkspaceDocument::UIWorkspaceDocument_IsDirty );
        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_MarkAsDirty", UIWorkspaceDocument::UIWorkspaceDocument_MarkAsDirty );
        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_Open", UIWorkspaceDocument::UIWorkspaceDocument_Open );
        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_RequestClose", UIWorkspaceDocument::UIWorkspaceDocument_RequestClose );
        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_ForceClose", UIWorkspaceDocument::UIWorkspaceDocument_ForceClose );
        ICall( "UIWorkspaceDocument::UIWorkspaceDocument_RegisterSaveDelegate",
               UIWorkspaceDocument::UIWorkspaceDocument_RegisterSaveDelegate );

        ICall( "UIMenuItem::UIMenuItem_Create", UIMenuItem::UIMenuItem_Create );
        ICall( "UIMenuItem::UIMenuItem_CreateWithText", UIMenuItem::UIMenuItem_CreateWithText );
        ICall( "UIMenuItem::UIMenuItem_CreateWithTextAndShortcut", UIMenuItem::UIMenuItem_CreateWithTextAndShortcut );
        ICall( "UIMenuItem::UIMenuItem_Destroy", UIMenuItem::UIMenuItem_Destroy );
        ICall( "UIMenuItem::UIMenuItem_SetText", UIMenuItem::UIMenuItem_SetText );
        ICall( "UIMenuItem::UIMenuItem_SetShortcut", UIMenuItem::UIMenuItem_SetShortcut );
        ICall( "UIMenuItem::UIMenuItem_SetTextColor", UIMenuItem::UIMenuItem_SetTextColor );
        ICall( "UIMenuItem::UIMenuItem_OnTrigger", UIMenuItem::UIMenuItem_OnTrigger );

        ICall( "UIMenuSeparator::UIMenuSeparator_Create", UIMenuSeparator::UIMenuSeparator_Create );
        ICall( "UIMenuSeparator::UIMenuSeparator_Destroy", UIMenuSeparator::UIMenuSeparator_Destroy );

        ICall( "UIMenu::UIMenu_Create", UIMenu::UIMenu_Create );
        ICall( "UIMenu::UIMenu_CreateWithText", UIMenu::UIMenu_CreateWithText );
        ICall( "UIMenu::UIMenu_Destroy", UIMenu::UIMenu_Destroy );
        ICall( "UIMenu::UIMenu_AddAction", UIMenu::UIMenu_AddAction );
        ICall( "UIMenu::UIMenu_AddMenu", UIMenu::UIMenu_AddMenu );
        ICall( "UIMenu::UIMenu_AddSeparator", UIMenu::UIMenu_AddSeparator );
        ICall( "UIMenu::UIMenu_Update", UIMenu::UIMenu_Update );

        ICall( "UIPropertyValue::UIPropertyValue_Create", UIPropertyValue::UIPropertyValue_Create );
        ICall( "UIPropertyValue::UIPropertyValue_CreateWithText", UIPropertyValue::UIPropertyValue_CreateWithText );
        ICall( "UIPropertyValue::UIPropertyValue_CreateWithTextAndOrientation",
               UIPropertyValue::UIPropertyValue_CreateWithTextAndOrientation );
        ICall( "UIPropertyValue::UIPropertyValue_Destroy", UIPropertyValue::UIPropertyValue_Destroy );
        ICall( "UIPropertyValue::UIPropertyValue_SetValue", UIPropertyValue::UIPropertyValue_SetValue );
        ICall( "UIPropertyValue::UIPropertyValue_SetValueFont", UIPropertyValue::UIPropertyValue_SetValueFont );
        ICall( "UIPropertyValue::UIPropertyValue_SetNameFont", UIPropertyValue::UIPropertyValue_SetNameFont );

        ICall( "UITextInput::UITextInput_Create", UITextInput::UITextInput_Create );
        ICall( "UITextInput::UITextInput_CreateWithText", UITextInput::UITextInput_CreateWithText );
        ICall( "UITextInput::UITextInput_Destroy", UITextInput::UITextInput_Destroy );
        ICall( "UITextInput::UITextInput_GetText", UITextInput::UITextInput_GetText );
        ICall( "UITextInput::UITextInput_SetHintText", UITextInput::UITextInput_SetHintText );
        ICall( "UITextInput::UITextInput_SetTextColor", UITextInput::UITextInput_SetTextColor );
        ICall( "UITextInput::UITextInput_SetBufferSize", UITextInput::UITextInput_SetBufferSize );
        ICall( "UITextInput::UITextInput_OnTextChanged", UITextInput::UITextInput_OnTextChanged );

        ICall( "UIProgressBar::UIProgressBar_Create", UIProgressBar::UIProgressBar_Create );
        ICall( "UIProgressBar::UIProgressBar_Destroy", UIProgressBar::UIProgressBar_Destroy );
        ICall( "UIProgressBar::UIProgressBar_SetProgressValue", UIProgressBar::UIProgressBar_SetProgressValue );
        ICall( "UIProgressBar::UIProgressBar_SetProgressColor", UIProgressBar::UIProgressBar_SetProgressColor );
        ICall( "UIProgressBar::UIProgressBar_SetText", UIProgressBar::UIProgressBar_SetText );
        ICall( "UIProgressBar::UIProgressBar_SetTextColor", UIProgressBar::UIProgressBar_SetTextColor );
        ICall( "UIProgressBar::UIProgressBar_SetThickness", UIProgressBar::UIProgressBar_SetThickness );

        ICall( "UIDropdownButton::UIDropdownButton_Create", UIDropdownButton::UIDropdownButton_Create );
        ICall( "UIDropdownButton::UIDropdownButton_Destroy", UIDropdownButton::UIDropdownButton_Destroy );
        ICall( "UIDropdownButton::UIDropdownButton_SetContent", UIDropdownButton::UIDropdownButton_SetContent );
        ICall( "UIDropdownButton::UIDropdownButton_SetContentSize", UIDropdownButton::UIDropdownButton_SetContentSize );
        ICall( "UIDropdownButton::UIDropdownButton_SetImage", UIDropdownButton::UIDropdownButton_SetImage );
        ICall( "UIDropdownButton::UIDropdownButton_SetText", UIDropdownButton::UIDropdownButton_SetText );
        ICall( "UIDropdownButton::UIDropdownButton_SetTextColor", UIDropdownButton::UIDropdownButton_SetTextColor );

        ICall( "UIContainer::UIContainer_Create", UIContainer::UIContainer_Create );
        ICall( "UIContainer::UIContainer_Destroy", UIContainer::UIContainer_Destroy );
        ICall( "UIContainer::UIContainer_SetContent", UIContainer::UIContainer_SetContent );

        ICall( "UITreeView::UITreeView_Create", UITreeView::UITreeView_Create );
        ICall( "UITreeView::UITreeView_Destroy", UITreeView::UITreeView_Destroy );
        ICall( "UITreeView::UITreeView_SetIndent", UITreeView::UITreeView_SetIndent );
        ICall( "UITreeView::UITreeView_SetIconSpacing", UITreeView::UITreeView_SetIconSpacing );
        ICall( "UITreeView::UITreeView_Add", UITreeView::UITreeView_Add );

        ICall( "UITreeViewNode::UITreeViewNode_Create", UITreeViewNode::UITreeViewNode_Create );
        ICall( "UITreeViewNode::UITreeViewNode_Destroy", UITreeViewNode::UITreeViewNode_Destroy );
        ICall( "UITreeViewNode::UITreeViewNode_SetIcon", UITreeViewNode::UITreeViewNode_SetIcon );
        ICall( "UITreeViewNode::UITreeViewNode_SetIndicator", UITreeViewNode::UITreeViewNode_SetIndicator );
        ICall( "UITreeViewNode::UITreeViewNode_SetText", UITreeViewNode::UITreeViewNode_SetText );
        ICall( "UITreeViewNode::UITreeViewNode_SetTextColor", UITreeViewNode::UITreeViewNode_SetTextColor );
        ICall( "UITreeViewNode::UITreeViewNode_Add", UITreeViewNode::UITreeViewNode_Add );

        ICall( "UIFileTree::UIFileTree_Create", UIFileTree::UIFileTree_Create );
        ICall( "UIFileTree::UIFileTree_Destroy", UIFileTree::UIFileTree_Destroy );
        ICall( "UIFileTree::UIFileTree_Add", UIFileTree::UIFileTree_Add );
        // ICall( "UIFileTree::UIFileView_SetIndent", UIFileTree::UITreeView_SetIndent );
        // ICall( "UIFileTreeNode::UIFileTreeNode_Create", UIFileTreeNode::UITreeViewNode_Create );
        // ICall( "UIFileTreeNode::UIFileTreeNode_Destroy", UIFileTreeNode::UITreeViewNode_Destroy );
        // ICall( "UIFileTreeNode::UIFileTreeNode_SetIcon", UIFileTreeNode::UITreeViewNode_SetIcon );
        // ICall( "UIFileTreeNode::UIFileTreeNode_SetIndicator", UIFileTreeNode::UITreeViewNode_SetIndicator );
        // ICall( "UIFileTreeNode::UIFileTreeNode_SetText", UIFileTreeNode::UITreeViewNode_SetText );
        // ICall( "UIFileTreeNode::UIFileTreeNode_SetTextColor", UIFileTreeNode::UITreeViewNode_SetTextColor );
        // ICall( "UIFileTreeNode::UIFileTreeNode_Add", UIFileTreeNode::UITreeViewNode_Add );

        ICall( "UIColorButton::UIColorButton_Create", UIColorButton::UIColorButton_Create );
        ICall( "UIColorButton::UIColorButton_Destroy", UIColorButton::UIColorButton_Destroy );

    }

} // namespace SE::Core