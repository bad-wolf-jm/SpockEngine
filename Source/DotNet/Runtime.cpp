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

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "EntityRegistry.h"
#include "InternalCalls.h"
#include "Utils.h"

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
#include "UI/Components/PropertyValue.h"
#include "UI/Components/Table.h"
#include "UI/Components/TextOverlay.h"
#include "UI/Components/TextToggleButton.h"
#include "UI/Components/Workspace.h"
#include "UI/Components/TextInput.h"
#include "UI/Dialog.h"
#include "UI/Form.h"
#include "UI/Layouts/BoxLayout.h"
#include "UI/Layouts/ZLayout.h"

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
        MonoDomain     *mRootDomain = nullptr;
        MonoDomain     *mAppDomain  = nullptr;
        sAssemblyData   mCoreAssembly{};
        ClassMapping    mCoreClasses      = {};
        PathList        mAppAssemblyFiles = {};
        AssemblyMapping mAssemblies       = {};
        ClassMapping    mClasses          = {};

        std::function<void( std::string )> mConsoleOut;

        std::map<std::string, std::vector<sAssemblyData *>> mCategories;
        HINSTANCE                                           mMonoPosixHelper;
    };

    static sMonoRuntimeData *sRuntimeData = nullptr;

    namespace
    {
        template <typename _Tx, typename _Ty>
        void MergeMaps( std::map<_Tx, _Ty> &aDest, std::map<_Tx, _Ty> const &aSrc )
        {
            for( auto &[lKey, lValue] : aSrc )
            {
                if( aDest.find( lKey ) == aDest.end() )
                    aDest[lKey] = lValue;
                else
                    aDest[lKey] = lValue;
            }
        }

        std::map<std::string, DotNetClass> LoadImageClasses( MonoImage *aImage, fs::path aPath )
        {
            std::map<std::string, DotNetClass> lClasses{};

            if( !aImage ) return lClasses;

            const MonoTableInfo *lTypeDefinitionsTable = mono_image_get_table_info( aImage, MONO_TABLE_TYPEDEF );
            int32_t              lTypesCount           = mono_table_info_get_rows( lTypeDefinitionsTable );

            for( int32_t i = 0; i < lTypesCount; i++ )
            {
                uint32_t lCols[MONO_TYPEDEF_SIZE];
                mono_metadata_decode_row( lTypeDefinitionsTable, i, lCols, MONO_TYPEDEF_SIZE );

                const char *lNameSpace = mono_metadata_string_heap( aImage, lCols[MONO_TYPEDEF_NAMESPACE] );
                const char *lClassName = mono_metadata_string_heap( aImage, lCols[MONO_TYPEDEF_NAME] );

                if( !std::strncmp( lClassName, "<", 1 ) ) continue;

                std::string lFullName;
                if( strlen( lNameSpace ) != 0 )
                    lFullName = fmt::format( "{}.{}", lNameSpace, lClassName );
                else
                    lFullName = lClassName;

                MonoClass *lClass = mono_class_from_name( aImage, lNameSpace, lClassName );
                if( lClass != nullptr )
                {
                    // Add nested classes
                    void *lIterator = nullptr;
                    while( MonoClass *lNestedClass = mono_class_get_nested_types( lClass, &lIterator ) )
                    {
                        const char *lClassName = mono_class_get_name( lNestedClass );

                        if( !std::strncmp( lClassName, "<", 1 ) ) continue;

                        auto lNestedClassName = fmt::format( "{}.{}", lFullName, lClassName );

                        if( lClasses.find( lNestedClassName ) == lClasses.end() )
                            lClasses[lNestedClassName] = DotNetClass( lNestedClass, lFullName, lClassName, aImage, aPath, true );
                    }

                    lClasses[lFullName] = DotNetClass( lNameSpace, lClassName, aImage, aPath );

                    lClass = mono_class_get_parent( lClass );
                    while( lClass != nullptr )
                    {
                        const char *lNameSpace = mono_class_get_namespace( lClass );
                        const char *lClassName = mono_class_get_name( lClass );

                        std::string lFullName;
                        if( strlen( lNameSpace ) != 0 )
                            lFullName = fmt::format( "{}.{}", lNameSpace, lClassName );
                        else
                            lFullName = lClassName;

                        if( lClasses.find( lFullName ) == lClasses.end() )
                            lClasses[lFullName] = DotNetClass( lClass, lNameSpace, lClassName, aImage, aPath );

                        lClass = mono_class_get_parent( lClass );
                    }
                }
            }

            return lClasses;
        }
    } // namespace

    uint32_t DotNetRuntime::CountAssemblies() { return sRuntimeData->mAppAssemblyFiles.size(); }

    std::vector<std::string> DotNetRuntime::GetClassNames()
    {
        std::vector<std::string> lResult;
        for( auto const &[lName, lValue] : sRuntimeData->mClasses )
        {
            lResult.push_back( lName );
        }

        return lResult;
    }

    std::map<std::string, DotNetClass> &DotNetRuntime::GetClasses() { return sRuntimeData->mClasses; }

    bool DotNetRuntime::AssembliesNeedReloading()
    {
        for( auto const &[lKey, lValue] : sRuntimeData->mAssemblies )
            if( lValue.mNeedsReloading ) return true;
        return false;
    }

    void DotNetRuntime::GetAssemblies( std::vector<fs::path> &lOut )
    {
        lOut.resize( sRuntimeData->mAppAssemblyFiles.size() );

        uint32_t i = 0;
        for( auto const &lFile : sRuntimeData->mAppAssemblyFiles ) lOut[i++] = lFile.string();
    }

    MonoObject *DotNetRuntime::InstantiateClass( MonoClass *aMonoClass, bool aIsCore )
    {
        MonoObject *aInstance = mono_object_new( sRuntimeData->mAppDomain, aMonoClass );

        return aInstance;
    }

    void DotNetRuntime::LoadCoreAssembly( const fs::path &aFilepath )
    {
        sRuntimeData->mAppDomain = mono_domain_create_appdomain( "SE_Runtime", nullptr );
        mono_domain_set_config( sRuntimeData->mAppDomain, ".", "XXX" );
        mono_domain_set( sRuntimeData->mAppDomain, true );

        sRuntimeData->mCoreAssembly.mPath       = aFilepath.parent_path();
        sRuntimeData->mCoreAssembly.mFilename   = aFilepath.filename();
        sRuntimeData->mCoreAssembly.mCategory   = "CORE";
        sRuntimeData->mCoreAssembly.mFileExists = fs::exists( aFilepath );
        sRuntimeData->mCoreAssembly.mAssembly   = Mono::Utils::LoadMonoAssembly( aFilepath );
        sRuntimeData->mCoreAssembly.mImage      = mono_assembly_get_image( sRuntimeData->mCoreAssembly.mAssembly );

        sRuntimeData->mCoreAssembly.mNeedsReloading = false;

        sRuntimeData->mClasses = {};
        MergeMaps( sRuntimeData->mClasses, LoadImageClasses( sRuntimeData->mCoreAssembly.mImage, aFilepath ) );
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
        sRuntimeData->mAssemblies[aFilepath].mPath           = aFilepath.parent_path();
        sRuntimeData->mAssemblies[aFilepath].mFilename       = aFilepath.filename();
        sRuntimeData->mAssemblies[aFilepath].mFileExists     = fs::exists( aFilepath );
        sRuntimeData->mAssemblies[aFilepath].mCategory       = aCategory;
        sRuntimeData->mAssemblies[aFilepath].mNeedsReloading = true;

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

    void DotNetRuntime::RegisterComponentTypes()
    {
        RegisterComponentType<sTag>();
        RegisterComponentType<sNodeTransformComponent>();
        RegisterComponentType<sLightComponent>();
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

        MonoDomain *lRootDomain = mono_jit_init( "SpockEngineRuntime" );

        sRuntimeData->mRootDomain = lRootDomain;
    }

    void DotNetRuntime::ShutdownMono()
    {
        mono_domain_set( mono_get_root_domain(), false );

        mono_domain_unload( sRuntimeData->mAppDomain );
        sRuntimeData->mAppDomain = nullptr;

        mono_jit_cleanup( sRuntimeData->mRootDomain );
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

    void DotNetRuntime::LoadAssemblyClasses()
    {
        if( sRuntimeData->mAssemblies.empty() ) return;

        for( auto const &lAssemblyPath : sRuntimeData->mAppAssemblyFiles )
        {
            const auto lAssemblyImage = sRuntimeData->mAssemblies[lAssemblyPath].mImage;

            MergeMaps( sRuntimeData->mClasses, LoadImageClasses( lAssemblyImage, lAssemblyPath ) );
        }
    }

    void DotNetRuntime::RecreateClassTree()
    {
        std::map<std::string, DotNetClass *> lLookupTable;

        for( auto &[lKey, lValue] : sRuntimeData->mClasses )
        {
            lValue.mParent = nullptr;
            lValue.mDerived.clear();
            lLookupTable[lValue.FullName()] = &lValue;
        }

        for( auto &[lKey, lValue] : sRuntimeData->mClasses )
        {
            if( lKey == "Test.TestScript" )
            {
                SE::Logging::Info( "FOO" );
            }

            auto *lParentClass = mono_class_get_parent( lValue.Class() );
            if( !lParentClass ) continue;

            auto lParentClassNamespace = std::string( mono_class_get_namespace( lParentClass ) );
            auto lParentClassName      = std::string( mono_class_get_name( lParentClass ) );
            auto lParentClassFullName  = fmt::format( "{}.{}", lParentClassNamespace, lParentClassName );

            if( lLookupTable.find( lParentClassFullName ) != lLookupTable.end() )
            {
                lValue.mParent = lLookupTable[lParentClassFullName];
                lLookupTable[lParentClassFullName]->mDerived.push_back( &lValue );
            }
        }
    }

    void DotNetRuntime::ReloadAssemblies()
    {
        if( !AssembliesNeedReloading() ) return;

        mono_domain_set( mono_get_root_domain(), true );
        if( sRuntimeData->mAppDomain != nullptr ) mono_domain_unload( sRuntimeData->mAppDomain );

        LoadCoreAssembly( sRuntimeData->mCoreAssembly.mPath / sRuntimeData->mCoreAssembly.mFilename );

        for( auto &[lFile, lData] : sRuntimeData->mAssemblies )
        {
            lData.mFileExists = fs::exists( lFile );
            if( lData.mFileExists )
            {
                lData.mAssembly = Mono::Utils::LoadMonoAssembly( lData.mPath / lData.mFilename );
                lData.mImage    = mono_assembly_get_image( lData.mAssembly );
            }
            else
            {
                lData.mAssembly = nullptr;
                lData.mImage    = nullptr;
            }

            lData.mNeedsReloading = false;
        }

        LoadAssemblyClasses();
        RecreateClassTree();

        RegisterComponentTypes();
    }

    DotNetClass &DotNetRuntime::GetClassType( const std::string &aClassName )
    {
        if( sRuntimeData->mClasses.find( aClassName ) != sRuntimeData->mClasses.end() ) return sRuntimeData->mClasses[aClassName];

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

    void DotNetRuntime::DisplayAssemblies()
    {
        bool        lAllFilesExist = true;
        math::vec2  lWindowSize    = UI::GetAvailableContentSpace();
        auto        lDrawList      = ImGui::GetWindowDrawList();
        const float lFontSize      = lDrawList->_Data->FontSize;
        float       lCircleXOffset = ImGui::CalcTextSize( "M" ).x / 2.0f;

        for( auto const &[lCategory, lAssemblies] : sRuntimeData->mCategories )
        {
            bool lCategoryAllFilesExist = true;
            for( auto const &lAssembly : lAssemblies ) lCategoryAllFilesExist &= lAssembly->mFileExists;

            bool lNeedsReload = false;
            for( auto const &lAssembly : lAssemblies ) lNeedsReload |= lAssembly->mNeedsReloading;

            if( lCategoryAllFilesExist && lNeedsReload )
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 255.0f / 255.0f, 229.0f / 255.0f, 159.0f / 255.0f, 1.0f } );
            else if( !lCategoryAllFilesExist )
                ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 160.0f / 255.0f, 69.0f / 255.0f, 55.0f / 255.0f, 1.0f } );

            UI::SetCursorPositionX( 10.0f );
            auto lPos = UI::GetCurrentCursorPosition();
            UI::Text( "{}", lCategory );
            if( lCategoryAllFilesExist && lNeedsReload )
                ImGui::PopStyleColor();
            else if( !lCategoryAllFilesExist )
                ImGui::PopStyleColor();

            UI::SetCursorPosition( math::vec2{ lWindowSize.x - 20.0f, ImGui::GetCursorPos().y - 12.0f } );
            if( lCategoryAllFilesExist && lNeedsReload )
                lDrawList->AddCircleFilled( ImGui::GetCursorScreenPos() + ImVec2{ lCircleXOffset, 0.0f }, 4,
                                            IM_COL32( 255, 229, 159, 255 ), 16 );
            else if( !lCategoryAllFilesExist )
                lDrawList->AddCircleFilled( ImGui::GetCursorScreenPos() + ImVec2{ lCircleXOffset, 0.0f }, 4,
                                            IM_COL32( 160, 69, 55, 255 ), 16 );
            else
                lDrawList->AddCircleFilled( ImGui::GetCursorScreenPos() + ImVec2{ lCircleXOffset, 0.0f }, 4,
                                            IM_COL32( 255, 255, 255, 255 ), 16 );
            UI::SetCursorPosition( lPos + math::vec2{ 0.0f, lFontSize } );

            for( auto const &lAssembly : lAssemblies )
            {
                UI::SetCursorPositionX( 30.0f );
                if( lAssembly->mNeedsReloading && lAssembly->mFileExists )
                    ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 255.0f / 255.0f, 229.0f / 255.0f, 159.0f / 255.0f, 1.0f } );
                else if( !lAssembly->mFileExists )
                    ImGui::PushStyleColor( ImGuiCol_Text, ImVec4{ 160.0f / 255.0f, 69.0f / 255.0f, 55.0f / 255.0f, 1.0f } );

                Text( lAssembly->mFilename.stem().string() );
                if( lAssembly->mNeedsReloading && lAssembly->mFileExists )
                {
                    UI::SameLine();
                    UI::SetCursorPositionX( lWindowSize.x - 20.0f );
                    UI::Text( "M" );
                    ImGui::PopStyleColor();
                }
                else if( !lAssembly->mFileExists )
                {
                    UI::SameLine();
                    UI::SetCursorPositionX( lWindowSize.x - 20.0f );
                    UI::Text( "D" );
                    ImGui::PopStyleColor();
                    lAllFilesExist = false;
                }
            }
        }

        if( DotNetRuntime::AssembliesNeedReloading() )
            if( UI::Button( "Reload assemblies", { 150.0f, 30.0f } ) ) DotNetRuntime::ReloadAssemblies();
    }

    void DotNetRuntime::OnConsoleOut( std::function<void( std::string const & )> aFunction ) { sRuntimeData->mConsoleOut = aFunction; }

    void DotNetRuntime::ConsoleWrite( MonoString *aBuffer )
    {
        if( sRuntimeData->mConsoleOut ) sRuntimeData->mConsoleOut( DotNetRuntime::NewString( aBuffer ) );
    }

    static MonoString *OpenFile( MonoString *aFilter )
    {
        auto  lFilter     = DotNetRuntime::NewString( aFilter );
        char *lCharacters = lFilter.data();

        for( uint32_t i = 0; i < lFilter.size(); i++ ) lCharacters[i] = ( lCharacters[i] == '|' ) ? '\0' : lCharacters[i];
        auto lFilePath = FileDialogs::OpenFile( SE::Core::Engine::GetInstance()->GetMainApplicationWindow(), lFilter.c_str() );

        if( lFilePath.has_value() ) return DotNetRuntime::NewString( lFilePath.value() );

        return DotNetRuntime::NewString( "" );
        ;
    }

    void DotNetRuntime::RegisterInternalCppFunctions()
    {
        using namespace MonoInternalCalls;

        mono_add_internal_call( "SpockEngine.CppCall::Console_Write", ConsoleWrite );
        mono_add_internal_call( "SpockEngine.CppCall::OpenFile", OpenFile );

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

        SE_ADD_INTERNAL_CALL( UI_Text );
        SE_ADD_INTERNAL_CALL( UI_Button );

        mono_add_internal_call( "SpockEngine.UIColor::GetStyleColor", SE::Core::UI::GetStyleColor );

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
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_SetRect", UIBaseImage::UIBaseImage_SetRect );
        mono_add_internal_call( "SpockEngine.UIBaseImage::UIBaseImage_SetTintColor", UIBaseImage::UIBaseImage_SetTintColor );

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
        mono_add_internal_call( "SpockEngine.UICheckBox::UIComboBox_SetCurrent", UIComboBox::UIComboBox_SetCurrent );
        mono_add_internal_call( "SpockEngine.UIComboBox::UIComboBox_SetItemList", UIComboBox::UIComboBox_SetItemList );

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

        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_Create", UIZLayout::UIZLayout_Create );
        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_Destroy", UIZLayout::UIZLayout_Destroy );
        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_AddAlignedNonFixed", UIZLayout::UIZLayout_AddAlignedNonFixed );
        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_AddNonAlignedNonFixed", UIZLayout::UIZLayout_AddNonAlignedNonFixed );
        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_AddAlignedFixed", UIZLayout::UIZLayout_AddAlignedFixed );
        mono_add_internal_call( "SpockEngine.UIZLayout::UIZLayout_AddNonAlignedFixed", UIZLayout::UIZLayout_AddNonAlignedFixed );

        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_Create", sFloat64Column::UIFloat64Column_Create );
        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_CreateFull",
                                sFloat64Column::UIFloat64Column_CreateFull );
        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_Destroy", sFloat64Column::UIFloat64Column_Destroy );
        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_Clear", sFloat64Column::UIFloat64Column_Clear );
        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_SetData", sFloat64Column::UIFloat64Column_SetData );
        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_SetDataWithForegroundColor",
                                sFloat64Column::UIFloat64Column_SetDataWithForegroundColor );
        mono_add_internal_call( "SpockEngine.UIFloat64Column::UIFloat64Column_SetDataWithForegroundAndBackgroundColor",
                                sFloat64Column::UIFloat64Column_SetDataWithForegroundAndBackgroundColor );

        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_Create", sUint32Column::UIUint32Column_Create );
        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_CreateFull", sUint32Column::UIUint32Column_CreateFull );
        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_Destroy", sUint32Column::UIUint32Column_Destroy );
        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_Clear", sUint32Column::UIUint32Column_Clear );
        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_SetData", sUint32Column::UIUint32Column_SetData );
        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_SetDataWithForegroundColor",
                                sUint32Column::UIUint32Column_SetDataWithForegroundColor );
        mono_add_internal_call( "SpockEngine.UIUint32Column::UIUint32Column_SetDataWithForegroundAndBackgroundColor",
                                sUint32Column::UIUint32Column_SetDataWithForegroundAndBackgroundColor );

        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_Create", sStringColumn::UIStringColumn_Create );
        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_CreateFull", sStringColumn::UIStringColumn_CreateFull );
        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_Destroy", sStringColumn::UIStringColumn_Destroy );
        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_Clear", sStringColumn::UIStringColumn_Clear );
        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_SetData", sStringColumn::UIStringColumn_SetData );
        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_SetDataWithForegroundColor",
                                sStringColumn::UIStringColumn_SetDataWithForegroundColor );
        mono_add_internal_call( "SpockEngine.UIStringColumn::UIStringColumn_SetDataWithForegroundAndBackgroundColor",
                                sStringColumn::UIStringColumn_SetDataWithForegroundAndBackgroundColor );

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

        mono_add_internal_call( "SpockEngine.UITextOverlay::UITextOverlay_Create", UITextOverlay::UITextOverlay_Create );
        mono_add_internal_call( "SpockEngine.UITextOverlay::UITextOverlay_Destroy", UITextOverlay::UITextOverlay_Destroy );
        mono_add_internal_call( "SpockEngine.UITextOverlay::UITextOverlay_AddText", UITextOverlay::UITextOverlay_AddText );

        mono_add_internal_call( "SpockEngine.UIWorkspace::UIWorkspace_Create", UIWorkspace::UIWorkspace_Create );
        mono_add_internal_call( "SpockEngine.UIWorkspace::UIWorkspace_Destroy", UIWorkspace::UIWorkspace_Destroy );
        mono_add_internal_call( "SpockEngine.UIWorkspace::UIWorkspace_Add", UIWorkspace::UIWorkspace_Add );

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
    }

} // namespace SE::Core