#include "MonoRuntime.h"

#include "Core/Logging.h"
#include "Core/Memory.h"

#include "Engine/Engine.h"

#include "Scene/Scene.h"

#include "mono/jit/jit.h"
#include "mono/metadata/assembly.h"
#include "mono/metadata/object.h"
#include "mono/metadata/tabledefs.h"

#include <unordered_map>

#ifdef WIN32_LEAN_AND_MEAN
#    undef WIN32_LEAN_AND_MEAN
#endif
#include "Core/FileWatch.hpp"

#include "EntityRegistry.h"
#include "InternalCalls.h"
#include "MonoScriptUtils.h"

namespace fs = std::filesystem;

namespace SE::Core
{
    using PathList     = std::vector<fs::path>;
    using ClassMapping = std::map<std::string, MonoScriptClass>;

    struct sAssemblyData
    {
        fs::path      mPath           = "";
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

        std::map<std::string, sAssemblyData *> mCategories;
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

        std::map<std::string, MonoScriptClass> LoadImageClasses( MonoImage *aImage, fs::path aPath )
        {
            std::map<std::string, MonoScriptClass> lClasses{};

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

                if( mono_class_from_name( aImage, lNameSpace, lClassName ) )
                    lClasses[lFullName] = MonoScriptClass( lNameSpace, lClassName, aImage, aPath );
            }

            return lClasses;
        }
    } // namespace

    uint32_t MonoRuntime::CountAssemblies() { return sRuntimeData->mAppAssemblyFiles.size(); }

    bool MonoRuntime::AssembliesNeedReloading()
    {
        for( auto const &[lKey, lValue] : sRuntimeData->mAssemblies )
            if( lValue.mNeedsReloading ) return true;
        return false;
    }

    void MonoRuntime::GetAssemblies( std::vector<fs::path> &lOut )
    {
        lOut.resize( sRuntimeData->mAppAssemblyFiles.size() );

        uint32_t i = 0;
        for( auto const &lFile : sRuntimeData->mAppAssemblyFiles ) lOut[i++] = lFile.string();
    }

    MonoObject *MonoRuntime::InstantiateClass( MonoClass *aMonoClass, bool aIsCore )
    {
        MonoObject *aInstance = mono_object_new( sRuntimeData->mAppDomain, aMonoClass );

        return aInstance;
    }

    void MonoRuntime::LoadCoreAssembly( const fs::path &aFilepath )
    {
        sRuntimeData->mAppDomain = mono_domain_create_appdomain( "SE_Runtime", nullptr );
        mono_domain_set_config( sRuntimeData->mAppDomain, ".", "XXX" );
        mono_domain_set( sRuntimeData->mAppDomain, true );

        sRuntimeData->mCoreAssembly.mPath       = aFilepath;
        sRuntimeData->mCoreAssembly.mCategory   = "CORE";
        sRuntimeData->mCoreAssembly.mFileExists = fs::exists( aFilepath );
        sRuntimeData->mCoreAssembly.mAssembly   = Mono::Utils::LoadMonoAssembly( aFilepath );
        sRuntimeData->mCoreAssembly.mImage      = mono_assembly_get_image( sRuntimeData->mCoreAssembly.mAssembly );

        sRuntimeData->mCoreAssembly.mNeedsReloading = false;

        sRuntimeData->mCoreClasses = {};
        MergeMaps( sRuntimeData->mCoreClasses, LoadImageClasses( sRuntimeData->mCoreAssembly.mImage, aFilepath ) );
    }

    static void OnAppAssemblyFileSystemEvent( const std::string &path, const filewatch::Event change_type )
    {
        if( !sRuntimeData->mAssemblies[path].mNeedsReloading && change_type == filewatch::Event::modified )
            sRuntimeData->mAssemblies[path].mNeedsReloading = true;
    }

    void MonoRuntime::AddAppAssemblyPath( const fs::path &aFilepath )
    {
        if( std::find( sRuntimeData->mAppAssemblyFiles.begin(), sRuntimeData->mAppAssemblyFiles.end(), aFilepath ) !=
            sRuntimeData->mAppAssemblyFiles.end() )
            return;

        if( !fs::exists( aFilepath ) ) return;

        sRuntimeData->mAppAssemblyFiles.push_back( aFilepath );

        // sRuntimeData->mAssemblyReloadPending = true;
        sRuntimeData->mAssemblies[aFilepath] = sAssemblyData{};

        sRuntimeData->mAssemblies[aFilepath].mFileExists = fs::exists( aFilepath );
        sRuntimeData->mAssemblies[aFilepath].mCategory   = "";
        sRuntimeData->mAssemblies[aFilepath].mWatcher =
            std::make_shared<filewatch::FileWatch<std::string>>( aFilepath.string(), OnAppAssemblyFileSystemEvent );
    }

    void MonoRuntime::Initialize( fs::path &aMonoPath, const fs::path &aCoreAssemblyPath )
    {
        if( sRuntimeData != nullptr ) return;

        sRuntimeData = new sMonoRuntimeData();

        InitMono( aMonoPath );

        RegisterInternalCppFunctions();

        LoadCoreAssembly( aCoreAssemblyPath );
    }

    void MonoRuntime::RegisterComponentTypes()
    {
        RegisterComponentType<sTag>();
        RegisterComponentType<sNodeTransformComponent>();
        RegisterComponentType<sLightComponent>();
    }

    void MonoRuntime::Shutdown()
    {
        ShutdownMono();

        delete sRuntimeData;

        sRuntimeData = nullptr;
    }

    void MonoRuntime::InitMono( fs::path &aMonoPath )
    {
        mono_set_assemblies_path( aMonoPath.string().c_str() );

        MonoDomain *lRootDomain = mono_jit_init( "SpockEngineRuntime" );

        sRuntimeData->mRootDomain = lRootDomain;
    }

    void MonoRuntime::ShutdownMono()
    {
        mono_domain_set( mono_get_root_domain(), false );

        mono_domain_unload( sRuntimeData->mAppDomain );
        sRuntimeData->mAppDomain = nullptr;

        mono_jit_cleanup( sRuntimeData->mRootDomain );
        sRuntimeData->mRootDomain = nullptr;
    }

    MonoString *MonoRuntime::NewString( std::string const &aString )
    {
        return mono_string_new( sRuntimeData->mAppDomain, aString.c_str() );
    }

    std::string MonoRuntime::NewString( MonoString *aString ) { return std::string( mono_string_to_utf8( aString ) ); }

    void MonoRuntime::LoadAssemblyClasses()
    {
        sRuntimeData->mClasses = {};
        if( sRuntimeData->mAssemblies.empty() ) return;

        for( auto const &lAssemblyPath : sRuntimeData->mAppAssemblyFiles )
        {
            const auto lAssemblyImage = sRuntimeData->mAssemblies[lAssemblyPath].mImage;

            MergeMaps( sRuntimeData->mClasses, LoadImageClasses( lAssemblyImage, lAssemblyPath ) );
        }
    }

    void MonoRuntime::RecreateClassTree()
    {
        std::map<std::string, MonoScriptClass *> lLookupTable;

        for( auto &[lKey, lValue] : sRuntimeData->mClasses )
        {
            lValue.mDerived.clear();
            lLookupTable[lValue.FullName()] = &lValue;
        }

        for( auto &[lKey, lValue] : sRuntimeData->mCoreClasses )
        {
            lValue.mDerived.clear();
            lLookupTable[lValue.FullName()] = &lValue;
        }

        for( auto &[lKey, lValue] : sRuntimeData->mClasses )
        {
            auto *lParentClass = mono_class_get_parent( lValue.Class() );
            if( !lParentClass ) continue;

            auto lParentClassFullName = std::string( mono_class_get_name( lParentClass ) );

            if( lLookupTable.find( lParentClassFullName ) != lLookupTable.end() )
            {
                lValue.mParent = lLookupTable[lParentClassFullName];
                lLookupTable[lParentClassFullName]->mDerived.push_back( &lValue );
            }
        }
    }

    void MonoRuntime::ReloadAssemblies()
    {
        if( !AssembliesNeedReloading() ) return;

        mono_domain_set( mono_get_root_domain(), true );
        if( sRuntimeData->mAppDomain != nullptr ) mono_domain_unload( sRuntimeData->mAppDomain );

        sRuntimeData->mAssemblies = {};

        LoadCoreAssembly( sRuntimeData->mCoreAssembly.mPath );

        for( auto const &lFile : sRuntimeData->mAppAssemblyFiles )
        {
            sRuntimeData->mAssemblies[lFile] = sAssemblyData{};

            sRuntimeData->mAssemblies[lFile].mPath       = lFile;
            sRuntimeData->mAssemblies[lFile].mFileExists = fs::exists( lFile );
            sRuntimeData->mAssemblies[lFile].mCategory   = "";
            sRuntimeData->mAssemblies[lFile].mAssembly   = Mono::Utils::LoadMonoAssembly( lFile );
            sRuntimeData->mAssemblies[lFile].mImage      = mono_assembly_get_image( sRuntimeData->mAssemblies[lFile].mAssembly );

            sRuntimeData->mAssemblies[lFile].mNeedsReloading = false;
        }

        LoadAssemblyClasses();
        RecreateClassTree();

        RegisterComponentTypes();
    }

    MonoScriptClass &MonoRuntime::GetClassType( const std::string &aClassName ) { return sRuntimeData->mClasses[aClassName]; }

    MonoType *MonoRuntime::GetCoreTypeFromName( std::string &aName )
    {
        MonoType *lMonoType = mono_reflection_type_from_name( aName.data(), sRuntimeData->mCoreAssembly.mImage );
        if( !lMonoType )
        {
            SE::Logging::Info( "Could not find type '{}'", aName );

            return nullptr;
        }

        return lMonoType;
    }

    void MonoRuntime::RegisterInternalCppFunctions()
    {
        using namespace MonoInternalCalls;

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
    }
} // namespace SE::Core