#include "Runtime.h"

// #include "Core/File.h"
#include "Core/Logging.h"
#include "Core/Memory.h"
#include "Core/String.h"

// #include "Engine/Engine.h"

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

#include "Utils.h"

#include "InternalCalls.h"

namespace fs = std::filesystem;
using namespace SE::MonoInternalCalls;

namespace SE::Core
{
    using PathList     = std::vector<path_t>;
    using ClassMapping = std::map<string_t, DotNetClass>;

    struct sAssemblyData
    {
        path_t                 mPath           = "";
        path_t                 mFilename       = "";
        string_t              mCategory       = "";
        MonoAssembly            *mAssembly       = nullptr;
        MonoImage               *mImage          = nullptr;
        bool                     mNeedsReloading = false;
        bool                     mFileExists     = false;
        std::vector<string_t> mClasses{};

        sAssemblyData()                        = default;
        sAssemblyData( const sAssemblyData & ) = default;
    };

    using AssemblyMapping = std::map<path_t, sAssemblyData>;

    struct sMonoRuntimeData
    {
        MonoDomain     *mRootDomain = nullptr;
        MonoDomain     *mAppDomain  = nullptr;
        sAssemblyData   mCoreAssembly{};
        PathList        mAppAssemblyFiles = {};
        AssemblyMapping mAssemblies       = {};
        ClassMapping    mClasses          = {};

        std::map<string_t, std::vector<sAssemblyData *>> mCategories;
        HINSTANCE                                           mMonoPosixHelper;
    };

    static sMonoRuntimeData *sRuntimeData = nullptr;

    MonoObject *DotNetRuntime::InstantiateClass( MonoClass *aMonoClass, bool aIsCore )
    {
        MonoObject *aInstance = mono_object_new( sRuntimeData->mAppDomain, aMonoClass );

        return aInstance;
    }

    void DotNetRuntime::LoadCoreAssembly( const path_t &aFilepath )
    {
        sRuntimeData->mCoreAssembly.mPath     = aFilepath.parent_path();
        sRuntimeData->mCoreAssembly.mFilename = aFilepath.filename();
    }

    void DotNetRuntime::AddAppAssemblyPath( const path_t &aFilepath, string_t const &aCategory )
    {
        if( std::find( sRuntimeData->mAppAssemblyFiles.begin(), sRuntimeData->mAppAssemblyFiles.end(), aFilepath ) !=
            sRuntimeData->mAppAssemblyFiles.end() )
            return;

        if( !fs::exists( aFilepath.parent_path() ) ) return;

        sRuntimeData->mAppAssemblyFiles.push_back( aFilepath );

        sRuntimeData->mAssemblies.emplace( aFilepath, sAssemblyData{} );
        sRuntimeData->mAssemblies[aFilepath].mPath     = aFilepath.parent_path();
        sRuntimeData->mAssemblies[aFilepath].mFilename = aFilepath.filename();

        Ref<path_t> lAssemblyFilePath = New<path_t>( aFilepath );

        if( sRuntimeData->mCategories.find( aCategory ) == sRuntimeData->mCategories.end() )
            sRuntimeData->mCategories[aCategory] = std::vector<sAssemblyData *>{};
        sRuntimeData->mCategories[aCategory].push_back( &sRuntimeData->mAssemblies[aFilepath] );
    }

    void DotNetRuntime::Initialize( path_t &aMonoPath, const path_t &aCoreAssemblyPath )
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

    void DotNetRuntime::InitMono( path_t &aMonoPath )
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

    MonoString *DotNetRuntime::NewString( string_t const &aString )
    {
        return mono_string_new( sRuntimeData->mAppDomain, aString.c_str() );
    }

    string_t DotNetRuntime::NewString( MonoString *aString )
    {
        auto *lCharacters = mono_string_to_utf8( aString );
        auto  lString     = string_t( mono_string_to_utf8( aString ) );
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

    DotNetClass &DotNetRuntime::GetClassType( const string_t &aClassName )
    {
        if( sRuntimeData->mClasses.find( aClassName ) != sRuntimeData->mClasses.end() ) return sRuntimeData->mClasses[aClassName];

        for( auto const &[lPath, lAssembly] : sRuntimeData->mAssemblies )
        {
            std::size_t lPos       = aClassName.find_last_of( "." );
            string_t lNameSpace = aClassName.substr( 0, lPos );
            string_t lClassName = aClassName.substr( lPos + 1 );

            MonoClass *lClass = mono_class_from_name( lAssembly.mImage, lNameSpace.c_str(), lClassName.c_str() );
            if( lClass != nullptr )
            {
                sRuntimeData->mClasses[aClassName] = DotNetClass( lClass, aClassName, lClassName, lAssembly.mImage, lPath, true );

                return sRuntimeData->mClasses[aClassName];
            }
        }
        return DotNetClass{};
    }

    MonoType *DotNetRuntime::GetCoreTypeFromName( string_t &aName )
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

    static void ICall( string_t const &aName, void *aFunction )
    {
        auto lFullName = fmt::format( "SpockEngine.{}", aName );

        mono_add_internal_call( lFullName.c_str(), aFunction );
    }

    void DotNetRuntime::RegisterInternalCppFunctions()
    {
        using namespace SE::Core::Interop;

        // ICall( "UIColor::GetStyleColor", SE::Core::UI::GetStyleColor );
    }
} // namespace SE::Core