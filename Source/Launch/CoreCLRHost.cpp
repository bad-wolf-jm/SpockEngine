#include "CoreCLRHost.h"
#include "Core/Logging.h"
#include <codecvt>
#include <filesystem>
#include <locale>
#include <set>
#include <sstream>
#include <string>

#include <nethost.h>

namespace SE::Core
{
    using directory_iterator = std::filesystem::directory_iterator;

    static std::string make_ascii_string( wchar_t *aCharacters )
    {
        std::wstring u16str( aCharacters );

        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> convert;
        std::string                                                     utf8 = convert.to_bytes( u16str );

        return utf8;
    }

    std::wstring make_ascii_string( const std::string &utf8 )
    {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> convert;

        std::wstring utf16 = convert.from_bytes( utf8 );

        return utf16;
    }

    CoreCLRHost::CoreCLRHost( std::string const &aDomainName, std::string const &aExePath, std::string const &aCoreRoot,
                              std::string const &aCoreLibraries )
        : mHostPath{ aExePath }
        , mDomainName{ aDomainName }
        , mCoreRoot{ aCoreRoot }
        , mCoreLibraries{ aCoreLibraries }
    {
        // BuildTrustedPlatformAssemblies();
        // TryLoadHostPolicy();

        // std::stringstream lNativeSearchPath;
        // if( !mCoreLibraries.empty() ) lNativeSearchPath << mCoreLibraries << ";";
        // if( !mCoreRoot.empty() ) lNativeSearchPath << mCoreRoot << ";";

        TryLoadCoreCLR();

        mFxrInitialize     = (hostfxr_initialize_for_runtime_config_fn)TryGetExport( "hostfxr_initialize_for_runtime_config" );
        mFxrCreateDelegate = (hostfxr_get_runtime_delegate_fn)TryGetExport( "hostfxr_get_runtime_delegate" );
        mFxrShutdown       = (hostfxr_close_fn)TryGetExport( "hostfxr_close" );
        // mFxrInitialize      = (CoreclrInitialize_ptr)TryGetExport( "coreclr_initialize" );
        // mCoreclrShutdown        = (CoreclrShutdown_ptr)TryGetExport( "coreclr_shutdown_2" );
        // mCoreclrCreateDelegate  = (CoreclrCreateDelegate_ptr)TryGetExport( "coreclr_create_delegate" );
        // mCoreclrExecuteAssembly = (CoreclrExecuteAssembly_ptr)TryGetExport( "coreclr_execute_assembly" );
    }

    void CoreCLRHost::Initialize()
    {
        if( mFxrInitialize == nullptr ) return;

        // const char *lPropertyKeys[]   = { "TRUSTED_PLATFORM_ASSEMBLIES", "APP_PATHS", "NATIVE_DLL_SEARCH_DIRECTORIES" };
        // const char *lPropertyValues[] = { mTrustedPlatformAssemblies.c_str(), mAppPath.c_str(), mNativeDllSearchDirectories.c_str()
        // };
        const std::wstring config_path = L"C:\\GitLab\\SpockEngine\\Programs\\TestCoreCLR\\DotNetLib.runtimeconfig.json";

        int lResult;
        lResult = mFxrInitialize( config_path.c_str(), nullptr, &mCoreCLR );

        if( ( lResult < 0 ) || ( mCoreCLR == nullptr ) )
        {
            Logging::Info( "coreclr_initialize failed - Error: {:#08x}\n", lResult );
            return;
        }

        int rc = mFxrCreateDelegate( mCoreCLR, hdt_load_assembly_and_get_function_pointer,
                                     (void **)&load_assembly_and_get_function_pointer );

        if( ( rc != 0 ) || ( load_assembly_and_get_function_pointer == nullptr ) )
            Logging::Info( "coreclr_initialize failed - Error: {:#08x}\n", rc );

        mFxrShutdown( mCoreCLR );
    }

    void *CoreCLRHost::TryGetExport( const char *symbol )
    {
        if( mNetHostLibraryHandle == nullptr || symbol == nullptr ) return nullptr;

        void *fptr = ::GetProcAddress( mNetHostLibraryHandle, symbol );
        if( fptr == nullptr ) Logging::Info( "Export '{}' not found.\n", symbol );

        return fptr;
    }

    void CoreCLRHost::TryLoadCoreCLR()
    {
        char_t buffer[MAX_PATH];
        size_t buffer_size = sizeof( buffer ) / sizeof( char_t );
        int    rc          = get_hostfxr_path( buffer, &buffer_size, nullptr );
        if( rc != 0 ) return;

        // Load hostfxr and get desired exports
        mNetHostLibraryHandle = ::LoadLibraryW( buffer );

        // return (init_fptr && get_delegate_fptr && close_fptr);

        // std::filesystem::path coreclr_path = std::filesystem::path( mCoreRoot ) / "coreclr.dll";

        // mCoreCLR = ::LoadLibraryExA( coreclr_path.string().c_str(), nullptr, 0 );
        // if( mCoreCLR == nullptr )
        // {
        //     Logging::Info( "Failed to load: '{}'.  - Error: {:#08x}\n", coreclr_path.string(), ::GetLastError() );

        //     return;
        // }

        // HMODULE unused;
        // if( !::GetModuleHandleExA( GET_MODULE_HANDLE_EX_FLAG_PIN, coreclr_path.string().c_str(), &unused ) )
        // {
        //     Logging::Info( "Failed to pin: '{}'.  - Error: {:#08x}\n", coreclr_path.string(), ::GetLastError() );
        // }
    }

    // void CoreCLRHost::TryLoadHostPolicy()
    // {
    //     const char *hostpolicyName = "hostpolicy.dll";
    //     void       *hMod           = (void *)::GetModuleHandleA( hostpolicyName );
    //     if( hMod != nullptr ) return;

    //     std::filesystem::path hostpolicy_path = std::filesystem::path( mCoreRoot ) / "hostpolicy.dll";

    //     mHostPolicy = ::LoadLibraryExA( hostpolicy_path.string().c_str(), nullptr,
    //                                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS );
    //     if( mHostPolicy == nullptr )
    //     {
    //         Logging::Info( "Failed to load: '{}'.  - Error: {:#08x}\n", hostpolicy_path.string(), ::GetLastError() );

    //         return;
    //     }

    //     // // Check if a hostpolicy exists and if it does, load it.
    //     // if( pal::does_file_exist( mock_hostpolicy_value ) )
    //     //     hMod = ( pal::mod_t )::LoadLibraryExW( mock_hostpolicy_value.c_str(), nullptr,
    //     //                                            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS );

    //     // if( hMod == nullptr )
    //     //     pal::fprintf( stderr, W( "Failed to load mock hostpolicy at path '%s'. Error: 0x%08x\n" ),
    //     //     mock_hostpolicy_value.c_str(),
    //     //                   ::GetLastError() );

    //     // return hMod != nullptr;
    // }

    void CoreCLRHost::Shutdown()
    {
        // if( mFxrShutdown == nullptr ) return;

        // int lResult;
        // lResult = mFxrShutdown( mCoreCLR );

        // if( lResult < 0 )
        // {
        //     Logging::Info( "coreclr_shutdown_2 failed - Error: {:#08x}\n", lResult );
        // }
    }

    // static bool EndsWith( std::string const &str, std::string const &suffix )
    // {
    //     if( str.length() < suffix.length() )
    //     {
    //         return false;
    //     }
    //     return str.compare( str.length() - suffix.length(), suffix.length(), suffix ) == 0;
    // }

    // std::string CoreCLRHost::BuildFileList( const std::string &dir, const char *ext, std::function<bool( const char * )> aShouldAdd
    // )
    // {
    //     // assert( ext != nullptr );

    //     std::stringstream lFileList;

    //     for( const auto &lDirEntry : directory_iterator( dir ) )
    //     {
    //         if( std::filesystem::is_directory( lDirEntry.path() ) ) continue;

    //         auto const &lName = lDirEntry.path().filename().string();
    //         std::string ext_local{ ext };

    //         if( !EndsWith( lName, ext_local ) ) continue;

    //         if( aShouldAdd( lName.c_str() ) ) lFileList << lDirEntry.path().string() << ";";
    //     }

    //     return lFileList.str();
    // }

    // void CoreCLRHost::BuildTrustedPlatformAssemblies()
    // {
    //     static const char *const tpa_extensions[] = { ".ni.dll", ".dll", ".ni.exe", ".exe", nullptr };

    //     std::set<std::string> name_set;
    //     std::stringstream     tpa_list;

    //     // Iterate over all extensions.
    //     for( const char *const *curr_ext = tpa_extensions; *curr_ext != nullptr; ++curr_ext )
    //     {
    //         const char  *ext     = *curr_ext;
    //         const size_t ext_len = strlen( ext );

    //         // Iterate over all supplied directories.
    //         for( const std::string &dir : { mCoreLibraries, mCoreRoot } )
    //         {
    //             if( dir.empty() ) continue;

    //             // assert( dir.back() == '\\' );
    //             std::string tmp = BuildFileList( dir, ext,
    //                                              [&]( const char *file )
    //                                              {
    //                                                  std::string file_local{ file };
    //                                                  std::string ext_local{ ext };

    //                                                  if( EndsWith( file_local, ext_local ) )
    //                                                      file_local = file_local.substr( 0, file_local.length() - ext_len );

    //                                                  return name_set.insert( file_local ).second;
    //                                              } );

    //             tpa_list << tmp;
    //         }
    //     }

    //     mTrustedPlatformAssemblies = tpa_list.str();
    // }

    void CoreCLRHost::LoadApplicationAssembly( std::string const &aAssemblyPath, std::string const &aApplicationClass )
    {
        auto lPath = std::filesystem::path( aAssemblyPath );

        mAppPath = lPath.parent_path().string() + "\\";

        std::stringstream lNativePath;
        lNativePath << mAppPath << ";" << mNativeDllSearchDirectories;
        mNativeDllSearchDirectories = lNativePath.str();

        auto lFile = "OlmDevTool"; // lPath.filename();

        Initialize();

        // Configure delegate
        std::wstring lDelegateType0 =
            make_ascii_string( fmt::format( "{}+{}, {}", aApplicationClass, "ConfigureDelegate", "OlmDevTool" ) );
        std::wstring lDelegateType1 = make_ascii_string( fmt::format( "{}+{}, {}", aApplicationClass, "TickDelegate", "OlmDevTool" ) );
        std::wstring lDelegateType3 =
            make_ascii_string( fmt::format( "{}+{}, {}", aApplicationClass, "UpdateDelegate", "OlmDevTool" ) );

        std::wstring lAssemblyPath     = make_ascii_string( aAssemblyPath );
        std::wstring lApplicationClass = make_ascii_string( fmt::format( "{}, {}", aApplicationClass, "OlmDevTool" ) );

        load_assembly_and_get_function_pointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"Configure", lDelegateType0.c_str(), nullptr,
                                                (void **)&mConfigureDelegate );
        if( mConfigureDelegate == nullptr ) mConfigureDelegate = ConfigureDelegateDefault;

        // Update delegate
        load_assembly_and_get_function_pointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"Update", lDelegateType1.c_str(), nullptr,
                                                (void **)&mUpdateDelegate );
        if( mUpdateDelegate == nullptr ) mUpdateDelegate = UpdateDelegateDefault;

        // Update delegate
        load_assembly_and_get_function_pointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"UpdateUI", lDelegateType1.c_str(), nullptr,
                                                (void **)&mUpdateUIDelegate );
        if( mUpdateUIDelegate == nullptr ) mUpdateUIDelegate = UpdateUIDelegateDefault;

        // Update delegate
        load_assembly_and_get_function_pointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"UpdateMenu", lDelegateType3.c_str(), nullptr,
                                                (void **)&mUpdateMenuDelegate );
        if( mUpdateMenuDelegate == nullptr ) mUpdateMenuDelegate = UpdateMenuDelegateDefault;

        // Teardown delegate
        load_assembly_and_get_function_pointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"Teardown", lDelegateType0.c_str(), nullptr,
                                                (void **)&mTeardownDelegate );
        if( mTeardownDelegate == nullptr ) mTeardownDelegate = TeardownDelegateDefault;
    }

    // int CoreCLRHost::Execute( std::string const &aAssemblyPath )
    // {
    //     auto lPath = std::filesystem::path( aAssemblyPath );

    //     mAppPath = lPath.parent_path().string();

    //     std::stringstream lNativePath;
    //     lNativePath << mAppPath << ";" << mNativeDllSearchDirectories;
    //     mNativeDllSearchDirectories = lNativePath.str();

    //     auto lFile = lPath.filename();

    //     Initialize();

    //     int exit_code;

    //     return mCoreclrExecuteAssembly( mHandle, mDomainID, 0, nullptr, aAssemblyPath.c_str(), (uint32_t *)&exit_code );
    // }

    void CoreCLRHost::Configure( std::string aConfigPath )
    {
        char *pszReturn = (char *)::CoTaskMemAlloc( aConfigPath.size() * sizeof( char ) + 1 );
        strcpy( pszReturn, aConfigPath.c_str() );

        mConfigureDelegate( pszReturn );
    }

    void CoreCLRHost::Update( float aTimestamp ) { mUpdateDelegate( aTimestamp ); }
    void CoreCLRHost::UpdateUI( float aTimestamp ) { mUpdateUIDelegate( aTimestamp ); }
    bool CoreCLRHost::UpdateMenu() { return mUpdateMenuDelegate(); }
    void CoreCLRHost::Teardown( std::string aConfigPath )
    {
        char *pszReturn = (char *)::CoTaskMemAlloc( aConfigPath.size() * sizeof( char ) + 1 );
        strcpy( pszReturn, aConfigPath.c_str() );

        mTeardownDelegate( pszReturn );
    }

} // namespace SE::Core