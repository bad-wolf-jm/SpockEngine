#include "CoreCLRHost.h"
#include "Core/Logging.h"
#include <filesystem>
#include <set>
#include <sstream>

namespace SE::Core
{
    using directory_iterator = std::filesystem::directory_iterator;

    CoreCLRHost::CoreCLRHost( std::string const &aDomainName, std::string const &aExePath, std::string const &aCoreRoot,
                              std::string const &aCoreLibraries )
        : mHostPath{ aExePath }
        , mDomainName{ aDomainName }
        , mCoreRoot{ aCoreRoot }
        , mCoreLibraries{ aCoreLibraries }
    {
        BuildTrustedPlatformAssemblies();

        std::stringstream lNativeSearchPath;
        if( !mCoreLibraries.empty() ) lNativeSearchPath << mCoreLibraries << ";";
        if( !mCoreRoot.empty() ) lNativeSearchPath << mCoreRoot << ";";

        TryLoadCoreCLR();

        mCoreclrInitialize      = (CoreclrInitialize_ptr)TryGetExport( "coreclr_initialize" );
        mCoreclrShutdown        = (CoreclrShutdown_ptr)TryGetExport( "coreclr_shutdown_2" );
        mCoreclrCreateDelegate  = (CoreclrCreateDelegate_ptr)TryGetExport( "coreclr_create_delegate" );
        mCoreclrExecuteAssembly = (CoreclrExecuteAssembly_ptr)TryGetExport( "coreclr_execute_assembly" );
    }

    void CoreCLRHost::Initialize()
    {
        if( mCoreclrInitialize == nullptr ) return;

        const char *lPropertyKeys[]   = { "TRUSTED_PLATFORM_ASSEMBLIES", "APP_PATHS", "NATIVE_DLL_SEARCH_DIRECTORIES" };
        const char *lPropertyValues[] = { mTrustedPlatformAssemblies.c_str(), mAppPath.c_str(), mNativeDllSearchDirectories.c_str() };

        int lResult;
        lResult = mCoreclrInitialize( mHostPath.c_str(), mDomainName.c_str(), sizeof( lPropertyKeys ) / sizeof( char * ),
                                      lPropertyKeys, lPropertyValues, &mHandle, &mDomainID );

        if( lResult < 0 )
        {
            Logging::Info( "coreclr_initialize failed - Error: {:#08x}\n", lResult );
            return;
        }
    }

    void *CoreCLRHost::TryGetExport( const char *symbol )
    {
        if( mCoreCLR == nullptr || symbol == nullptr ) return nullptr;

        void *fptr = ::GetProcAddress( mCoreCLR, symbol );
        if( fptr == nullptr ) Logging::Info( "Export '{}' not found.\n", symbol );

        return fptr;
    }

    void CoreCLRHost::TryLoadCoreCLR()
    {
        std::filesystem::path coreclr_path = std::filesystem::path( mCoreRoot ) / "coreclr.dll";

        mCoreCLR = ::LoadLibraryExA( coreclr_path.string().c_str(), nullptr, 0 );
        if( mCoreCLR == nullptr )
        {
            Logging::Info( "Failed to load: '{}'.  - Error: {:#08x}\n", coreclr_path.string(), ::GetLastError() );

            return;
        }

        HMODULE unused;
        if( !::GetModuleHandleExA( GET_MODULE_HANDLE_EX_FLAG_PIN, coreclr_path.string().c_str(), &unused ) )
        {
            Logging::Info( "Failed to pin: '{}'.  - Error: {:#08x}\n", coreclr_path.string(), ::GetLastError() );
        }
    }

    void CoreCLRHost::Shutdown()
    {
        if( mCoreclrInitialize == nullptr ) return;

        int lResult;
        lResult = mCoreclrShutdown( mHandle, mDomainID, &mLatchedExitCode );

        if( lResult < 0 )
        {
            Logging::Info( "coreclr_shutdown_2 failed - Error: {:#08x}\n", lResult );
        }
    }

    static bool EndsWith( std::string const &str, std::string const &suffix )
    {
        if( str.length() < suffix.length() )
        {
            return false;
        }
        return str.compare( str.length() - suffix.length(), suffix.length(), suffix ) == 0;
    }

    std::string CoreCLRHost::BuildFileList( const std::string &dir, const char *ext, std::function<bool( const char * )> aShouldAdd )
    {
        // assert( ext != nullptr );

        std::stringstream lFileList;

        for( const auto &lDirEntry : directory_iterator( dir ) )
        {
            if( std::filesystem::is_directory( lDirEntry.path() ) ) continue;

            auto const &lName = lDirEntry.path().filename().string();
            std::string ext_local{ ext };

            if( !EndsWith( lName, ext_local ) ) continue;

            if( aShouldAdd( lName.c_str() ) ) lFileList << lDirEntry.path().string() << ";";
        }

        return lFileList.str();
    }

    void CoreCLRHost::BuildTrustedPlatformAssemblies()
    {
        static const char *const tpa_extensions[] = { ".ni.dll", ".dll", ".ni.exe", ".exe", nullptr };

        std::set<std::string> name_set;
        std::stringstream     tpa_list;

        // Iterate over all extensions.
        for( const char *const *curr_ext = tpa_extensions; *curr_ext != nullptr; ++curr_ext )
        {
            const char  *ext     = *curr_ext;
            const size_t ext_len = strlen( ext );

            // Iterate over all supplied directories.
            for( const std::string &dir : { mCoreLibraries, mCoreRoot } )
            {
                if( dir.empty() ) continue;

                // assert( dir.back() == '\\' );
                std::string tmp = BuildFileList( dir, ext,
                                                 [&]( const char *file )
                                                 {
                                                     std::string file_local{ file };
                                                     std::string ext_local{ ext };

                                                     if( EndsWith( file_local, ext_local ) )
                                                         file_local = file_local.substr( 0, file_local.length() - ext_len );

                                                     return name_set.insert( file_local ).second;
                                                 } );

                tpa_list << tmp;
            }
        }

        mTrustedPlatformAssemblies = tpa_list.str();
    }

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
        mCoreclrCreateDelegate( mHandle, mDomainID, lFile, aApplicationClass.c_str(), "Configure", (void **)&mConfigureDelegate );
        if( mConfigureDelegate == nullptr ) mConfigureDelegate = ConfigureDelegateDefault;

        // Update delegate
        mCoreclrCreateDelegate( mHandle, mDomainID, lFile, aApplicationClass.c_str(), "Update", (void **)&mUpdateDelegate );
        if( mUpdateDelegate == nullptr ) mUpdateDelegate = UpdateDelegateDefault;

        // Update delegate
        mCoreclrCreateDelegate( mHandle, mDomainID, lFile, aApplicationClass.c_str(), "UpdateUI", (void **)&mUpdateUIDelegate );
        if( mUpdateUIDelegate == nullptr ) mUpdateUIDelegate = UpdateUIDelegateDefault;

        // Update delegate
        mCoreclrCreateDelegate( mHandle, mDomainID, lFile, aApplicationClass.c_str(), "UpdateMenu", (void **)&mUpdateMenuDelegate );
        if( mUpdateMenuDelegate == nullptr ) mUpdateMenuDelegate = UpdateMenuDelegateDefault;

        // Teardown delegate
        mCoreclrCreateDelegate( mHandle, mDomainID, lFile, aApplicationClass.c_str(), "Teardown", (void **)&mTeardownDelegate );
        if( mTeardownDelegate == nullptr ) mTeardownDelegate = TeardownDelegateDefault;
    }

    int CoreCLRHost::Execute( std::string const &aAssemblyPath )
    {
        auto lPath = std::filesystem::path( aAssemblyPath );

        mAppPath = lPath.parent_path().string();

        std::stringstream lNativePath;
        lNativePath << mAppPath << ";" << mNativeDllSearchDirectories;
        mNativeDllSearchDirectories = lNativePath.str();

        auto lFile = lPath.filename();

        Initialize();

        int exit_code;

        return mCoreclrExecuteAssembly( mHandle, mDomainID, 0, nullptr, aAssemblyPath.c_str(), (uint32_t *)&exit_code );
    }

    void CoreCLRHost::Configure( std::string aConfigPath ) { mConfigureDelegate(); }
    void CoreCLRHost::Update( float aTimestamp ) { mUpdateDelegate( aTimestamp ); }
    void CoreCLRHost::UpdateUI( float aTimestamp ) { mUpdateUIDelegate( aTimestamp ); }
    bool CoreCLRHost::UpdateMenu() { return mUpdateMenuDelegate(); }
    void CoreCLRHost::Teardown( std::string aConfigPath ) { mTeardownDelegate(); }

} // namespace SE::Core