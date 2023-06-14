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
        TryLoadCoreCLR();

        mFxrInitialize     = (hostfxr_initialize_for_runtime_config_fn)TryGetExport( "hostfxr_initialize_for_runtime_config" );
        mFxrCreateDelegate = (hostfxr_get_runtime_delegate_fn)TryGetExport( "hostfxr_get_runtime_delegate" );
        mFxrShutdown       = (hostfxr_close_fn)TryGetExport( "hostfxr_close" );
    }

    void CoreCLRHost::Initialize()
    {
        if( mFxrInitialize == nullptr ) return;

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
    }

    void CoreCLRHost::Shutdown()
    {
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
        std::wstring lDelegateType0 =
            make_ascii_string( fmt::format( "{}+{}, {}", aApplicationClass, "ConfigureDelegate", "OlmDevTool" ) );
        std::wstring lDelegateType1 = make_ascii_string( fmt::format( "{}+{}, {}", aApplicationClass, "TickDelegate", "OlmDevTool" ) );
        std::wstring lDelegateType3 =
            make_ascii_string( fmt::format( "{}+{}, {}", aApplicationClass, "UpdateDelegate", "OlmDevTool" ) );

        std::wstring lAssemblyPath     = make_ascii_string( aAssemblyPath );
        std::wstring lApplicationClass = make_ascii_string( fmt::format( "{}, {}", aApplicationClass, "OlmDevTool" ) );

        load_assembly_and_get_function_pointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"Configure", lDelegateType0.c_str(),
                                                nullptr, (void **)&mConfigureDelegate );
        if( mConfigureDelegate == nullptr ) mConfigureDelegate = ConfigureDelegateDefault;

        // Update delegate
        load_assembly_and_get_function_pointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"Update", lDelegateType1.c_str(),
                                                nullptr, (void **)&mUpdateDelegate );
        if( mUpdateDelegate == nullptr ) mUpdateDelegate = UpdateDelegateDefault;

        // Update delegate
        load_assembly_and_get_function_pointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"UpdateUI", lDelegateType1.c_str(),
                                                nullptr, (void **)&mUpdateUIDelegate );
        if( mUpdateUIDelegate == nullptr ) mUpdateUIDelegate = UpdateUIDelegateDefault;

        // Update delegate
        load_assembly_and_get_function_pointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"UpdateMenu",
                                                lDelegateType3.c_str(), nullptr, (void **)&mUpdateMenuDelegate );
        if( mUpdateMenuDelegate == nullptr ) mUpdateMenuDelegate = UpdateMenuDelegateDefault;

        // Teardown delegate
        load_assembly_and_get_function_pointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"Teardown", lDelegateType0.c_str(),
                                                nullptr, (void **)&mTeardownDelegate );
        if( mTeardownDelegate == nullptr ) mTeardownDelegate = TeardownDelegateDefault;
    }

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