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

    static string_t make_ascii_string( wchar_t *aCharacters )
    {
        std::wstring u16str( aCharacters );

        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> convert;
        string_t                                                        utf8 = convert.to_bytes( u16str );

        return utf8;
    }

    std::wstring make_ascii_string( const string_t &utf8 )
    {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> convert;

        std::wstring utf16 = convert.from_bytes( utf8 );

        return utf16;
    }

    CoreCLRHost::CoreCLRHost()
    {
        TryLoadCoreCLR();

        mFxrInitialize = (Internal::hostfxr_initialize_for_runtime_config_fn)TryGetExport( "hostfxr_initialize_for_runtime_config" );
        mFxrSetRuntimePropertyValue =
            (Internal::hostfxr_set_runtime_property_value_fn)TryGetExport( "hostfxr_set_runtime_property_value" );
        mFxrCreateDelegate = (Internal::hostfxr_get_runtime_delegate_fn)TryGetExport( "hostfxr_get_runtime_delegate" );
        mFxrShutdown       = (Internal::hostfxr_close_fn)TryGetExport( "hostfxr_close" );
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

        int rc = mFxrCreateDelegate( mCoreCLR, Internal::hdt_load_assembly_and_get_function_pointer, (void **)&mGetFunctionPointer );

        if( ( rc != 0 ) || ( mGetFunctionPointer == nullptr ) ) Logging::Info( "coreclr_initialize failed - Error: {:#08x}\n", rc );

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
        wchar_t buffer[MAX_PATH];
        size_t  buffer_size = sizeof( buffer ) / sizeof( wchar_t );
        int     rc          = get_hostfxr_path( buffer, &buffer_size, nullptr );
        if( rc != 0 ) return;

        // Load hostfxr and get desired exports
        mNetHostLibraryHandle = ::LoadLibraryW( buffer );
    }

    void CoreCLRHost::Shutdown() {}

    void CoreCLRHost::LoadApplicationAssembly( string_t const &aAssemblyPath, string_t const &aApplicationName )
    {
        Initialize();

        auto lApplicationClassName = fmt::format( "{}.{}", aApplicationName, aApplicationName );

        std::wstring lAssemblyPath     = make_ascii_string( aAssemblyPath );
        std::wstring lApplicationClass = make_ascii_string( fmt::format( "{}, {}", lApplicationClassName, aApplicationName ) );

        std::wstring lConfigureDelegate =
            make_ascii_string( fmt::format( "{}+{}, {}", lApplicationClassName, "ConfigureDelegate", aApplicationName ) );
        mGetFunctionPointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"Configure", lConfigureDelegate.c_str(), nullptr,
                             (void **)&mConfigureDelegate );
        if( mConfigureDelegate == nullptr ) mConfigureDelegate = ConfigureDelegateDefault;
        mGetFunctionPointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"Teardown", lConfigureDelegate.c_str(), nullptr,
                             (void **)&mTeardownDelegate );
        if( mTeardownDelegate == nullptr ) mTeardownDelegate = TeardownDelegateDefault;

        std::wstring lTickDelegate =
            make_ascii_string( fmt::format( "{}+{}, {}", lApplicationClassName, "TickDelegate", aApplicationName ) );
        mGetFunctionPointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"Update", lTickDelegate.c_str(), nullptr,
                             (void **)&mUpdateDelegate );
        if( mUpdateDelegate == nullptr ) mUpdateDelegate = UpdateDelegateDefault;
        mGetFunctionPointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"UpdateUI", lTickDelegate.c_str(), nullptr,
                             (void **)&mUpdateUIDelegate );
        if( mUpdateUIDelegate == nullptr ) mUpdateUIDelegate = UpdateUIDelegateDefault;

        std::wstring lUpdateMenuDelegate =
            make_ascii_string( fmt::format( "{}+{}, {}", lApplicationClassName, "UpdateDelegate", aApplicationName ) );
        mGetFunctionPointer( lAssemblyPath.c_str(), lApplicationClass.c_str(), L"UpdateMenu", lUpdateMenuDelegate.c_str(), nullptr,
                             (void **)&mUpdateMenuDelegate );
        if( mUpdateMenuDelegate == nullptr ) mUpdateMenuDelegate = UpdateMenuDelegateDefault;
    }

    void CoreCLRHost::Configure( string_t aConfigPath )
    {
        char *pszReturn = (char *)::CoTaskMemAlloc( aConfigPath.size() * sizeof( char ) + 1 );
        strcpy( pszReturn, aConfigPath.c_str() );

        mConfigureDelegate( pszReturn );
    }

    void CoreCLRHost::Update( float aTimestamp ) { mUpdateDelegate( aTimestamp ); }
    void CoreCLRHost::UpdateUI( float aTimestamp ) { mUpdateUIDelegate( aTimestamp ); }
    bool CoreCLRHost::UpdateMenu() { return mUpdateMenuDelegate(); }
    void CoreCLRHost::Teardown( string_t aConfigPath )
    {
        char *pszReturn = (char *)::CoTaskMemAlloc( aConfigPath.size() * sizeof( char ) + 1 );
        strcpy( pszReturn, aConfigPath.c_str() );

        mTeardownDelegate( pszReturn );
    }

} // namespace SE::Core