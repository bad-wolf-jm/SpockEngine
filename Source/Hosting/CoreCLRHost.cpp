#include "CoreCLRHost.h"
#include "Core/Logging.h"

namespace SE::Core
{
    void CoreCLRHost::Initialize()
    {
        if( mCoreclrInitialize == nullptr ) return;

        const char *[] lPropertyKeys   = { "TRUSTED_PLATFORM_ASSEMBLIES", "APP_PATHS", "NATIVE_DLL_SEARCH_DIRECTORIES" };
        const char *[] lPropertyValues = { GetTrustedPoatformAssemblies(), GetAppPaths(), GetNativeDllSearchDirectories() };

        HRESULT lResult;
        lResult = mCoreclrInitialize( mHostPath.c_str(), mDomainName.c_str(), sizeof( lPropertyKeys ) / sizeof( char * ),
                                      lPropertyKeys, lPropertyValues, &mHandle, &mDomainID );

        if( FAILED( lResult ) )
        {
            Logging:Info( "coreclr_initialize failed - Error: {:#08x}\n" , lResult );
        }
    }

    void CoreCLRHost::Shutdown()
    {
        if( mCoreclrInitialize == nullptr ) return;

        HRESULT lResult;
        lResult = mCoreclrShutdown( mHandle, mDomainID, &mLatchedExitCode );

        if( FAILED( lResult ) )
        {
            Logging:Info( "coreclr_shutdown_2 failed - Error: {:#08x}\n" , lResult );
        }
    }

} // namespace SE::Core