#pragma once

#define CDECL __cdecl
#include <Windows.h>
#include <functional>
#include <string>

namespace SE::Core
{

#define CORECLR_HOSTING_API_0( function, ... )                \
    typedef int( __stdcall * function##_ptr )( __VA_ARGS__ ); \
    function##_ptr m##function = nullptr;

#define CORECLR_APPLICATION_API( function, ... )              \
    typedef int( __stdcall * function##_ptr )( __VA_ARGS__ ); \
    function##_ptr m##function = nullptr;

    class CoreCLRHost
    {
      public:
        CoreCLRHost( std::string const &aAssemblyName, std::string const &aExePath, std::string const &aCoreRoot,
                     std::string const &aCoreLibraries );

        void LoadApplicationAssembly( std::string const &aAssemblyPath, std::string const &aApplicationClass );
        void Initialize();
        void Shutdown();
        int Execute(std::string const &aAssemblyPath);

      private:
        CORECLR_HOSTING_API_0( CoreclrInitialize, const char *aExePath, const char *aAppDomainFriendlyName, int aPropertyCount,
                               const char **aPropertyKeys, const char **aPropertyValues, void **aHostHandle, unsigned int *aDomainId );

        CORECLR_HOSTING_API_0( CoreclrShutdown, void *aHostHandle, unsigned int aDomainId, int *latchedExitCode );

        CORECLR_HOSTING_API_0( CoreclrCreateDelegate, void *aHostHandle, unsigned int aDomainId, const char *aEntryPointAssemblyName,
                               const char *aEntryPointTypeName, const char *aEntryPointMethodName, void **aDelegate );

        CORECLR_HOSTING_API_0( CoreclrExecuteAssembly, void *hostHandle, unsigned int domainId, int argc, const char **argv,
                               const char *managedAssemblyPath, unsigned int *exitCode );

        void        BuildTrustedPlatformAssemblies();
        std::string BuildFileList( const std::string &dir, const char *ext, std::function<bool( const char * )> aShouldAdd );
        void        TryLoadCoreCLR();
        void       *TryGetExport( const char *aName );


      private:
        HMODULE  mCoreCLR         = nullptr;
        void    *mHandle          = nullptr;
        uint32_t mDomainID        = -1;
        int      mLatchedExitCode = 0;

        std::string mHostPath      = "";
        std::string mCoreRoot      = "";
        std::string mCoreLibraries = "";
        std::string mDomainName    = "";

      private:
        std::string mAppPath;
        std::string mTrustedPlatformAssemblies;
        std::string mNativeDllSearchDirectories;

        CORECLR_APPLICATION_API( ConfigureDelegate );
        CORECLR_APPLICATION_API( UpdateDelegatate, float aTimestamp );
        CORECLR_APPLICATION_API( TeardownDelegatate );
    };
} // namespace SE::Core