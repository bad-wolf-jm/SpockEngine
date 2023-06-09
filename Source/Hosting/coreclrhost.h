#pragma once

namespace SE::Core
{

#define CORECLR_CALLING_CONVENTION __stdcall

#define CORECLR_HOSTING_API( function, ... )                                   \
    typedef int( CORECLR_CALLING_CONVENTION * function##_ptr )( __VA_ARGS__ ); \
    function##_ptr m##function = nullptr;

    class CoreCLRHost
    {
      public:
        CoreCLRHost( fs::path cosnt &aCoreRoot );

        void Initialize();
        void Shutdown();

      private:
        CORECLR_HOSTING_API( CoreclrInitialize, const char *aExePath, const char *aAppDomainFriendlyName, int aPropertyCount,
                             const char **aPropertyKeys, const char **aPropertyValues, void **aHostHandle, unsigned int *aDomainId );

        CORECLR_HOSTING_API( CoreclrShutdown, void *aHostHandle, unsigned int aDomainId, int *latchedExitCode );

        CORECLR_HOSTING_API( CoreclrCreateDelegate, void *aHostHandle, unsigned int aDomainId, const char *aEntryPointAssemblyName,
                             const char *aEntryPointTypeName, const char *aEntryPointMethodName, void **aDelegate );

      private:
        void *mHandle          = nullptr;
        int   mDomainID        = -1;
        int   mLatchedExitCode = 0;

        std::string mHostPath = "CoreCLRHost";
        std::string mDomainName = "CoreCLRHost";
    }
} // namespace SE::Core