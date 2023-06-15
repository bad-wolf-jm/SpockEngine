#pragma once

#ifndef CDECL
#    define CDECL __cdecl
#endif

#include "coreclr_delegates.h"
#include "hostfxr.h"

#include <Windows.h>
#include <functional>
#include <string>

namespace SE::Core
{

#define CORECLR_APPLICATION_API( function, ... )               \
    typedef void( __stdcall * function##_ptr )( __VA_ARGS__ ); \
    function##_ptr m##function = nullptr;                      \
    static void __stdcall function##Default( __VA_ARGS__ ) {}

#define CORECLR_APPLICATION_NON_VOID_API( function, return_type, ... ) \
    typedef return_type( __stdcall *function##_ptr )( __VA_ARGS__ );   \
    function##_ptr m##function = nullptr;                              \
    static return_type __stdcall function##Default( __VA_ARGS__ ) { return return_type{}; }

    class CoreCLRHost
    {
      public:
        CoreCLRHost();

        void LoadApplicationAssembly( string_t const &aAssemblyPath, string_t const &aApplicationClass );
        void Initialize();
        void Shutdown();
        int  Execute( string_t const &aAssemblyPath );

        void Configure( string_t aConfigPath );
        void Update( float aTimestamp );
        void UpdateUI( float aTimestamp );
        bool UpdateMenu();
        void Teardown( string_t aConfigPath );

      private:
        hostfxr_initialize_for_runtime_config_fn mFxrInitialize     = nullptr;
        hostfxr_get_runtime_delegate_fn          mFxrCreateDelegate = nullptr;
        hostfxr_close_fn                         mFxrShutdown       = nullptr;

        load_assembly_and_get_function_pointer_fn GetFunctionPointer = nullptr;

        void  TryLoadCoreCLR();
        void *TryGetExport( const char *aName );

      private:
        hostfxr_handle mCoreCLR              = nullptr;
        HMODULE        mNetHostLibraryHandle = nullptr;

      private:
        CORECLR_APPLICATION_API( ConfigureDelegate, const char *aConfigPath );
        CORECLR_APPLICATION_API( UpdateDelegate, float aTimestamp );
        CORECLR_APPLICATION_API( UpdateUIDelegate, float aTimestamp );
        CORECLR_APPLICATION_NON_VOID_API( UpdateMenuDelegate, bool );
        CORECLR_APPLICATION_API( TeardownDelegate, const char *aConfigPath );
    };
} // namespace SE::Core