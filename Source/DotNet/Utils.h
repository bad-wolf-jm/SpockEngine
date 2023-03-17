#pragma once

#include <filesystem>
#include "Typedefs.h"
#include "Class.h"

namespace SE::Core::Mono::Utils
{
    char *ReadBytes( const std::filesystem::path &aFilepath, uint32_t *aOutSize );
    MonoAssembly *LoadMonoAssembly( const std::filesystem::path &lAssemblyPath );
    void PrintAssemblyTypes( MonoAssembly *aAssembly );
    eScriptFieldType MonoTypeToScriptFieldType( MonoType *aMonoType );
} // namespace Mono::Utils
