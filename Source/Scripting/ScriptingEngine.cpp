#include "ScriptingEngine.h"

#include "Core/Definitions.h"
#include "Core/Logging.h"
#include "Core/Math/Types.h"
#include "Cuda/Tensor.h"
// #include "Math/Module.h"
#include "Math/MatrixTypes.h"
#include "Math/VectorTypes.h"
#include <type_traits>

#include "ArrayTypes.h"
#include "Scripting/Core/Texture.h"
#include "Scripting/Cuda/Texture.h"
#include "TensorOps/ScalarTypes.h"

namespace SE::Core
{
    using namespace math;
    using namespace sol;

    script_bindings::script_bindings()
    {
        _scriptState.open_libraries( lib::base );
        Initialize();
    }

    void script_bindings::Initialize()
    {
        _typesModule = _scriptState["dtypes"].get_or_create<sol::table>();

        // clang-format off
        declare_primitive_type<uint8_t>  ( _typesModule, "uint8"   );
        declare_primitive_type<uint16_t> ( _typesModule, "uint16"  );
        declare_primitive_type<uint32_t> ( _typesModule, "uint32"  );
        declare_primitive_type<uint64_t> ( _typesModule, "uint64"  );
        declare_primitive_type<int8_t>   ( _typesModule, "int8"    );
        declare_primitive_type<int16_t>  ( _typesModule, "int16"   );
        declare_primitive_type<int32_t>  ( _typesModule, "int32"   );
        declare_primitive_type<int64_t>  ( _typesModule, "int64"   );
        declare_primitive_type<float>    ( _typesModule, "float32" );
        declare_primitive_type<double>   ( _typesModule, "float64" );
        // clang-format on

        // clang-format off
        _scriptState.new_enum( "types",
            "float32", scalar_type_t::FLOAT32,
            "float64", scalar_type_t::FLOAT64,
            "uint8",   scalar_type_t::UINT8,
            "uint16",  scalar_type_t::UINT16,
            "uint32",  scalar_type_t::UINT32,
            "uint64",  scalar_type_t::UINT64,
            "int8",    scalar_type_t::INT8,
            "int16",   scalar_type_t::INT16,
            "int32",   scalar_type_t::INT32,
            "int64",   scalar_type_t::INT64  );
        // clang-format on

        auto mathModule       = _scriptState["Math"].get_or_create<sol::table>();
        mathModule["radians"] = []( float degrees ) { return radians( degrees ); };
        mathModule["degrees"] = []( float radians ) { return degrees( radians ); };

        define_vector_types( mathModule );
        define_matrix_types( mathModule );

        open_entity_registry_library( _scriptState );

        auto cudaModule = _scriptState["Cuda"].get_or_create<sol::table>();
        open_tensor_library( cudaModule );
        require_cuda_texture( cudaModule );

        auto coreModule = _scriptState["Core"].get_or_create<sol::table>();
        open_core_library( coreModule );
        define_array_types( coreModule );
    }

    environment_t script_bindings::LoadFile( fs::path path )
    {
        environment_t newEnvironment = NewEnvironment();

        _scriptState.script_file( path.string(), newEnvironment, load_mode::any );

        return newEnvironment;
    }

    environment_t script_bindings::NewEnvironment()
    {
        environment_t newEnvironment( _scriptState, create, _scriptState.globals() );

        return newEnvironment;
    }

    void script_bindings::Execute( std::string string )
    {
        _scriptState.script( string );
    }

    void script_bindings::Execute( environment_t &environment, std::string string )
    {
        _scriptState.script( string, environment );
    }

} // namespace SE::Core