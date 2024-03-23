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

        auto lMathModule       = _scriptState["Math"].get_or_create<sol::table>();
        lMathModule["radians"] = []( float aDegrees ) { return radians( aDegrees ); };
        lMathModule["degrees"] = []( float aRadians ) { return degrees( aRadians ); };

        define_vector_types( lMathModule );
        define_matrix_types( lMathModule );

        open_entity_registry_library( _scriptState );

        auto lCudaModule = _scriptState["Cuda"].get_or_create<sol::table>();
        open_tensor_library( lCudaModule );
        require_cuda_texture( lCudaModule );

        auto lCoreModule = _scriptState["Core"].get_or_create<sol::table>();
        open_core_library( lCoreModule );
        define_array_types( lCoreModule );
    }

    environment_t script_bindings::LoadFile( fs::path aPath )
    {
        environment_t lNewEnvironment = NewEnvironment();

        _scriptState.script_file( aPath.string(), lNewEnvironment, load_mode::any );

        return lNewEnvironment;
    }

    environment_t script_bindings::NewEnvironment()
    {
        environment_t lNewEnvironment( _scriptState, create, _scriptState.globals() );

        return lNewEnvironment;
    }

    void script_bindings::Execute( std::string aString )
    {
        _scriptState.script( aString );
    }

    void script_bindings::Execute( environment_t &aEnvironment, std::string aString )
    {
        _scriptState.script( aString, aEnvironment );
    }

} // namespace SE::Core