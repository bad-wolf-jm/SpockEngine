#include "ScriptingEngine.h"

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
#include "SensorModel.h"
#include "TensorOps/ScalarTypes.h"


namespace LTSE::Core
{
    using namespace math;
    using namespace sol;

    ScriptingEngine::ScriptingEngine()
    {
        ScriptState.open_libraries( lib::base );

        Initialize();
    }

    void ScriptingEngine::Initialize()
    {
        mTypesModule = ScriptState["dtypes"].get_or_create<sol::table>();

        // clang-format off
        DeclarePrimitiveType<uint8_t>  ( mTypesModule, "uint8"   );
        DeclarePrimitiveType<uint16_t> ( mTypesModule, "uint16"  );
        DeclarePrimitiveType<uint32_t> ( mTypesModule, "uint32"  );
        DeclarePrimitiveType<uint64_t> ( mTypesModule, "uint64"  );
        DeclarePrimitiveType<int8_t>   ( mTypesModule, "int8"    );
        DeclarePrimitiveType<int16_t>  ( mTypesModule, "int16"   );
        DeclarePrimitiveType<int32_t>  ( mTypesModule, "int32"   );
        DeclarePrimitiveType<int64_t>  ( mTypesModule, "int64"   );
        DeclarePrimitiveType<float>    ( mTypesModule, "float32" );
        DeclarePrimitiveType<double>   ( mTypesModule, "float64" );
        // clang-format on

        // clang-format off
        ScriptState.new_enum( "types",
            "float32", TensorOps::eScalarType::FLOAT32,
            "float64", TensorOps::eScalarType::FLOAT64,
            "uint8",   TensorOps::eScalarType::UINT8,
            "uint16",  TensorOps::eScalarType::UINT16,
            "uint32",  TensorOps::eScalarType::UINT32,
            "uint64",  TensorOps::eScalarType::UINT64,
            "int8",    TensorOps::eScalarType::INT8,
            "int16",   TensorOps::eScalarType::INT16,
            "int32",   TensorOps::eScalarType::INT32,
            "int64",   TensorOps::eScalarType::INT64  );
        // clang-format on

        auto lMathModule       = ScriptState["Math"].get_or_create<sol::table>();
        lMathModule["radians"] = []( float aDegrees ) { return radians( aDegrees ); };
        lMathModule["degrees"] = []( float aRadians ) { return degrees( aRadians ); };

        DefineVectorTypes( lMathModule );
        DefineMatrixTypes( lMathModule );

        auto lEntityRegistryModule = ScriptState["EntityRegistry"].get_or_create<sol::table>();
        OpenEntityRegistry( lEntityRegistryModule );
        auto lRelationshipComponent = DeclarePrimitiveType<sRelationshipComponent>( lEntityRegistryModule, "sRelationshipComponent" );
        lRelationshipComponent["children"] = readonly(&sRelationshipComponent::mChildren);
        lRelationshipComponent["parent"] = readonly(&sRelationshipComponent::mParent);
        auto lTagComponent = DeclarePrimitiveType<sTag>( lEntityRegistryModule, "sTag" );
        lTagComponent["value"] = readonly(&sTag::mValue);

        auto lCudaModule = ScriptState["Cuda"].get_or_create<sol::table>();
        OpenTensorLibrary( lCudaModule );
        RequireCudaTexture( lCudaModule );

        auto lCoreModule = ScriptState["Core"].get_or_create<sol::table>();
        OpenCoreLibrary( lCoreModule );
        DefineArrayTypes( lCoreModule );

        auto lSensorModule = ScriptState["Sensor"].get_or_create<sol::table>();
        OpenSensorModelLibrary( lSensorModule );
    }

    ScriptEnvironment ScriptingEngine::LoadFile( fs::path aPath )
    {
        ScriptEnvironment lNewEnvironment = NewEnvironment();

        ScriptState.script_file( aPath.string(), lNewEnvironment, load_mode::any );

        return lNewEnvironment;
    }

    ScriptEnvironment ScriptingEngine::NewEnvironment()
    {
        ScriptEnvironment lNewEnvironment( ScriptState, create, ScriptState.globals() );

        return lNewEnvironment;
    }

    void ScriptingEngine::Execute( std::string aString ) { ScriptState.script( aString ); }
    void ScriptingEngine::Execute( ScriptEnvironment &aEnvironment, std::string aString ) { ScriptState.script( aString, aEnvironment ); }

} // namespace LTSE::Core