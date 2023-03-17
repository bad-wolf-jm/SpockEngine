#include "InternalCalls.h"
#include "EntityRegistry.h"
#include "TypeReflection.h"
#include <iostream>
#include <string>
#include <list>

#include "Core/Logging.h"

#include "UI/UI.h"
#include "UI/Widgets.h"

#include "MonoRuntime.h"
#include "MonoScriptUtils.h"

namespace SE::MonoInternalCalls
{
    // std::list<std::string> sConsoleLines;

    // std::list<std::string> &Console_GetLines( )
    // {
    //     return sConsoleLines;
    // }

    // void Console_Write( MonoString *aBuffer )
    // {
    //     auto lString = MonoRuntime::NewString(aBuffer);


    //     SE::Logging::Info("{}", MonoRuntime::NewString(aBuffer));
    // }

    uint32_t Entity_Create( EntityCollection *aRegistry, MonoString *aName, uint32_t aEntityID )
    {
        auto lName      = std::string( mono_string_to_utf8( aName ) );
        auto lNewEntity = aRegistry->CreateEntity( aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ), lName );

        return static_cast<uint32_t>( lNewEntity );
    }

    bool Entity_IsValid( uint32_t aEntityID, EntityCollection *aRegistry )
    {
        return aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ).IsValid();
    }

    bool Entity_Has( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny =
            Core::InvokeMetaFunction( lMetaType, "Has"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ) );

        return lMaybeAny.cast<bool>();
    }

    MonoObject *Entity_Get( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny = Core::InvokeMetaFunction(
            lMetaType, "Get"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ), DotNetClass( lMonoType ) );

        return lMaybeAny.cast<DotNetInstance>().GetInstance();
    }

    void Entity_Replace( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent )
    {
        MonoType  *lMonoType  = mono_reflection_type_get_type( aComponentType );
        MonoClass *lMonoClass = mono_class_from_mono_type( lMonoType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        Core::InvokeMetaFunction( lMetaType, "Replace"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ),
                                  DotNetInstance( lMonoClass, aNewComponent ) );
    }

    void Entity_Add( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType, MonoObject *aNewComponent )
    {
        MonoType  *lMonoType  = mono_reflection_type_get_type( aComponentType );
        MonoClass *lMonoClass = mono_class_from_mono_type( lMonoType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        Core::InvokeMetaFunction( lMetaType, "Add"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ),
                                  DotNetInstance( lMonoClass, aNewComponent ) );
    }

    void Entity_Remove( uint32_t aEntityID, EntityCollection *aRegistry, MonoReflectionType *aComponentType )
    {
        MonoType *lMonoType = mono_reflection_type_get_type( aComponentType );

        const entt::meta_type lMetaType = Core::GetMetaType( lMonoType );
        const entt::meta_any  lMaybeAny =
            Core::InvokeMetaFunction( lMetaType, "Remove"_hs, aRegistry->WrapEntity( static_cast<entt::entity>( aEntityID ) ) );
    }

    size_t OpNode_NewTensorShape( MonoArray *aShape, uint32_t aRank, uint32_t aLayers, uint32_t aElementSize )
    {
        std::vector<std::vector<uint32_t>> lShape{};

        for( uint32_t i = 0; i < aLayers; i++ )
        {
            lShape.push_back( {} );
            for( uint32_t j = 0; j < aRank; j++ )
            {
                uint32_t lDim = *( mono_array_addr( aShape, uint32_t, ( i * aRank + j ) ) );

                lShape.back().push_back( lDim );
            }
        }

        return (size_t)( new Cuda::sTensorShape( lShape, aElementSize ) );
    }

    void OpNode_DestroyTensorShape( Cuda::sTensorShape *aTensorShape ) { delete aTensorShape; }

    uint32_t OpNode_CountLayers( Cuda::sTensorShape *aTensorShape ) { return aTensorShape->CountLayers(); }

    MonoArray *OpNode_GetDimension( Cuda::sTensorShape *aTensorShape, int i )
    {
        auto lDim = aTensorShape->GetDimension( i );

        MonoArray *lNewArray = mono_array_new( mono_domain_get(), mono_get_uint32_class(), lDim.size() );

        for( uint32_t i = 0; i < lDim.size(); i++ ) mono_array_set( lNewArray, uint32_t, i, lDim[i] );

        return lNewArray;
    }

    void OpNode_Trim( Cuda::sTensorShape *aTensorShape, int aToDimension ) { aTensorShape->Trim( aToDimension ); }

    void OpNode_Flatten( Cuda::sTensorShape *aTensorShape, int aToDimension ) { aTensorShape->Flatten( aToDimension ); }

    void OpNode_InsertDimension( Cuda::sTensorShape *aTensorShape, int aPosition, MonoArray *aDimension )
    {
        uint32_t lArrayLength = static_cast<uint32_t>( mono_array_length( aDimension ) );

        std::vector<uint32_t> lNewDimension{};

        for( uint32_t i = 0; i < lArrayLength; i++ )
        {
            lNewDimension.push_back( *( mono_array_addr( aDimension, uint32_t, i ) ) );
        }

        aTensorShape->InsertDimension( aPosition, lNewDimension );
    }

    size_t OpNode_NewScope( uint32_t aMemorySize ) { return (size_t)( new Scope( aMemorySize ) ); }

    void OpNode_DestroyScope( Scope *aScope ) { delete aScope; }

    static Scope *ToScope( MonoObject *aScope )
    {
        auto lScopeClass = DotNetRuntime::GetClassType( "SpockEngine.Scope" );
        auto lScope      = DotNetInstance( lScopeClass.Class(), aScope );

        Scope *lRetVal = lScope.GetFieldValue<Scope *>( "mInternalScope" );
        return lRetVal;

        // return lScope.GetFieldValue<Scope *>( "mInternalScope" );
    }

    static Cuda::sTensorShape *ToShape( MonoObject *aShape )
    {
        auto lTensorShapeClass = DotNetRuntime::GetClassType( "SpockEngine.sTensorShape" );
        auto lTensorShape      = DotNetInstance( lTensorShapeClass.Class(), aShape );

        Cuda::sTensorShape *lRetVal = lTensorShape.GetFieldValue<Cuda::sTensorShape *>( "mInternalTensorShape" );
        return lRetVal;
    }

    static OpNode ToOpNode( MonoObject *aNode )
    {
        auto lOpNodeClass = DotNetRuntime::GetClassType( "SpockEngine.OpNode" );
        auto lOpNode      = DotNetInstance( lOpNodeClass.Class(), aNode );

        auto  lEntityID = lOpNode.GetFieldValue<uint32_t>( "mEntityID" );
        auto *lScope    = ToScope( lOpNode.GetFieldValue<MonoObject *>( "mScope" ) );

        return lScope->GetNodesRegistry().WrapEntity( static_cast<entt::entity>( lEntityID ) );
    }

    template <typename _Ty>
    static inline _Ty UnboxScalarType( MonoObject *aObject )
    {
        return *(_Ty *)mono_object_unbox( aObject );
    }

    template <typename _Ty>
    static inline uint32_t CreateConstantMultiTensor( Scope *aScope, sTensorShape *aShape, MonoObject *aObject )
    {
        auto lValue = UnboxScalarType<_Ty>( aObject );
        auto lNode  = MultiTensorValue( *aScope, sConstantValueInitializerComponent( lValue ), *aShape );

        return static_cast<uint32_t>( lNode );
    }

    template <typename _Ty>
    static inline uint32_t CreateConstantMultiTensor( Scope *aScope, sTensorShape *aShape, MonoArray *aObject )
    {
        std::vector<_Ty> lLayerConstants( aShape->CountLayers() );
        for( uint32_t j = 0; j < aShape->CountLayers(); j++ ) lLayerConstants[j] = *( mono_array_addr( aObject, _Ty, j ) );

        auto lNode = MultiTensorValue( *aScope, sVectorInitializerComponent( lLayerConstants ), *aShape );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_CreateMultiTensor_Constant_Initializer( MonoObject *aScope, MonoReflectionType *aType, MonoObject *aInitializer,
                                                            MonoObject *aShape )
    {
        auto *lScope = ToScope( aScope );
        auto *lShape = ToShape( aShape );

        MonoType *lMonoType = mono_reflection_type_get_type( aType );
        auto      lDataType = SE::Core::Mono::Utils::MonoTypeToScriptFieldType( lMonoType );

        switch( lDataType )
        {
        case eScriptFieldType::Float: return CreateConstantMultiTensor<float>( lScope, lShape, aInitializer );
        case eScriptFieldType::Double: return CreateConstantMultiTensor<double>( lScope, lShape, aInitializer );
        case eScriptFieldType::Bool: return CreateConstantMultiTensor<uint8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Char: return CreateConstantMultiTensor<uint8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Byte: return CreateConstantMultiTensor<int8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Short: return CreateConstantMultiTensor<int16_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Int: return CreateConstantMultiTensor<int32_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Long: return CreateConstantMultiTensor<int64_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::UByte: return CreateConstantMultiTensor<uint8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::UShort: return CreateConstantMultiTensor<uint16_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::UInt: return CreateConstantMultiTensor<uint32_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::ULong: return CreateConstantMultiTensor<uint64_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::None:
        default: return 0;
        }

        return 0;
    }

    uint32_t OpNode_CreateMultiTensor_Vector_Initializer( MonoObject *aScope, MonoReflectionType *aType, MonoArray *aInitializer,
                                                          MonoObject *aShape )
    {
        auto *lScope = ToScope( aScope );
        auto *lShape = ToShape( aShape );

        MonoType *lMonoType = mono_reflection_type_get_type( aType );
        auto      lDataType = SE::Core::Mono::Utils::MonoTypeToScriptFieldType( lMonoType );

        switch( lDataType )
        {
        case eScriptFieldType::Float: return CreateConstantMultiTensor<float>( lScope, lShape, aInitializer );
        case eScriptFieldType::Double: return CreateConstantMultiTensor<double>( lScope, lShape, aInitializer );
        case eScriptFieldType::Bool: return CreateConstantMultiTensor<uint8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Char: return CreateConstantMultiTensor<int8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Byte: return CreateConstantMultiTensor<uint8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Short: return CreateConstantMultiTensor<int16_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Int: return CreateConstantMultiTensor<int32_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Long: return CreateConstantMultiTensor<int64_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::UByte: return CreateConstantMultiTensor<uint8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::UShort: return CreateConstantMultiTensor<uint16_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::UInt: return CreateConstantMultiTensor<uint32_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::ULong: return CreateConstantMultiTensor<uint64_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::None:
        default: return 0;
        }

        return 0;
    }

    template <typename _Ty>
    static inline uint32_t CreateDataMultiTensor( Scope *aScope, sTensorShape *aShape, MonoArray *aObject )
    {
        std::vector<_Ty> lLayerConstants( aShape->CountLayers() );
        for( uint32_t j = 0; j < aShape->CountLayers(); j++ ) lLayerConstants[j] = *( mono_array_addr( aObject, _Ty, j ) );

        auto lNode = MultiTensorValue( *aScope, sDataInitializerComponent( lLayerConstants ), *aShape );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_CreateMultiTensor_Data_Initializer( MonoObject *aScope, MonoReflectionType *aType, MonoArray *aInitializer,
                                                        MonoObject *aShape )
    {
        auto *lScope = ToScope( aScope );
        auto *lShape = ToShape( aShape );

        MonoType *lMonoType = mono_reflection_type_get_type( aType );
        auto      lDataType = SE::Core::Mono::Utils::MonoTypeToScriptFieldType( lMonoType );

        switch( lDataType )
        {
        case eScriptFieldType::Float: return CreateDataMultiTensor<float>( lScope, lShape, aInitializer );
        case eScriptFieldType::Double: return CreateDataMultiTensor<double>( lScope, lShape, aInitializer );
        case eScriptFieldType::Bool: return CreateDataMultiTensor<uint8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Char: return CreateDataMultiTensor<int8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Byte: return CreateDataMultiTensor<uint8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Short: return CreateDataMultiTensor<int16_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Int: return CreateDataMultiTensor<int32_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::Long: return CreateDataMultiTensor<int64_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::UByte: return CreateDataMultiTensor<uint8_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::UShort: return CreateDataMultiTensor<uint16_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::UInt: return CreateDataMultiTensor<uint32_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::ULong: return CreateDataMultiTensor<uint64_t>( lScope, lShape, aInitializer );
        case eScriptFieldType::None:
        default: return 0;
        }

        return 0;
    }

    template <typename _Ty>
    static inline uint32_t CreateRandomDataMultiTensor( Scope *aScope, sTensorShape *aShape, eScalarType aType )
    {
        sRandomUniformInitializerComponent lInitializer{};
        lInitializer.mType = aType;

        auto lNode = MultiTensorValue( *aScope, lInitializer, *aShape );

        return static_cast<uint32_t>( lNode );
    }

    template <typename _Ty>
    static inline uint32_t CreateRandomDataMultiTensor( Scope *aScope, sTensorShape *aShape, eScalarType aType, MonoObject *aMean,
                                                        MonoObject *aStd )
    {
        sRandomNormalInitializerComponent lInitializer{};
        lInitializer.mType = aType;
        lInitializer.mMean = UnboxScalarType<_Ty>( aMean );
        lInitializer.mStd  = UnboxScalarType<_Ty>( aStd );

        auto lNode = MultiTensorValue( *aScope, lInitializer, *aShape );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_CreateMultiTensor_Random_Uniform_Initializer( MonoObject *aScope, MonoReflectionType *aType, MonoObject *aShape )
    {
        auto *lScope = ToScope( aScope );
        auto *lShape = ToShape( aShape );

        MonoType *lMonoType = mono_reflection_type_get_type( aType );
        auto      lDataType = SE::Core::Mono::Utils::MonoTypeToScriptFieldType( lMonoType );

        switch( lDataType )
        {
        case eScriptFieldType::Float: return CreateRandomDataMultiTensor<float>( lScope, lShape, eScalarType::FLOAT32 );
        case eScriptFieldType::Double: return CreateRandomDataMultiTensor<double>( lScope, lShape, eScalarType::FLOAT64 );
        default: return 0;
        }

        return 0;
    }

    uint32_t OpNode_CreateMultiTensor_Random_Normal_Initializer( MonoObject *aScope, MonoReflectionType *aType, MonoObject *aMean,
                                                                 MonoObject *aStd, MonoObject *aShape )
    {
        auto *lScope = ToScope( aScope );
        auto *lShape = ToShape( aShape );

        MonoType *lMonoType = mono_reflection_type_get_type( aType );
        auto      lDataType = SE::Core::Mono::Utils::MonoTypeToScriptFieldType( lMonoType );

        switch( lDataType )
        {
        case eScriptFieldType::Float: return CreateRandomDataMultiTensor<float>( lScope, lShape, eScalarType::FLOAT32, aMean, aStd );
        case eScriptFieldType::Double: return CreateRandomDataMultiTensor<double>( lScope, lShape, eScalarType::FLOAT64, aMean, aStd );
        default: return 0;
        }

        return 0;
    }

    uint32_t OpNode_CreateVector( MonoObject *aScope, MonoArray *aValues )
    {
        auto *lScope = ToScope( aScope );

        return 0;
    }

    template <typename _Ty>
    static inline uint32_t CreateScalarValue( Scope *aScope, MonoObject *aObject )
    {
        auto lValue = UnboxScalarType<_Ty>( aObject );
        auto lNode  = ConstantScalarValue( *aScope, lValue );

        return static_cast<uint32_t>( lNode );
    }

    template <typename _Ty>
    static inline uint32_t CreateScalarValue( Scope *aScope, MonoArray *aObject )
    {
        uint32_t         aArrayLength = static_cast<uint32_t>( mono_array_length( aObject ) );
        std::vector<_Ty> lScalarValues( aArrayLength );
        for( uint32_t j = 0; j < aArrayLength; j++ ) lScalarValues[j] = *( mono_array_addr( aObject, _Ty, j ) );

        auto lNode = VectorValue( *aScope, lScalarValues );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_CreateScalarVector( MonoObject *aScope, MonoReflectionType *aType, MonoArray *aInitializer )
    {
        auto *lScope = ToScope( aScope );

        MonoType *lMonoType = mono_reflection_type_get_type( aType );
        auto      lDataType = SE::Core::Mono::Utils::MonoTypeToScriptFieldType( lMonoType );

        switch( lDataType )
        {
        case eScriptFieldType::Float: return CreateScalarValue<float>( lScope, aInitializer );
        case eScriptFieldType::Double: return CreateScalarValue<double>( lScope, aInitializer );
        case eScriptFieldType::Bool: return CreateScalarValue<uint8_t>( lScope, aInitializer );
        case eScriptFieldType::Char: return CreateScalarValue<int8_t>( lScope, aInitializer );
        case eScriptFieldType::Byte: return CreateScalarValue<uint8_t>( lScope, aInitializer );
        case eScriptFieldType::Short: return CreateScalarValue<int16_t>( lScope, aInitializer );
        case eScriptFieldType::Int: return CreateScalarValue<int32_t>( lScope, aInitializer );
        case eScriptFieldType::Long: return CreateScalarValue<int64_t>( lScope, aInitializer );
        case eScriptFieldType::UByte: return CreateScalarValue<uint8_t>( lScope, aInitializer );
        case eScriptFieldType::UShort: return CreateScalarValue<uint16_t>( lScope, aInitializer );
        case eScriptFieldType::UInt: return CreateScalarValue<uint32_t>( lScope, aInitializer );
        case eScriptFieldType::ULong: return CreateScalarValue<uint64_t>( lScope, aInitializer );
        case eScriptFieldType::None:
        default: return 0;
        }

        return 0;
    }

    uint32_t OpNode_CreateScalarValue( MonoObject *aScope, MonoReflectionType *aType, MonoObject *aInitializer )
    {
        auto *lScope = ToScope( aScope );

        MonoType *lMonoType = mono_reflection_type_get_type( aType );
        auto      lDataType = SE::Core::Mono::Utils::MonoTypeToScriptFieldType( lMonoType );

        switch( lDataType )
        {
        case eScriptFieldType::Float: return CreateScalarValue<float>( lScope, aInitializer );
        case eScriptFieldType::Double: return CreateScalarValue<double>( lScope, aInitializer );
        case eScriptFieldType::Bool: return CreateScalarValue<uint8_t>( lScope, aInitializer );
        case eScriptFieldType::Char: return CreateScalarValue<int8_t>( lScope, aInitializer );
        case eScriptFieldType::Byte: return CreateScalarValue<uint8_t>( lScope, aInitializer );
        case eScriptFieldType::Short: return CreateScalarValue<int16_t>( lScope, aInitializer );
        case eScriptFieldType::Int: return CreateScalarValue<int32_t>( lScope, aInitializer );
        case eScriptFieldType::Long: return CreateScalarValue<int64_t>( lScope, aInitializer );
        case eScriptFieldType::UByte: return CreateScalarValue<uint8_t>( lScope, aInitializer );
        case eScriptFieldType::UShort: return CreateScalarValue<uint16_t>( lScope, aInitializer );
        case eScriptFieldType::UInt: return CreateScalarValue<uint32_t>( lScope, aInitializer );
        case eScriptFieldType::ULong: return CreateScalarValue<uint64_t>( lScope, aInitializer );
        case eScriptFieldType::None:
        default: return 0;
        }

        return 0;
    }

    uint32_t OpNode_Add( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight )
    {
        auto *lScope = ToScope( aScope );
        auto  lLeft  = ToOpNode( aLeft );
        auto  lRight = ToOpNode( aRight );
        auto  lNode  = Add( *lScope, lLeft, lRight );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Subtract( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight )
    {
        auto *lScope = ToScope( aScope );
        auto  lLeft  = ToOpNode( aLeft );
        auto  lRight = ToOpNode( aRight );
        auto  lNode  = Subtract( *lScope, lLeft, lRight );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Divide( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight )
    {
        auto *lScope = ToScope( aScope );
        auto  lLeft  = ToOpNode( aLeft );
        auto  lRight = ToOpNode( aRight );
        auto  lNode  = Divide( *lScope, lLeft, lRight );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Multiply( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight )
    {
        auto *lScope = ToScope( aScope );
        auto  lLeft  = ToOpNode( aLeft );
        auto  lRight = ToOpNode( aRight );
        auto  lNode  = Multiply( *lScope, lLeft, lRight );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_And( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight )
    {
        auto *lScope = ToScope( aScope );
        auto  lLeft  = ToOpNode( aLeft );
        auto  lRight = ToOpNode( aRight );
        auto  lNode  = And( *lScope, lLeft, lRight );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Or( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight )
    {
        auto *lScope = ToScope( aScope );
        auto  lLeft  = ToOpNode( aLeft );
        auto  lRight = ToOpNode( aRight );
        auto  lNode  = Or( *lScope, lLeft, lRight );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Not( MonoObject *aScope, MonoObject *aOperand )
    {
        auto *lScope   = ToScope( aScope );
        auto  lOperand = ToOpNode( aOperand );
        auto  lNode    = Not( *lScope, lOperand );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_BitwiseAnd( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight )
    {
        auto *lScope = ToScope( aScope );
        auto  lLeft  = ToOpNode( aLeft );
        auto  lRight = ToOpNode( aRight );
        auto  lNode  = BitwiseAnd( *lScope, lLeft, lRight );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_BitwiseOr( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight )
    {
        auto *lScope = ToScope( aScope );
        auto  lLeft  = ToOpNode( aLeft );
        auto  lRight = ToOpNode( aRight );
        auto  lNode  = BitwiseOr( *lScope, lLeft, lRight );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_BitwiseNot( MonoObject *aScope, MonoObject *aOperand )
    {
        auto *lScope   = ToScope( aScope );
        auto  lOperand = ToOpNode( aOperand );
        auto  lNode    = BitwiseNot( *lScope, lOperand );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_InInterval( MonoObject *aScope, MonoObject *aX, MonoObject *aLower, MonoObject *aUpper, bool aStrictLower,
                                bool aStrictUpper )
    {
        auto *lScope = ToScope( aScope );
        auto  lX     = ToOpNode( aX );
        auto  lLower = ToOpNode( aLower );
        auto  lUpper = ToOpNode( aUpper );
        auto  lNode  = InInterval( *lScope, lX, lLower, lUpper, aStrictLower, aStrictUpper );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Equal( MonoObject *aScope, MonoObject *aX, MonoObject *aY )
    {
        auto *lScope = ToScope( aScope );
        auto  lX     = ToOpNode( aX );
        auto  lY     = ToOpNode( aY );
        auto  lNode  = Equal( *lScope, lX, lY );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_LessThan( MonoObject *aScope, MonoObject *aX, MonoObject *aY )
    {
        auto *lScope = ToScope( aScope );
        auto  lX     = ToOpNode( aX );
        auto  lY     = ToOpNode( aY );
        auto  lNode  = LessThan( *lScope, lX, lY );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_LessThanOrEqual( MonoObject *aScope, MonoObject *aX, MonoObject *aY )
    {
        auto *lScope = ToScope( aScope );
        auto  lX     = ToOpNode( aX );
        auto  lY     = ToOpNode( aY );
        auto  lNode  = LessThanOrEqual( *lScope, lX, lY );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_GreaterThan( MonoObject *aScope, MonoObject *aX, MonoObject *aY )
    {
        auto *lScope = ToScope( aScope );
        auto  lX     = ToOpNode( aX );
        auto  lY     = ToOpNode( aY );
        auto  lNode  = GreaterThan( *lScope, lX, lY );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_GreaterThanOrEqual( MonoObject *aScope, MonoObject *aX, MonoObject *aY )
    {
        auto *lScope = ToScope( aScope );
        auto  lX     = ToOpNode( aX );
        auto  lY     = ToOpNode( aY );
        auto  lNode  = GreaterThanOrEqual( *lScope, lX, lY );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Where( MonoObject *aScope, MonoObject *aCondition, MonoObject *aValueIfTrue, MonoObject *aValueIfFalse )
    {
        auto *lScope        = ToScope( aScope );
        auto  lCondition    = ToOpNode( aCondition );
        auto  lValueIfTrue  = ToOpNode( aValueIfTrue );
        auto  lValueIfFalse = ToOpNode( aValueIfFalse );
        auto  lNode         = Where( *lScope, lCondition, lValueIfTrue, lValueIfFalse );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Mix( MonoObject *aScope, MonoObject *aA, MonoObject *aB, MonoObject *aT )
    {
        auto *lScope = ToScope( aScope );
        auto  lA     = ToOpNode( aA );
        auto  lB     = ToOpNode( aB );
        auto  lT     = ToOpNode( aT );
        auto  lNode  = Mix( *lScope, lA, lB, lT );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_AffineTransform( MonoObject *aScope, MonoObject *aA, MonoObject *aX, MonoObject *aB )
    {
        auto *lScope = ToScope( aScope );
        auto  lA     = ToOpNode( aA );
        auto  lX     = ToOpNode( aX );
        auto  lB     = ToOpNode( aB );
        auto  lNode  = AffineTransform( *lScope, lA, lX, lB );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_ARange( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight, MonoObject *aDelta )
    {
        auto *lScope = ToScope( aScope );
        auto  lLeft  = ToOpNode( aLeft );
        auto  lRight = ToOpNode( aRight );
        auto  lDelta = ToOpNode( aDelta );
        auto  lNode  = ARange( *lScope, lLeft, lRight, lDelta );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_LinearSpace( MonoObject *aScope, MonoObject *aLeft, MonoObject *aRight, MonoObject *aSubdivisions )
    {
        auto *lScope        = ToScope( aScope );
        auto  lLeft         = ToOpNode( aLeft );
        auto  lRight        = ToOpNode( aRight );
        auto  lSubdivisions = ToOpNode( aSubdivisions );
        auto  lNode         = LinearSpace( *lScope, lLeft, lRight, lSubdivisions );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Repeat( MonoObject *aScope, MonoObject *aArray, MonoObject *aRepetitions )
    {
        auto *lScope       = ToScope( aScope );
        auto  lArray       = ToOpNode( aArray );
        auto  lRepetitions = ToOpNode( aRepetitions );
        auto  lNode        = Repeat( *lScope, lArray, lRepetitions );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Tile( MonoObject *aScope, MonoObject *aArray, MonoObject *aRepetitions )
    {
        auto *lScope       = ToScope( aScope );
        auto  lArray       = ToOpNode( aArray );
        auto  lRepetitions = ToOpNode( aRepetitions );
        auto  lNode        = Tile( *lScope, lArray, lRepetitions );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Sample2D( MonoObject *aScope, MonoObject *aX, MonoObject *aY, MonoObject *aTextures )
    {
        auto *lScope    = ToScope( aScope );
        auto  lX        = ToOpNode( aX );
        auto  lY        = ToOpNode( aY );
        auto  lTextures = ToOpNode( aTextures );
        auto  lNode     = Sample2D( *lScope, lX, lY, lTextures );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Collapse( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = Collapse( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Expand( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = Expand( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Reshape( MonoObject *aScope, MonoObject *aArray, MonoObject *aNewShape )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto *lShape = ToShape( aNewShape );
        auto  lNode  = Reshape( *lScope, lArray, *lShape );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Relayout( MonoObject *aScope, MonoObject *aArray, MonoObject *aNewShape )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto *lShape = ToShape( aNewShape );
        auto  lNode  = Relayout( *lScope, lArray, *lShape );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_FlattenNode( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = Flatten( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Slice( MonoObject *aScope, MonoObject *aArray, MonoObject *aBegin, MonoObject *aEnd )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lBegin = ToOpNode( aBegin );
        auto  lEnd   = ToOpNode( aEnd );
        auto  lNode  = Slice( *lScope, lArray, lBegin, lEnd );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Summation( MonoObject *aScope, MonoObject *aArray, MonoObject *aBegin, MonoObject *aEnd )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lBegin = ToOpNode( aBegin );
        auto  lEnd   = ToOpNode( aEnd );
        auto  lNode  = Summation( *lScope, lArray, lBegin, lEnd );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_CountTrue( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = CountTrue( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_CountNonZero( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = CountNonZero( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_CountZero( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = CountZero( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Floor( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = Floor( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Ceil( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = Ceil( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Abs( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = Abs( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Sqrt( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = Sqrt( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Round( MonoObject *aScope, MonoObject *aArray )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );
        auto  lNode  = Round( *lScope, lArray );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Diff( MonoObject *aScope, MonoObject *aArray, uint32_t aCount )
    {
        auto *lScope = ToScope( aScope );
        auto  lArray = ToOpNode( aArray );

        auto lNode = Diff( *lScope, lArray, aCount );
        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Shift( MonoObject *aScope, MonoObject *aArray, int32_t aCount, MonoObject *aFillValue )
    {
        auto *lScope     = ToScope( aScope );
        auto  lArray     = ToOpNode( aArray );
        auto  lFillValue = ToOpNode( aFillValue );
        auto  lNode      = Shift( *lScope, lArray, aCount, lFillValue );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_Conv1D( MonoObject *aScope, MonoObject *aArray0, MonoObject *aArray1 )
    {
        auto *lScope  = ToScope( aScope );
        auto  lArray0 = ToOpNode( aArray0 );
        auto  lArray1 = ToOpNode( aArray1 );
        auto  lNode   = Conv1D( *lScope, lArray0, lArray1 );

        return static_cast<uint32_t>( lNode );
    }

    uint32_t OpNode_HCat( MonoObject *aScope, MonoObject *aArray0, MonoObject *aArray1 )
    {
        auto *lScope  = ToScope( aScope );
        auto  lArray0 = ToOpNode( aArray0 );
        auto  lArray1 = ToOpNode( aArray1 );
        auto  lNode   = HCat( *lScope, lArray0, lArray1 );

        return static_cast<uint32_t>( lNode );
    }

    void UI_Text( MonoString *aString ) { UI::Text( std::string( mono_string_to_utf8( aString ) ) ); }

    bool UI_Button( MonoString *aText ) { return UI::Button( mono_string_to_utf8( aText ), math::vec2( 100, 30 ) ); }
} // namespace SE::MonoInternalCalls