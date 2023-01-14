#include "SerializeComponents.h"

namespace SE::Core
{
    // clang-format off
    static std::unordered_map<std::string, std::string> gTypeTags = 
    {
        { typeid(sTag).name(),                        "TAG" },
        { typeid(sRelationshipComponent).name(),      "RELATIONSHIP" },
        { typeid(sCameraComponent).name(),            "CAMERA" },
        { typeid(sAnimationChooser).name(),           "ANIMATION_CHOOSER" },
        { typeid(sAnimationComponent).name(),         "ANIMATION" },
        { typeid(sActorComponent).name(),             "ACTOR" },
        { typeid(sAnimatedTransformComponent).name(), "ANIMATED_TRANSFORM" },
        { typeid(sNodeTransformComponent).name(),     "NODE_TRANSFORM" },
        { typeid(sTransformMatrixComponent).name(),   "TRANSFORM_MATRIX" },
        { typeid(sStaticMeshComponent).name(),        "STATIC_MESH" },
        { typeid(sParticleSystemComponent).name(),    "PARTICLE_SYSTEM" },
        { typeid(sParticleShaderComponent).name(),    "PARTICLE_SHADER" },
        { typeid(sSkeletonComponent).name(),          "SKELETON" },
        { typeid(sWireframeComponent).name(),         "WIREFRAME" },
        { typeid(sWireframeMeshComponent).name(),     "WIREFRAME_MESH" },
        { typeid(sBoundingBoxComponent).name(),       "BOUNDING_BOX" },
        { typeid(sRayTracingTargetComponent).name(),  "RAY_TRACING_TARGET" },
        { typeid(sMaterialComponent).name(),          "MATERIAL_COMPONENT" },
        { typeid(sMaterialShaderComponent).name(),    "MATERIAL_SHADER" },
        { typeid(sBackgroundComponent).name(),        "BACKGROUND" },
        { typeid(sAmbientLightingComponent).name(),   "AMBIENT_LIGHTING" },
        { typeid(sLightComponent).name(),             "LIGHT" }
    };
    // clang-format on

    std::string const &GetTypeTag( std::string const &aTypeName )
    {
        if( gTypeTags.find( aTypeName ) != gTypeTags.end() ) return gTypeTags[aTypeName];

        return "VOID";
    }

    template <uint32_t N>
    bool HasAll( YAML::Node const &aNode, std::array<std::string, N> const &aKeys )
    {
        for( auto &lKey : aKeys )
        {
            if( !( aNode[lKey] ) ) return false;
        }
        return true;
    }

    math::vec2 Get( YAML::Node const &aNode, std::array<std::string, 2> const &aKeys, math::vec2 const &aDefault )
    {
        if( !HasAll( aNode, aKeys ) ) return aDefault;

        return math::vec2{ Get( aNode[aKeys[0]], aDefault.x ), Get( aNode[aKeys[1]], aDefault.y ) };
    }

    math::vec3 Get( YAML::Node const &aNode, std::array<std::string, 3> const &aKeys, math::vec3 const &aDefault )
    {
        if( !HasAll( aNode, aKeys ) ) return aDefault;

        return math::vec3{ Get( aNode[aKeys[0]], aDefault.x ), Get( aNode[aKeys[1]], aDefault.y ),
                           Get( aNode[aKeys[2]], aDefault.z ) };
    }

    math::vec4 Get( YAML::Node const &aNode, std::array<std::string, 4> const &aKeys, math::vec4 const &aDefault )
    {
        if( !HasAll( aNode, aKeys ) ) return aDefault;

        return math::vec4{ Get( aNode[aKeys[0]], aDefault.x ), Get( aNode[aKeys[1]], aDefault.y ), Get( aNode[aKeys[2]], aDefault.z ),
                           Get( aNode[aKeys[3]], aDefault.w ) };
    }

    void ReadComponent( sTag &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.mValue = Get<std::string>( aNode["mValue"], "" );
    }

    void ReadComponent( sCameraComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.Position    = Get( aNode["Position"], { "x", "y", "z" }, math::vec3{ 0.0f, 0.0f, 0.0f } );
        aComponent.Pitch       = Get( aNode["Pitch"], 0.0f );
        aComponent.Yaw         = Get( aNode["Yaw"], 0.0f );
        aComponent.Roll        = Get( aNode["Roll"], 0.0f );
        aComponent.Near        = Get( aNode["Near"], 0.0f );
        aComponent.Far         = Get( aNode["Far"], 0.0f );
        aComponent.FieldOfView = Get( aNode["FieldOfView"], 0.0f );
        aComponent.AspectRatio = Get( aNode["AspectRatio"], 0.0f );
    }

    void ReadComponent( sActorComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.mClassFullName = Get( aNode["mClassFullName"], std::string{ "" } );
    }

    void ReadComponent( sAnimationChooser &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        for( YAML::const_iterator it = aNode.begin(); it != aNode.end(); ++it )
        {
            std::string lAnimationUUID = Get( *it, std::string{ "" } );
            Entity      lAnimationNode = aReadConext.mEntities[lAnimationUUID];

            aComponent.Animations.push_back( lAnimationNode );
            SE::Logging::Info( "ANIMATION {}", lAnimationUUID );
        }
    }

    void ReadComponent( sAnimationComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext,
                        std::vector<sImportedAnimationSampler> &aInterpolationData )
    {
        aComponent.Duration  = Get( aNode["Duration"], 0.0f );
        aComponent.mChannels = std::vector<sAnimationChannel>{};

        auto const &lChannels = aNode["mChannels"];
        for( YAML::const_iterator it = lChannels.begin(); it != lChannels.end(); ++it )
        {
            auto &aInterpolationDataNode = *it;

            sAnimationChannel lNewChannel{};
            std::string       lTargetNodeUUID = Get( aInterpolationDataNode["mTargetNode"], std::string{ "" } );

            lNewChannel.mTargetNode = aReadConext.mEntities[lTargetNodeUUID];
            lNewChannel.mChannelID =
                static_cast<sImportedAnimationChannel::Channel>( Get( aInterpolationDataNode["mChannelID"], uint32_t{ 0 } ) );
            lNewChannel.mInterpolation = aInterpolationData[Get( aInterpolationDataNode["mInterpolationDataIndex"], uint32_t{ 0 } )];

            aComponent.mChannels.push_back( lNewChannel );
        }
    }

    void ReadComponent( sAnimatedTransformComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.Translation = Get( aNode["Translation"], { "x", "y", "z" }, math::vec3{ 0.0f, 0.0f, 0.0f } );
        aComponent.Scaling     = Get( aNode["Scaling"], { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );

        auto lCoefficients    = Get( aNode["Rotation"], { "x", "y", "z", "w" }, math::vec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
        aComponent.Rotation.x = lCoefficients.x;
        aComponent.Rotation.y = lCoefficients.y;
        aComponent.Rotation.z = lCoefficients.z;
        aComponent.Rotation.w = lCoefficients.w;
    }

    static math::mat4 ReadMatrix( YAML::Node const &aNode )
    {
        std::vector<float> lMatrixEntries{};
        for( YAML::const_iterator it = aNode.begin(); it != aNode.end(); ++it ) lMatrixEntries.push_back( Get( *it, 0.0f ) );
        // aNode.ForEach( [&]( YAML::Node &aNode ) { lMatrixEntries.push_back( aNode.As<float>( 0.0f ) ); } );

        math::mat4 lMatrix;
        for( uint32_t c = 0; c < 4; c++ )
            for( uint32_t r = 0; r < 4; r++ ) lMatrix[c][r] = lMatrixEntries[4 * c + r];

        return lMatrix;
    }

    void ReadComponent( sNodeTransformComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.mMatrix = ReadMatrix( aNode["mMatrix"] );
    }

    void ReadComponent( sTransformMatrixComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.Matrix = ReadMatrix( aNode["mMatrix"] );
    }

    void ReadComponent( sStaticMeshComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.mVertexBuffer = nullptr;
        aComponent.mIndexBuffer  = nullptr;
        aComponent.mVertexOffset = Get( aNode["mVertexOffset"], uint32_t{ 0 } );
        aComponent.mVertexCount  = Get( aNode["mVertexCount"], uint32_t{ 0 } );
        aComponent.mIndexOffset  = Get( aNode["mIndexOffset"], uint32_t{ 0 } );
        aComponent.mIndexCount   = Get( aNode["mIndexCount"], uint32_t{ 0 } );
    }

    void ReadComponent( sParticleSystemComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        //
    }

    void ReadComponent( sParticleShaderComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        //
    }

    void ReadComponent( sWireframeComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        //
    }

    void ReadComponent( sWireframeMeshComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        //
    }

    void ReadComponent( sBoundingBoxComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        //
    }

    void ReadComponent( sSkeletonComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        auto const &lBonesNode = aNode["Bones"];
        for( YAML::const_iterator it = lBonesNode.begin(); it != lBonesNode.end(); ++it )
        {
            auto lUUID = Get( *it, std::string{ "" } );
            if( lUUID.empty() ) return;

            aComponent.Bones.push_back( aReadConext.mEntities[lUUID] );
        }

        auto const &lInverseBindMatrices = aNode["InverseBindMatrices"];
        for( YAML::const_iterator it = lInverseBindMatrices.begin(); it != lInverseBindMatrices.end(); ++it )
        {
            aComponent.InverseBindMatrices.push_back( ReadMatrix( *it ) );
        }

        auto const &lJointMatrices = aNode["JointMatrices"];
        for( YAML::const_iterator it = lJointMatrices.begin(); it != lJointMatrices.end(); ++it )
        {
            aComponent.JointMatrices.push_back( ReadMatrix( *it ) );
        }

        aComponent.BoneCount = aComponent.Bones.size();
    }

    void ReadComponent( sRayTracingTargetComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.Transform = ReadMatrix( aNode["Transform"] );
    }

    void ReadComponent( sMaterialComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.mMaterialID = 0; 
    }

    void ReadComponent( sMaterialShaderComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.Type              = static_cast<eCMaterialType>( Get( aNode["Type"], uint8_t{ 0 } ) );
        aComponent.IsTwoSided        = Get( aNode["IsTwoSided"], true );
        aComponent.UseAlphaMask      = Get( aNode["UseAlphaMask"], true );
        aComponent.LineWidth         = Get( aNode["LineWidth"], 1.0f );
        aComponent.AlphaMaskTheshold = Get( aNode["AlphaMaskTheshold"], 0.5f );
    }

    void ReadComponent( sBackgroundComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.Color = Get( aNode["Color"], { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );
    }

    void ReadComponent( sAmbientLightingComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        aComponent.Color     = Get( aNode["Color"], { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );
        aComponent.Intensity = Get( aNode["Intensity"], 0.0005f );
    }

    void ReadComponent( sLightComponent &aComponent, YAML::Node const &aNode, sReadContext &aReadConext )
    {
        std::unordered_map<std::string, eLightType> lLightTypeLookup = { { "DIRECTIONAL", eLightType::DIRECTIONAL },
                                                                         { "SPOTLIGHT", eLightType::SPOTLIGHT },
                                                                         { "POINT_LIGHT", eLightType::POINT_LIGHT },
                                                                         { "", eLightType::POINT_LIGHT } };

        aComponent.mType      = lLightTypeLookup[Get( aNode["mType"], std::string{ "" } )];
        aComponent.mColor     = Get( aNode["mColor"], { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );
        aComponent.mIntensity = Get( aNode["mIntensity"], 0.0005f );
        aComponent.mCone      = Get( aNode["mCone"], 0.0005f );
    }

    template <typename _Ty>
    void WriteTypeTag( ConfigurationWriter &aOut )
    {
        auto lInternalTypeName = std::string( typeid( _Ty ).name() );
        if( gTypeTags.find( lInternalTypeName ) != gTypeTags.end() ) aOut.WriteKey( gTypeTags[lInternalTypeName] );
    }

    void WriteComponent( ConfigurationWriter &aOut, sTag const &aComponent )
    {
        WriteTypeTag<sTag>( aOut );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "mValue", aComponent.mValue );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sRelationshipComponent const &aComponent )
    {
        WriteTypeTag<sRelationshipComponent>( aOut );
        aOut.BeginMap( true );
        {
            if( aComponent.mParent )
            {
                aOut.WriteKey( "mParent", aComponent.mParent.Get<sUUID>().mValue.str() );
            }
            else
            {
                aOut.WriteKey( "mParent" );
                aOut.WriteNull();
            }
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sCameraComponent const &aComponent )
    {
        WriteTypeTag<sCameraComponent>( aOut );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "Position" );
            aOut.Write( aComponent.Position, { "x", "y", "z" } );
            aOut.WriteKey( "Pitch", aComponent.Pitch );
            aOut.WriteKey( "Yaw", aComponent.Yaw );
            aOut.WriteKey( "Roll", aComponent.Roll );
            aOut.WriteKey( "Near", aComponent.Near );
            aOut.WriteKey( "Far", aComponent.Far );
            aOut.WriteKey( "FieldOfView", aComponent.FieldOfView );
            aOut.WriteKey( "AspectRatio", aComponent.AspectRatio );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sAnimationChooser const &aComponent )
    {
        WriteTypeTag<sAnimationChooser>( aOut );
        aOut.BeginSequence( true );
        {
            for( auto &lAnimationEntity : aComponent.Animations )
            {
                if( lAnimationEntity ) aOut.Write( lAnimationEntity.Get<sUUID>().mValue.str() );
            }
        }
        aOut.EndSequence();
    }

    void WriteComponent( ConfigurationWriter &aOut, sActorComponent const &aComponent )
    {
        WriteTypeTag<sActorComponent>( aOut );
        aOut.BeginMap( true );
        aOut.WriteKey( "mClassFullName", aComponent.mClassFullName );
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sAnimatedTransformComponent const &aComponent )
    {
        WriteTypeTag<sAnimatedTransformComponent>( aOut );
        aOut.BeginMap();
        {
            aOut.WriteKey( "Translation" );
            aOut.Write( aComponent.Translation, { "x", "y", "z" } );
            aOut.WriteKey( "Scaling" );
            aOut.Write( aComponent.Scaling, { "x", "y", "z" } );
            aOut.WriteKey( "Rotation" );
            aOut.Write( aComponent.Rotation, { "x", "y", "z", "w" } );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sNodeTransformComponent const &aComponent )
    {
        WriteTypeTag<sNodeTransformComponent>( aOut );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "mMatrix" );
            aOut.Write( aComponent.mMatrix );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sTransformMatrixComponent const &aComponent )
    {
        WriteTypeTag<sTransformMatrixComponent>( aOut );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "mMatrix" );
            aOut.Write( aComponent.Matrix );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sStaticMeshComponent const &aComponent, std::string const &aMeshPath )
    {
        WriteTypeTag<sStaticMeshComponent>( aOut );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "mMeshData", aMeshPath );
            aOut.WriteKey( "mVertexOffset", aComponent.mVertexOffset );
            aOut.WriteKey( "mVertexCount", aComponent.mVertexCount );
            aOut.WriteKey( "mIndexOffset", aComponent.mIndexOffset );
            aOut.WriteKey( "mIndexCount", aComponent.mIndexCount );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sParticleSystemComponent const &aComponent )
    {
        WriteTypeTag<sParticleSystemComponent>( aOut );
        aOut.WriteNull();
    }

    void WriteComponent( ConfigurationWriter &aOut, sParticleShaderComponent const &aComponent )
    {
        WriteTypeTag<sParticleShaderComponent>( aOut );
        aOut.WriteNull();
    }

    void WriteComponent( ConfigurationWriter &aOut, sSkeletonComponent const &aComponent )
    {
        WriteTypeTag<sSkeletonComponent>( aOut );
        aOut.BeginMap();
        {
            aOut.WriteKey( "BoneCount", (uint32_t)aComponent.BoneCount );
            aOut.WriteKey( "Bones" );
            aOut.BeginSequence( true );
            {
                for( auto &x : aComponent.Bones ) aOut.Write( x.Get<sUUID>().mValue.str() );
            }
            aOut.EndSequence();
            aOut.WriteKey( "InverseBindMatrices" );
            aOut.BeginSequence( true );
            {
                for( auto &x : aComponent.InverseBindMatrices ) aOut.Write( x );
            }
            aOut.EndSequence();
            aOut.WriteKey( "JointMatrices" );
            aOut.BeginSequence( true );
            {
                for( auto &x : aComponent.JointMatrices ) aOut.Write( x );
            }
            aOut.EndSequence();
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sWireframeComponent const &aComponent )
    {
        WriteTypeTag<sWireframeComponent>( aOut );
        aOut.WriteNull();
    }

    void WriteComponent( ConfigurationWriter &aOut, sWireframeMeshComponent const &aComponent )
    {
        WriteTypeTag<sWireframeMeshComponent>( aOut );
        aOut.WriteNull();
    }

    void WriteComponent( ConfigurationWriter &aOut, sBoundingBoxComponent const &aComponent )
    {
        WriteTypeTag<sBoundingBoxComponent>( aOut );
        aOut.WriteNull();
    }

    void WriteComponent( ConfigurationWriter &aOut, sRayTracingTargetComponent const &aComponent )
    {
        WriteTypeTag<sRayTracingTargetComponent>( aOut );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "Transform" );
            aOut.Write( aComponent.Transform );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sMaterialComponent const &aComponent, std::string const &aMaterialPath )
    {
        WriteTypeTag<sMaterialComponent>( aOut );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "mMaterialPath", aMaterialPath );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sMaterialShaderComponent const &aComponent )
    {
        WriteTypeTag<sMaterialShaderComponent>( aOut );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "Type", (uint32_t)aComponent.Type );
            aOut.WriteKey( "IsTwoSided", aComponent.IsTwoSided );
            aOut.WriteKey( "UseAlphaMask", aComponent.UseAlphaMask );
            aOut.WriteKey( "LineWidth", aComponent.LineWidth );
            aOut.WriteKey( "AlphaMaskTheshold", aComponent.AlphaMaskTheshold );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sBackgroundComponent const &aComponent )
    {
        WriteTypeTag<sBackgroundComponent>( aOut );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "Color" );
            aOut.Write( aComponent.Color, { "r", "g", "b" } );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sAmbientLightingComponent const &aComponent )
    {
        WriteTypeTag<sAmbientLightingComponent>( aOut );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "Intensity", aComponent.Intensity );
            aOut.WriteKey( "Color" );
            aOut.Write( aComponent.Color, { "r", "g", "b" } );
        }
        aOut.EndMap();
    }

    void WriteComponent( ConfigurationWriter &aOut, sLightComponent const &aComponent )
    {
        WriteTypeTag<sLightComponent>( aOut );
        aOut.BeginMap( true );
        {
            std::unordered_map<eLightType, std::string> lLightTypeLookup = { { eLightType::DIRECTIONAL, "DIRECTIONAL" },
                                                                             { eLightType::SPOTLIGHT, "SPOTLIGHT" },
                                                                             { eLightType::POINT_LIGHT, "POINT_LIGHT" } };

            aOut.WriteKey( "mType", lLightTypeLookup[aComponent.mType] );
            aOut.WriteKey( "mColor" );
            aOut.Write( aComponent.mColor, { "r", "g", "b" } );
            aOut.WriteKey( "mIntensity", aComponent.mIntensity );
            aOut.WriteKey( "mCone", aComponent.mCone );
        }
        aOut.EndMap();
    }
} // namespace SE::Core