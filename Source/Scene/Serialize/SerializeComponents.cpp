#include "SerializeComponents.h"

namespace SE::Core
{
    void ReadComponent( sTag &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sTag"].IsNull() )
        {
            aEntity.Add<sTag>( aNode["sTag"]["mValue"].As<std::string>( "" ) );
        }
    }

    void ReadComponent( sCameraComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sCameraComponent"].IsNull() )
        {
            auto &lComponent       = aEntity.Add<sCameraComponent>();
            lComponent.Position    = aNode["sCameraComponent"]["Position"].Vec( { "x", "y", "z" }, math::vec3{ 0.0f, 0.0f, 0.0f } );
            lComponent.Pitch       = aNode["sCameraComponent"]["Pitch"].As<float>( 0.0f );
            lComponent.Yaw         = aNode["sCameraComponent"]["Yaw"].As<float>( 0.0f );
            lComponent.Roll        = aNode["sCameraComponent"]["Roll"].As<float>( 0.0f );
            lComponent.Near        = aNode["sCameraComponent"]["Near"].As<float>( 0.0f );
            lComponent.Far         = aNode["sCameraComponent"]["Far"].As<float>( 0.0f );
            lComponent.FieldOfView = aNode["sCameraComponent"]["FieldOfView"].As<float>( 0.0f );
            lComponent.AspectRatio = aNode["sCameraComponent"]["AspectRatio"].As<float>( 0.0f );
        }
    }

    void ReadComponent( sActorComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sActorComponent"].IsNull() )
        {
            auto  lClassFullName      = aNode["sActorComponent"]["mClassFullName"].As<std::string>( "" );
            auto &lNewScriptComponent = aEntity.Add<sActorComponent>( lClassFullName );

            lNewScriptComponent.Initialize( aEntity );
        }
    }

    void ReadComponent( sAnimationChooser &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sAnimationChooser"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sAnimationChooser>();
            aNode["sAnimationChooser"].ForEach(
                [&]( ConfigurationNode &aNode )
                {
                    std::string lAnimationUUID = aNode.As<std::string>( "" );
                    Entity      lAnimationNode = aEntities[lAnimationUUID];

                    lComponent.Animations.push_back( lAnimationNode );
                    SE::Logging::Info( "ANIMATION {}", lAnimationUUID );
                } );
        }
    }

    void ReadComponent( sAnimationComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities,
                        std::vector<sImportedAnimationSampler> &aInterpolationData )
    {
        if( !aNode["sAnimationComponent"].IsNull() )
        {
            auto &lComponent     = aEntity.Add<sAnimationComponent>();
            lComponent.Duration  = aNode["sAnimationComponent"]["Duration"].As<float>( 0.0f );
            lComponent.mChannels = std::vector<sAnimationChannel>{};

            aNode["sAnimationComponent"]["mChannels"].ForEach(
                [&]( ConfigurationNode &aInterpolationDataNode )
                {
                    sAnimationChannel lNewChannel{};
                    std::string       lTargetNodeUUID = aInterpolationDataNode["mTargetNode"].As<std::string>( "" );

                    lNewChannel.mTargetNode = aEntities[lTargetNodeUUID];
                    lNewChannel.mChannelID =
                        static_cast<sImportedAnimationChannel::Channel>( aInterpolationDataNode["mChannelID"].As<uint32_t>( 0 ) );
                    lNewChannel.mInterpolation =
                        aInterpolationData[aInterpolationDataNode["mInterpolationDataIndex"].As<uint32_t>( 0 )];

                    lComponent.mChannels.push_back( lNewChannel );
                } );
        }
    }

    void ReadComponent( sAnimatedTransformComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sAnimatedTransformComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sAnimatedTransformComponent>();

            lComponent.Translation =
                aNode["sAnimatedTransformComponent"]["Translation"].Vec( { "x", "y", "z" }, math::vec3{ 0.0f, 0.0f, 0.0f } );
            lComponent.Scaling =
                aNode["sAnimatedTransformComponent"]["Scaling"].Vec( { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );

            auto lCoefficients =
                aNode["sAnimatedTransformComponent"]["Rotation"].Vec( { "x", "y", "z", "w" }, math::vec4{ 0.0f, 0.0f, 0.0f, 0.0f } );
            lComponent.Rotation.x = lCoefficients.x;
            lComponent.Rotation.y = lCoefficients.y;
            lComponent.Rotation.z = lCoefficients.z;
            lComponent.Rotation.w = lCoefficients.w;
        }
    }

    math::mat4 ReadMatrix( ConfigurationNode &aNode )
    {

        std::vector<float> lMatrixEntries{};
        aNode.ForEach( [&]( ConfigurationNode &aNode ) { lMatrixEntries.push_back( aNode.As<float>( 0.0f ) ); } );

        math::mat4 lMatrix;
        for( uint32_t c = 0; c < 4; c++ )
            for( uint32_t r = 0; r < 4; r++ ) lMatrix[c][r] = lMatrixEntries[4 * c + r];

        return lMatrix;
    }

    void ReadMatrix( math::mat4 &aMatrix, ConfigurationNode &aNode )
    {

        std::vector<float> lMatrixEntries{};
        aNode.ForEach( [&]( ConfigurationNode &aNode ) { lMatrixEntries.push_back( aNode.As<float>( 0.0f ) ); } );

        for( uint32_t c = 0; c < 4; c++ )
            for( uint32_t r = 0; r < 4; r++ ) aMatrix[c][r] = lMatrixEntries[4 * c + r];
    }

    void ReadComponent( sNodeTransformComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sLocalTransformComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sNodeTransformComponent>();

            ReadMatrix( lComponent.mMatrix, aNode["sLocalTransformComponent"]["mMatrix"] );
        }
    }

    void ReadComponent( sTransformMatrixComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["TransformMatrixComponent"].IsNull() )
        {
            auto &lComponent = aEntity.AddOrReplace<sTransformMatrixComponent>();

            ReadMatrix( lComponent.Matrix, aNode["TransformMatrixComponent"]["mMatrix"] );
        }
    }

    void ReadComponent( sStaticMeshComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sStaticMeshComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sStaticMeshComponent>();

            auto &lMeshData          = aNode["sStaticMeshComponent"];
            lComponent.mVertexOffset = lMeshData["mVertexOffset"].As<uint32_t>( 0 );
            lComponent.mVertexCount  = lMeshData["mVertexCount"].As<uint32_t>( 0 );
            lComponent.mIndexOffset  = lMeshData["mIndexOffset"].As<uint32_t>( 0 );
            lComponent.mIndexCount   = lMeshData["mIndexCount"].As<uint32_t>( 0 );
        }
    }

    void ReadComponent( sParticleSystemComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sParticleSystemComponent"].IsNull() )
        {
        }
    }

    void ReadComponent( sParticleShaderComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sParticleShaderComponent"].IsNull() )
        {
        }
    }

    void ReadComponent( sWireframeComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sWireframeComponent"].IsNull() )
        {
        }
    }

    void ReadComponent( sWireframeMeshComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sWireframeMeshComponent"].IsNull() )
        {
        }
    }

    void ReadComponent( sBoundingBoxComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sBoundingBoxComponent"].IsNull() )
        {
        }
    }

    void ReadComponent( sSkeletonComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sSkeletonComponent"].IsNull() )
        {
            auto &lData = aNode["sSkeletonComponent"];

            std::vector<Entity> lBones{};
            lData["Bones"].ForEach(
                [&]( ConfigurationNode &aNode )
                {
                    auto lUUID = aNode.As<std::string>( "" );
                    if( lUUID.empty() ) return;

                    lBones.push_back( aEntities[lUUID] );
                } );

            std::vector<math::mat4> lInverseBindMatrices{};
            lData["InverseBindMatrices"].ForEach( [&]( ConfigurationNode &aNode )
                                                  { lInverseBindMatrices.push_back( ReadMatrix( aNode ) ); } );

            std::vector<math::mat4> lJointMatrices{};
            lData["JointMatrices"].ForEach( [&]( ConfigurationNode &aNode ) { lJointMatrices.push_back( ReadMatrix( aNode ) ); } );

            auto &lComponent               = aEntity.Add<sSkeletonComponent>();
            lComponent.BoneCount           = lBones.size();
            lComponent.Bones               = lBones;
            lComponent.InverseBindMatrices = lInverseBindMatrices;
            lComponent.JointMatrices       = lJointMatrices;
        }
    }

    void ReadComponent( sRayTracingTargetComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sRayTracingTargetComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sRayTracingTargetComponent>();

            ReadMatrix( lComponent.Transform, aNode["sRayTracingTargetComponent"]["Transform"] );
        }
    }

    void ReadComponent( sMaterialComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {

        if( !aNode["sMaterialComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sMaterialComponent>();

            lComponent.mMaterialID = aNode["sMaterialComponent"]["mMaterialID"].As<uint32_t>( 0 );
            SE::Logging::Info( "{}", lComponent.mMaterialID );
        }
    }

    void ReadComponent( sMaterialShaderComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sMaterialShaderComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sMaterialShaderComponent>();
            auto &lData      = aNode["sMaterialShaderComponent"];

            lComponent.Type              = static_cast<eCMaterialType>( lData["Type"].As<uint8_t>( 0 ) );
            lComponent.IsTwoSided        = lData["IsTwoSided"].As<bool>( true );
            lComponent.UseAlphaMask      = lData["UseAlphaMask"].As<bool>( true );
            lComponent.LineWidth         = lData["LineWidth"].As<float>( 1.0f );
            lComponent.AlphaMaskTheshold = lData["AlphaMaskTheshold"].As<float>( .5f );
        }
    }

    void ReadComponent( sBackgroundComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sBackgroundComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sBackgroundComponent>();

            lComponent.Color = aNode["sBackgroundComponent"]["Color"].Vec( { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );
        }
    }

    void ReadComponent( sAmbientLightingComponent &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sAmbientLightingComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sAmbientLightingComponent>();

            lComponent.Color = aNode["sAmbientLightingComponent"]["Color"].Vec( { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );
            lComponent.Intensity = aNode["sAmbientLightingComponent"]["Intensity"].As<float>( .0005f );
        }
    }

    void ReadComponent( sLightComponent cons &aComponent, ConfigurationNode const &aNode, EntityMap &aEntities )
    {
        if( !aNode["sLightComponent"].IsNull() )
        {
            auto &lComponent = aEntity.Add<sLightComponent>();

            std::unordered_map<std::string, eLightType> lLightTypeLookup = { { "DIRECTIONAL", eLightType::DIRECTIONAL },
                                                                             { "SPOTLIGHT", eLightType::SPOTLIGHT },
                                                                             { "POINT_LIGHT", eLightType::POINT_LIGHT },
                                                                             { "", eLightType::POINT_LIGHT } };

            lComponent.mType      = lLightTypeLookup[aNode["sLightComponent"]["mType"].As<std::string>( "" )];
            lComponent.mColor     = aNode["sLightComponent"]["mColor"].Vec( { "x", "y", "z" }, math::vec3{ 1.0f, 1.0f, 1.0f } );
            lComponent.mIntensity = aNode["sLightComponent"]["mIntensity"].As<float>( .0005f );
            lComponent.mCone      = aNode["sLightComponent"]["mCone"].As<float>( .0005f );
        }
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sTag const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "mValue", aComponent.mValue );
        }
        aOut.EndMap();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sRelationshipComponent const &aComponent )
    {
        aOut.WriteKey( aName );
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

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sCameraComponent const &aComponent )
    {
        aOut.WriteKey( aName );
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

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sAnimationChooser const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.BeginSequence( true );
        {
            for( auto &lAnimationEntity : aComponent.Animations )
            {
                if( lAnimationEntity ) aOut.Write( lAnimationEntity.Get<sUUID>().mValue.str() );
            }
        }
        aOut.EndSequence();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sActorComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.BeginMap( true );
        aOut.WriteKey( "mClassFullName", aComponent.mClassFullName );
        aOut.EndMap();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sAnimatedTransformComponent const &aComponent )
    {
        aOut.WriteKey( aName );
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

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sNodeTransformComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "mMatrix" );
            aOut.Write( aComponent.mMatrix );
        }
        aOut.EndMap();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sTransformMatrixComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "mMatrix" );
            aOut.Write( aComponent.Matrix );
        }
        aOut.EndMap();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sStaticMeshComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "mVertexOffset", aComponent.mVertexOffset );
            aOut.WriteKey( "mVertexCount", aComponent.mVertexCount );
            aOut.WriteKey( "mIndexOffset", aComponent.mIndexOffset );
            aOut.WriteKey( "mIndexCount", aComponent.mIndexCount );
        }
        aOut.EndMap();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sParticleSystemComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.WriteNull();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sParticleShaderComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.WriteNull();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sSkeletonComponent const &aComponent )
    {
        aOut.WriteKey( aName );
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

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sWireframeComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.WriteNull();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sWireframeMeshComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.WriteNull();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sBoundingBoxComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.WriteNull();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sRayTracingTargetComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "Transform" );
            aOut.Write( aComponent.Transform );
        }
        aOut.EndMap();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sMaterialComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "mMaterialID", aComponent.mMaterialID );
        }
        aOut.EndMap();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sMaterialShaderComponent const &aComponent )
    {
        aOut.WriteKey( aName );
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

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sBackgroundComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "Color" );
            aOut.Write( aComponent.Color, { "r", "g", "b" } );
        }
        aOut.EndMap();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sAmbientLightingComponent const &aComponent )
    {
        aOut.WriteKey( aName );
        aOut.BeginMap( true );
        {
            aOut.WriteKey( "Intensity", aComponent.Intensity );
            aOut.WriteKey( "Color" );
            aOut.Write( aComponent.Color, { "r", "g", "b" } );
        }
        aOut.EndMap();
    }

    void DoWriteComponent( ConfigurationWriter &aOut, std::string &aName, sLightComponent const &aComponent )
    {
        aOut.WriteKey( aName );
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