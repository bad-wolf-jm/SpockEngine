#include "glTFImporter.h"
#include <iostream>

#include "Core/Logging.h"
#include "fmt/core.h"

namespace SE::Core
{

    static eSamplerWrapping getLTSEWrapMode( int32_t aWrapMode )
    {
        switch( aWrapMode )
        {
        case 33071:
            return eSamplerWrapping::CLAMP_TO_EDGE;
        case 33648:
            return eSamplerWrapping::MIRRORED_REPEAT;
        case 10497:
        default:
            return eSamplerWrapping::REPEAT;
        }
    }

    static eSamplerFilter getLTSEFilterMode( int32_t aFilterMode )
    {
        switch( aFilterMode )
        {
        case 9729:
        case 9986:
        case 9987:
            return eSamplerFilter::LINEAR;
        case 9728:
        case 9984:
        case 9985:
        default:
            return eSamplerFilter::NEAREST;
        }
    }

    GlTFImporter::GlTFImporter( fs::path aPath )
    {
        tinygltf::TinyGLTF lGltfContext;
        std::string        lError;
        std::string        lWarning;

        std::string lExtension = aPath.extension().string();

        bool lFileLoaded = false;
        if( lExtension == ".glb" )
        {
            lFileLoaded = lGltfContext.LoadBinaryFromFile( &mGltfModel, &lError, &lWarning, aPath.string().c_str() );
        }
        else
        {
            lFileLoaded = lGltfContext.LoadASCIIFromFile( &mGltfModel, &lError, &lWarning, aPath.string().c_str() );
        }

        if( !lFileLoaded )
        {
            std::cerr << "Could not load gltf file: " << lError << std::endl;
        }

        SE::Logging::Info( "Samplers" );
        LoadSamplers();

        SE::Logging::Info( "Textures" );
        LoadTextures();

        SE::Logging::Info( "Materials" );
        LoadMaterials();

        SE::Logging::Info( "Nodes" );
        LoadNodes();

        SE::Logging::Info( "Animations" );
        LoadAnimations();

        SE::Logging::Info( "Skins" );
        LoadSkins();

        SE::Logging::Info( "Done" );
    }

    void GlTFImporter::LoadSamplers()
    {
        for( tinygltf::Sampler lGltfSampler : mGltfModel.samplers )
        {
            sTextureSamplingInfo lSamplerInfo{};
            lSamplerInfo.mFilter                = getLTSEFilterMode( lGltfSampler.magFilter );
            lSamplerInfo.mWrapping              = getLTSEWrapMode( lGltfSampler.wrapS );
            lSamplerInfo.mNormalizedCoordinates = true;
            lSamplerInfo.mNormalizedValues      = true;
            mTextureSamplers.push_back( lSamplerInfo );
        }
    }

    void GlTFImporter::CreateTexture( uint32_t aTextureIndex, std::string aName, tinygltf::Image const &aGltfimage,
                                      sTextureSamplingInfo const &aTextureSamplingInfo )
    {
        sImageData lImageData{};
        lImageData.mFormat = eColorFormat::RGBA8_UNORM;
        lImageData.mWidth  = aGltfimage.width;
        lImageData.mHeight = aGltfimage.height;

        if( aGltfimage.component == 3 )
        {
            lImageData.mByteSize  = aGltfimage.width * aGltfimage.height * 4;
            lImageData.mPixelData = std::vector<uint8_t>( lImageData.mByteSize );

            unsigned char       *rgba = lImageData.mPixelData.data();
            unsigned char const *rgb  = &aGltfimage.image[0];
            for( int32_t i = 0; i < aGltfimage.width * aGltfimage.height; ++i )
            {
                rgba[0] = rgb[0];
                rgba[1] = rgb[1];
                rgba[2] = rgb[2];

                rgba += 4;
                rgb += 3;
            }
        }
        else
        {
            lImageData.mPixelData = std::vector<uint8_t>( &aGltfimage.image[0], &aGltfimage.image[0] + aGltfimage.image.size() );
            lImageData.mByteSize  = aGltfimage.image.size();
        }

        sTextureCreateInfo lTextureCreateInfo{};
        lTextureCreateInfo.mMipLevels = 1;

        sImportedTexture lNewTexture{};
        lNewTexture.mName    = aName.empty() ? fmt::format( "TEXTURE_{}", mTextures.size() ) : aName;
        lNewTexture.mTexture = New<TextureData2D>( lTextureCreateInfo, lImageData );
        lNewTexture.mSampler = New<TextureSampler2D>( *lNewTexture.mTexture, aTextureSamplingInfo );

        mTextures.push_back( lNewTexture );
        mTextureIDLookup[aTextureIndex] = mTextures.size() - 1;
    }

    void GlTFImporter::LoadTextures()
    {
        uint32_t lTextureIndex = 0;
        for( tinygltf::Texture &tex : mGltfModel.textures )
        {
            tinygltf::Image      image = mGltfModel.images[tex.source];
            sTextureSamplingInfo lTextureSampler{};

            if( tex.sampler != -1 )
                lTextureSampler = mTextureSamplers[tex.sampler];

            CreateTexture( lTextureIndex++, image.name, image, lTextureSampler );
        }
    }

    sImportedMaterial::sTextureReference GlTFImporter::RetrieveTextureData( tinygltf::Material &aMaterial, std::string aName )
    {
        if( aMaterial.values.find( aName ) != aMaterial.values.end() )
        {
            uint32_t lTextureID = mTextureIDLookup[aMaterial.values[aName].TextureIndex()];
            uint32_t lUVChannel = aMaterial.values[aName].TextureTexCoord();

            return sImportedMaterial::sTextureReference{ lTextureID, lUVChannel };
        }

        return sImportedMaterial::sTextureReference{};
    }

    sImportedMaterial::sTextureReference GlTFImporter::RetrieveAdditionalTextureData( tinygltf::Material &aMaterial,
                                                                                      std::string         aName )
    {
        if( aMaterial.additionalValues.find( aName ) != aMaterial.additionalValues.end() )
        {
            uint32_t lTextureID = mTextureIDLookup[aMaterial.additionalValues[aName].TextureIndex()];
            uint32_t lUVChannel = aMaterial.additionalValues[aName].TextureTexCoord();

            return sImportedMaterial::sTextureReference{ lTextureID, lUVChannel };
        }

        return sImportedMaterial::sTextureReference{};
    }

    math::vec4 GlTFImporter::RetrieveVec4( tinygltf::Material &aMaterial, std::string aName, math::vec4 aDefault )
    {
        if( aMaterial.values.find( aName ) != aMaterial.values.end() )
        {
            return math::make_vec4( aMaterial.values[aName].ColorFactor().data() );
        }

        return aDefault;
    }

    std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GlTFImporter::RetrievePrimitiveCount( const tinygltf::Primitive &aPrimitive,
                                                                                             std::string                aName )
    {
        if( aPrimitive.attributes.find( aName ) != aPrimitive.attributes.end() )
        {
            const tinygltf::Accessor   &lAccessor   = mGltfModel.accessors[aPrimitive.attributes.find( aName )->second];
            const tinygltf::BufferView &lBufferView = mGltfModel.bufferViews[lAccessor.bufferView];

            auto *lBufferData       = &( mGltfModel.buffers[lBufferView.buffer].data[lAccessor.byteOffset + lBufferView.byteOffset] );
            auto  lBufferCount      = static_cast<uint32_t>( lAccessor.count );
            auto  lBufferByteStride = lAccessor.ByteStride( lBufferView );

            return { lBufferCount, lBufferByteStride, lAccessor.componentType,
                     tinygltf::GetComponentSizeInBytes( lAccessor.componentType ) };
        }
        return { 0, 0, 0, 0 };
    }

    void GlTFImporter::LoadMaterials()
    {
        uint32_t lMaterialIndex = 0;

        for( tinygltf::Material &lMaterial : mGltfModel.materials )
        {
            sImportedMaterial lNewImportedMaterial{};

            lNewImportedMaterial.mName =
                lMaterial.name.empty() ? fmt::format( "UNNAMED_MATERIAL_{}", lMaterialIndex ) : lMaterial.name;

            lNewImportedMaterial.mConstants.mIsTwoSided      = false;
            lNewImportedMaterial.mConstants.mMetallicFactor  = RetrieveValue( lMaterial, "metallicFactor", 0.0f );
            lNewImportedMaterial.mConstants.mRoughnessFactor = RetrieveValue( lMaterial, "roughnessFactor", 1.0f );

            lNewImportedMaterial.mConstants.mBaseColorFactor  = RetrieveVec4( lMaterial, "baseColorFactor", math::vec4( 1.0f ) );
            lNewImportedMaterial.mConstants.mEmissiveFactor   = RetrieveVec4( lMaterial, "emissiveFactor", math::vec4( 1.0f ) );
            lNewImportedMaterial.mConstants.mEmissiveFactor.w = 1.0f;

            lNewImportedMaterial.mAlpha.mCutOff = RetrieveValue( lMaterial, "alphaCutoff", 0.5f );

            lNewImportedMaterial.mTextures.mBaseColorTexture         = RetrieveTextureData( lMaterial, "baseColorTexture" );
            lNewImportedMaterial.mTextures.mMetallicRoughnessTexture = RetrieveTextureData( lMaterial, "metallicRoughnessTexture" );
            lNewImportedMaterial.mTextures.mNormalTexture            = RetrieveAdditionalTextureData( lMaterial, "normalTexture" );
            lNewImportedMaterial.mTextures.mEmissiveTexture          = RetrieveAdditionalTextureData( lMaterial, "emissiveTexture" );
            lNewImportedMaterial.mTextures.mOcclusionTexture         = RetrieveAdditionalTextureData( lMaterial, "occlusionTexture" );

            if( lMaterial.additionalValues.find( "alphaMode" ) != lMaterial.additionalValues.end() )
            {
                tinygltf::Parameter param = lMaterial.additionalValues["alphaMode"];
                if( param.string_value == "BLEND" )
                    lNewImportedMaterial.mAlpha.mMode = sImportedMaterial::AlphaMode::BLEND_MODE;

                if( param.string_value == "MASK" )
                    lNewImportedMaterial.mAlpha.mMode = sImportedMaterial::AlphaMode::ALPHA_MASK_MODE;
            }

            mMaterials.push_back( lNewImportedMaterial );
            mMaterialIDLookup[lMaterialIndex++] = mMaterials.size() - 1;
        }
        // Push a default lNewImportedMaterial at the end of the list for meshes with no material assigned
        mMaterials.push_back( sImportedMaterial{} );
    }

    void GlTFImporter::LoadNode( uint32_t aParentID, tinygltf::Node const &aNode, uint32_t aNodeID )
    {
        mNodes.push_back( sImportedNode{} );
        auto    &lNewNode = mNodes.back();
        uint32_t lNodeID  = static_cast<uint32_t>( mNodes.size() - 1 );

        mNodeIDLookup[aNodeID] = lNodeID;

        lNewNode.mName     = aNode.name.empty() ? fmt::format( "UNNAMED_NODE_{}", lNodeID ) : aNode.name;
        lNewNode.mParentID = aParentID;

        if( aNode.matrix.size() == 16 )
        {
            lNewNode.mTransform = math::make_mat4x4( aNode.matrix.data() );
        }
        else
        {
            // Generate local node matrix
            math::mat4 lTranslation = math::mat4( 1.0f );
            math::mat4 lRotation    = math::mat4( 1.0f );
            math::mat4 lScale       = math::mat4( 1.0f );

            if( aNode.translation.size() == 3 )
            {
                lTranslation = math::Translation( math::make_vec3( aNode.translation.data() ) );
            }
            if( aNode.rotation.size() == 4 )
            {
                math::quat lQ = math::make_quat( aNode.rotation.data() );
                lRotation     = math::mat4( lQ );
            }
            if( aNode.scale.size() == 3 )
            {
                lScale = math::Scaling( math::make_vec3( aNode.scale.data() ) );
            }

            lNewNode.mTransform = lTranslation * lRotation * lScale;
        }

        if( aNode.children.size() > 0 )
            for( size_t i = 0; i < aNode.children.size(); i++ )
                LoadNode( lNodeID, mGltfModel.nodes[aNode.children[i]], aNode.children[i] );

        if( aParentID != std::numeric_limits<uint32_t>::max() )
            mNodes[aParentID].mChildren.push_back( lNodeID );

        if( aNode.mesh > -1 )
        {
            const tinygltf::Mesh lMeshData = mGltfModel.meshes[aNode.mesh];

            for( size_t j = 0; j < lMeshData.primitives.size(); j++ )
            {
                const tinygltf::Primitive &lPrimitive = lMeshData.primitives[j];

                mMeshes.emplace_back();
                sImportedMesh &lNewImportedMesh = mMeshes.back();
                lNewImportedMesh.mName          = fmt::format( "MESH_{}", mMeshes.size() - 1 );

                switch( lPrimitive.mode )
                {
                case 0:
                    lNewImportedMesh.mPrimitive = Graphics::ePrimitiveTopology::POINTS;
                    break;
                case 1:
                    lNewImportedMesh.mPrimitive = Graphics::ePrimitiveTopology::LINES;
                    break;
                case 4:
                    lNewImportedMesh.mPrimitive = Graphics::ePrimitiveTopology::TRIANGLES;
                    break;
                case 2:
                case 3:
                case 5:
                case 6:
                default:
                    lNewImportedMesh.mPrimitive = Graphics::ePrimitiveTopology::POINTS;
                }

                lNewImportedMesh.mMaterialID = ( lPrimitive.material > -1 ) ? mMaterialIDLookup[lPrimitive.material] : 0;

                assert( lPrimitive.attributes.find( "POSITION" ) != lPrimitive.attributes.end() );
                RetrievePrimitiveAttribute<math::vec3>( lPrimitive, "POSITION", lNewImportedMesh.mPositions );

                RetrievePrimitiveAttribute<math::vec3>( lPrimitive, "NORMAL", lNewImportedMesh.mNormals );
                if( lNewImportedMesh.mNormals.size() == 0 )
                    lNewImportedMesh.mNormals = std::vector<math::vec3>( lNewImportedMesh.mPositions.size() );

                RetrievePrimitiveAttribute<math::vec2>( lPrimitive, "TEXCOORD_0", lNewImportedMesh.mUV0 );
                if( lNewImportedMesh.mUV0.size() == 0 )
                    lNewImportedMesh.mUV0 = std::vector<math::vec2>( lNewImportedMesh.mPositions.size() );

                RetrievePrimitiveAttribute<math::vec2>( lPrimitive, "TEXCOORD_1", lNewImportedMesh.mUV1 );
                if( lNewImportedMesh.mUV1.size() == 0 )
                    lNewImportedMesh.mUV1 = std::vector<math::vec2>( lNewImportedMesh.mPositions.size() );

                RetrievePrimitiveAttribute<math::vec4>( lPrimitive, "WEIGHTS_0", lNewImportedMesh.mWeights );
                if( lNewImportedMesh.mWeights.size() == 0 )
                    lNewImportedMesh.mWeights = std::vector<math::vec4>( lNewImportedMesh.mPositions.size() );

                std::vector<uint8_t> lJointData;
                RetrievePrimitiveAttribute<uint8_t>( lPrimitive, "JOINTS_0", lJointData );
                auto [lCount, lStride, lComponentType, lComponentSize] = RetrievePrimitiveCount( lPrimitive, "JOINTS_0" );
                if( lCount > 0 )
                {
                    uint32_t lComponentCount = lStride / lComponentSize;

                    switch( lComponentType )
                    {
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                    {
                        const uint16_t       *buf = reinterpret_cast<const uint16_t *>( lJointData.data() );
                        std::vector<uint16_t> lJoints0( lCount * lComponentCount );
                        memcpy( lJoints0.data(), buf, lCount * lStride );

                        std::vector<uint32_t> lJoints1( lJoints0.begin(), lJoints0.end() );
                        lNewImportedMesh.mJoints.resize( lCount );

                        for( uint32_t i = 0; i < lCount; i++ )
                            lNewImportedMesh.mJoints[i] =
                                math::uvec4( lJoints1[i * lComponentCount + 0], lJoints1[i * lComponentCount + 1],
                                             lJoints1[i * lComponentCount + 2], lJoints1[i * lComponentCount + 3] );
                        break;
                    }
                    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                    {
                        const uint8_t       *buf = reinterpret_cast<const uint8_t *>( lJointData.data() );
                        std::vector<uint8_t> lJoints0( lCount * lComponentCount );
                        memcpy( lJoints0.data(), buf, lCount * lStride );

                        std::vector<uint32_t> lJoints1( lJoints0.begin(), lJoints0.end() );
                        lNewImportedMesh.mJoints.resize( lCount );

                        for( uint32_t i = 0; i < lCount; i++ )
                            lNewImportedMesh.mJoints[i] =
                                math::uvec4( lJoints1[i * lComponentCount + 0], lJoints1[i * lComponentCount + 1],
                                             lJoints1[i * lComponentCount + 2], lJoints1[i * lComponentCount + 3] );
                        break;
                    }
                    default:
                        // Not supported by spec
                        std::cerr << "Joint component type " << lComponentType << " not supported!" << std::endl;
                        break;
                    }
                }
                else
                {
                    lNewImportedMesh.mJoints = std::vector<math::uvec4>( lNewImportedMesh.mPositions.size() );
                }

                if( lPrimitive.indices > -1 )
                {
                    const tinygltf::Accessor   &accessor   = mGltfModel.accessors[lPrimitive.indices > -1 ? lPrimitive.indices : 0];
                    const tinygltf::BufferView &bufferView = mGltfModel.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer     &buffer     = mGltfModel.buffers[bufferView.buffer];

                    auto        indexCount = static_cast<uint32_t>( accessor.count );
                    const void *dataPtr    = &( buffer.data[accessor.byteOffset + bufferView.byteOffset] );

                    lNewImportedMesh.mIndices.resize( indexCount );

                    switch( accessor.componentType )
                    {
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
                    {
                        const uint32_t *buf = static_cast<const uint32_t *>( dataPtr );
                        for( size_t index = 0; index < accessor.count; index++ )
                        {
                            lNewImportedMesh.mIndices[index] = buf[index];
                        }
                        break;
                    }
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
                    {
                        const uint16_t *buf = static_cast<const uint16_t *>( dataPtr );
                        for( size_t index = 0; index < accessor.count; index++ )
                        {
                            lNewImportedMesh.mIndices[index] = buf[index];
                        }
                        break;
                    }
                    case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
                    {
                        const uint8_t *buf = static_cast<const uint8_t *>( dataPtr );
                        for( size_t index = 0; index < accessor.count; index++ )
                        {
                            lNewImportedMesh.mIndices[index] = buf[index];
                        }
                        break;
                    }
                    default:
                        std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
                        return;
                    }
                }

                lNewNode.mMeshes.push_back( mMeshes.size() - 1 );
            }
        }
    }

    void GlTFImporter::LoadNodes()
    {
        const tinygltf::Scene &scene = mGltfModel.scenes[mGltfModel.defaultScene > -1 ? mGltfModel.defaultScene : 0];
        for( size_t i = 0; i < scene.nodes.size(); i++ )
        {
            const tinygltf::Node node = mGltfModel.nodes[scene.nodes[i]];
            LoadNode( std::numeric_limits<uint32_t>::max(), node, scene.nodes[i] );
        }
    }

    void GlTFImporter::LoadSkins()
    {
        uint32_t lSkinID = 0;
        for( tinygltf::Skin &source : mGltfModel.skins )
        {
            mSkins.emplace_back( sImportedSkin{} );
            auto &lNewImportedSkin = mSkins.back();

            lNewImportedSkin.mName = source.name.empty() ? fmt::format( "SKIN_{}", mSkins.size() - 1 ) : source.name;

            if( source.skeleton > -1 )
                lNewImportedSkin.mSkeletonRootNodeID = mNodeIDLookup[source.skeleton];

            for( int jointIndex : source.joints )
                lNewImportedSkin.mJointNodeID.push_back( mNodeIDLookup[jointIndex] );

            // Get inverse bind matrices from buffer
            if( source.inverseBindMatrices > -1 )
            {
                const tinygltf::Accessor   &accessor   = mGltfModel.accessors[source.inverseBindMatrices];
                const tinygltf::BufferView &bufferView = mGltfModel.bufferViews[accessor.bufferView];
                const tinygltf::Buffer     &buffer     = mGltfModel.buffers[bufferView.buffer];
                lNewImportedSkin.mInverseBindMatrices.resize( accessor.count );
                memcpy( lNewImportedSkin.mInverseBindMatrices.data(), &buffer.data[accessor.byteOffset + bufferView.byteOffset],
                        accessor.count * sizeof( math::mat4 ) );
            }

            mSkinIDLookup[lSkinID++] = mSkins.size() - 1;
        }

        uint32_t lNodeID = 0;
        for( auto &lNode : mGltfModel.nodes )
        {
            if( lNode.skin > -1 )
                mNodes[mNodeIDLookup[lNodeID]].mSkinID = mSkinIDLookup[lNode.skin];
            lNodeID++;
        }
    }

    void GlTFImporter::LoadAnimations()
    {
        for( tinygltf::Animation &anim : mGltfModel.animations )
        {
            mAnimations.emplace_back( sImportedAnimation{} );

            auto &lNewImportedAnimation = mAnimations.back();

            lNewImportedAnimation.mName = anim.name.empty() ? fmt::format( "ANIMATION_{}", mAnimations.size() ) : anim.name;

            // Samplers
            for( auto &samp : anim.samplers )
            {
                lNewImportedAnimation.mSamplers.emplace_back( sImportedAnimationSampler{} );
                auto &lNewSampler = lNewImportedAnimation.mSamplers.back();

                if( samp.interpolation == "LINEAR" )
                    lNewSampler.mInterpolation = sImportedAnimationSampler::Interpolation::LINEAR;

                if( samp.interpolation == "STEP" )
                    lNewSampler.mInterpolation = sImportedAnimationSampler::Interpolation::STEP;

                if( samp.interpolation == "CUBICSPLINE" )
                    lNewSampler.mInterpolation = sImportedAnimationSampler::Interpolation::CUBICSPLINE;

                // Read sampler input time values
                {
                    const tinygltf::Accessor   &accessor   = mGltfModel.accessors[samp.input];
                    const tinygltf::BufferView &bufferView = mGltfModel.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer     &buffer     = mGltfModel.buffers[bufferView.buffer];

                    assert( accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT );

                    const void  *dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];
                    const float *buf     = static_cast<const float *>( dataPtr );
                    for( size_t index = 0; index < accessor.count; index++ )
                    {
                        lNewSampler.mInputs.push_back( buf[index] );
                    }

                    for( auto input : lNewSampler.mInputs )
                    {
                        if( input < lNewImportedAnimation.mStart )
                            lNewImportedAnimation.mStart = input;

                        if( input > lNewImportedAnimation.mEnd )
                            lNewImportedAnimation.mEnd = input;
                    }
                }

                // Read sampler output T/R/S values
                {
                    const tinygltf::Accessor   &accessor   = mGltfModel.accessors[samp.output];
                    const tinygltf::BufferView &bufferView = mGltfModel.bufferViews[accessor.bufferView];
                    const tinygltf::Buffer     &buffer     = mGltfModel.buffers[bufferView.buffer];

                    assert( accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT );

                    const void *dataPtr = &buffer.data[accessor.byteOffset + bufferView.byteOffset];

                    switch( accessor.type )
                    {
                    case TINYGLTF_TYPE_VEC3:
                    {
                        const math::vec3 *buf = static_cast<const math::vec3 *>( dataPtr );
                        for( size_t index = 0; index < accessor.count; index++ )
                            lNewSampler.mOutputsVec4.push_back( math::vec4( buf[index], 0.0f ) );
                        break;
                    }
                    case TINYGLTF_TYPE_VEC4:
                    {
                        const math::vec4 *buf = static_cast<const math::vec4 *>( dataPtr );
                        for( size_t index = 0; index < accessor.count; index++ )
                            lNewSampler.mOutputsVec4.push_back( buf[index] );
                        break;
                    }
                    default:
                    {
                        std::cout << "unknown type" << std::endl;
                        break;
                    }
                    }
                }
            }

            // Channels
            for( auto &source : anim.channels )
            {
                // vkglTF::AnimationChannel channel{};
                lNewImportedAnimation.mChannels.emplace_back( sImportedAnimationChannel{} );
                auto &lNewChannels = lNewImportedAnimation.mChannels.back();

                lNewChannels.mSamplerIndex = source.sampler;
                lNewChannels.mNodeID       = mNodeIDLookup[source.target_node];
                if( lNewChannels.mNodeID == std::numeric_limits<uint32_t>::max() )
                    continue;

                if( source.target_path == "rotation" )
                    lNewChannels.mComponent = sImportedAnimationChannel::Channel::ROTATION;

                if( source.target_path == "translation" )
                    lNewChannels.mComponent = sImportedAnimationChannel::Channel::TRANSLATION;

                if( source.target_path == "scale" )
                    lNewChannels.mComponent = sImportedAnimationChannel::Channel::SCALE;

                if( source.target_path == "weights" )
                {
                    std::cout << "weights not yet supported, skipping channel" << std::endl;
                    continue;
                }
            }
        }
    }
} // namespace SE::Core