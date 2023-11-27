#pragma once

#include <optional>
#include <string>

#include "Core/Logging.h"
#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "UI/UI.h"

#include "Core/CUDA/Array/CudaBuffer.h"

#include "Graphics/API.h"

#include "Scene/Importer/ImporterData.h"

#include "Core/Entity/Collection.h"

#include "Primitives/Primitives.h"

// #include "DotNet/Runtime.h"

#include "Core/Profiling/BlockTimer.h"

#include "MaterialSystem/MaterialSystem.h"
#include "Renderer/MaterialSystem.h"

namespace SE::Core::EntityComponentSystem::Components
{

    using namespace math;
    using namespace literals;

    using namespace SE::Graphics;
    using namespace SE::Core::EntityComponentSystem;
    using namespace SE::Core;
    using namespace SE::Cuda;

    template <typename _Ty>
    struct Dirty
    {
    };

    struct sCameraComponent
    {
        vec3  Position = vec3{ 0.0f, 0.0f, 0.0f };
        float Pitch    = 0.0f;
        float Yaw      = 0.0f;
        float Roll     = 0.0f;

        float Near        = 0.001;
        float Far         = 1000.0f;
        float FieldOfView = 90.0f;
        float AspectRatio = 16.0f / 9.0f;

        sCameraComponent()                           = default;
        sCameraComponent( const sCameraComponent & ) = default;
    };

    struct sAnimationChannel
    {
        sImportedAnimationChannel::Channel mChannelID;
        sImportedAnimationSampler          mInterpolation;
        Entity                             mTargetNode;
    };

    struct sAnimationComponent
    {
        float    Duration       = 0.0f;
        float    TickCount      = 0.0f;
        float    TicksPerSecond = 0.0f;
        uint32_t CurrentTick    = 0;
        float    CurrentTime    = 0.0f;

        vector_t<sAnimationChannel> mChannels = {};

        sAnimationComponent()                              = default;
        sAnimationComponent( const sAnimationComponent & ) = default;
    };

    struct sAnimationChooser
    {
        vector_t<Entity> Animations = {};

        sAnimationChooser()                            = default;
        sAnimationChooser( const sAnimationChooser & ) = default;
    };

    struct sAnimatedTransformComponent
    {
        vec3 Translation;
        vec3 Scaling;
        quat Rotation;

        sAnimatedTransformComponent()                                      = default;
        sAnimatedTransformComponent( const sAnimatedTransformComponent & ) = default;
    };

    struct sNodeTransformComponent
    {
        mat4 mMatrix;

        sNodeTransformComponent()
            : mMatrix( mat4( 1.0f ) )
        {
        }
        sNodeTransformComponent( const sNodeTransformComponent & ) = default;
        sNodeTransformComponent( mat4 a_Matrix )
            : mMatrix{ a_Matrix }
        {
        }

        vec3 GetScale()
        {
            return Scaling( mMatrix );
        }
        vec3 GetTranslation() const
        {
            return Translation( mMatrix );
        }
        vec3 GetEulerRotation() const
        {
            mat3 lMatrix = Rotation( mMatrix );
            return vec3{ degrees( atan2f( lMatrix[1][2], lMatrix[2][2] ) ),
                         degrees( atan2f( -lMatrix[0][2], sqrtf( lMatrix[1][2] * lMatrix[1][2] + lMatrix[2][2] * lMatrix[2][2] ) ) ),
                         degrees( atan2f( lMatrix[0][1], lMatrix[0][0] ) ) };
        }

        sNodeTransformComponent( vec3 a_Position, vec3 a_Rotation, vec3 a_Scaling )
        {
            mat4 rot = Rotation( radians( a_Rotation.z ), z_axis() ) * Rotation( radians( a_Rotation.y ), y_axis() ) *
                       Rotation( radians( a_Rotation.x ), x_axis() );

            vec4 validScale;
            validScale.x = ( fabsf( a_Scaling.x ) < FLT_EPSILON ) ? 0.001f : a_Scaling.x;
            validScale.y = ( fabsf( a_Scaling.y ) < FLT_EPSILON ) ? 0.001f : a_Scaling.y;
            validScale.z = ( fabsf( a_Scaling.z ) < FLT_EPSILON ) ? 0.001f : a_Scaling.z;
            validScale.w = 1.0f;

            mat4 l_Scaling     = FromDiagonal( validScale );
            mat4 l_Translation = Translation( a_Position );

            mMatrix = l_Translation * rot * l_Scaling;
        }
    };

    struct sStaticTransformComponent
    {
        mat4 Matrix = mat4( 1.0f );

        sStaticTransformComponent()                                    = default;
        sStaticTransformComponent( const sStaticTransformComponent & ) = default;
        sStaticTransformComponent( mat4 a_Matrix )
        {
            Matrix = a_Matrix;
        };
    };

    struct sStaticMeshComponent
    {
        string_t mName = "";

        ePrimitiveTopology    mPrimitive         = ePrimitiveTopology::TRIANGLES;
        ref_t<IGraphicBuffer> mVertexBuffer      = nullptr;
        ref_t<IGraphicBuffer> mIndexBuffer       = nullptr;
        ref_t<IGraphicBuffer> mTransformedBuffer = nullptr;
        uint32_t              mVertexOffset      = 0;
        uint32_t              mVertexCount       = 0;
        uint32_t              mIndexOffset       = 0;
        uint32_t              mIndexCount        = 0;

        sStaticMeshComponent()                               = default;
        sStaticMeshComponent( const sStaticMeshComponent & ) = default;
    };

    struct sSkeletonComponent
    {
        uint32_t         BoneCount;
        vector_t<Entity> Bones;
        vector_t<mat4>   InverseBindMatrices;
        vector_t<mat4>   JointMatrices;

        sSkeletonComponent()                             = default;
        sSkeletonComponent( const sSkeletonComponent & ) = default;
    };

    struct sWireframeComponent
    {
        bool IsVisible = false;
        vec3 Color     = 0xffffff_rgbf;

        sWireframeComponent()                              = default;
        sWireframeComponent( const sWireframeComponent & ) = default;
    };

    struct sWireframeMeshComponent
    {
        uint32_t VertexCount = 0;

        sWireframeMeshComponent()                                  = default;
        sWireframeMeshComponent( const sWireframeMeshComponent & ) = default;
    };

    struct sBoundingBoxComponent
    {
        bool     IsVisible   = false;
        bool     Solid       = false;
        vec4     Color       = 0xffffffff_rgbaf;
        uint32_t VertexCount = 0;

        sBoundingBoxComponent()                                = default;
        sBoundingBoxComponent( const sBoundingBoxComponent & ) = default;
    };

    struct sRayTracingTargetComponent
    {
        mat4 Transform;

        sRayTracingTargetComponent()                                     = default;
        sRayTracingTargetComponent( const sRayTracingTargetComponent & ) = default;
    };

    struct sMaterialComponent
    {
        Material mMaterialID;

        sMaterialComponent() = default;
        sMaterialComponent( Material const &aMaterial )
            : mMaterialID{ aMaterial }
        {
        }
        sMaterialComponent( const sMaterialComponent & ) = default;
    };

    struct sBackgroundComponent
    {
        vec3 Color = { 1.0f, 1.0f, 1.0f };

        sBackgroundComponent()                               = default;
        sBackgroundComponent( const sBackgroundComponent & ) = default;
    };

    struct sAmbientLightingComponent
    {
        vec3  Color     = { 1.0f, 1.0f, 1.0f };
        float Intensity = 0.03f;

        sAmbientLightingComponent()                                    = default;
        sAmbientLightingComponent( const sAmbientLightingComponent & ) = default;
    };

    enum class eLightType : uint8_t
    {
        DIRECTIONAL = 0,
        SPOTLIGHT   = 1,
        POINT_LIGHT = 2
    };

    struct sLightComponent
    {
        eLightType mType = eLightType::POINT_LIGHT;

        bool  mIsOn      = true;
        float mIntensity = 100.0f;
        vec3  mColor     = { 1.0f, 1.0f, 1.0f };
        float mCone      = 60.0f;

        sLightComponent()                          = default;
        sLightComponent( const sLightComponent & ) = default;
    };

} // namespace SE::Core::EntityComponentSystem::Components
