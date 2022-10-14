#pragma once

#include <optional>
#include <string>

#include "Core/Math/Types.h"
#include "Core/Memory.h"
#include "Core/Types.h"

#include "UI/UI.h"

#include "Core/Cuda/CudaBuffer.h"
#include "Core/Cuda/ExternalMemory.h"

#include "Core/GraphicContext//Buffer.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//Texture2D.h"
#include "Core/GraphicContext//TextureCubeMap.h"

#include "Scene/Importer/ImporterData.h"

#include "Core/EntityRegistry/Registry.h"

#include "Core/GraphicContext//UI/UIContext.h"

#include "Renderer/MeshRenderer.h"
#include "Renderer/ParticleSystemRenderer.h"

#include "Primitives/Primitives.h"

namespace LTSE::Core::EntityComponentSystem::Components
{

    using namespace math;
    using namespace LTSE::Graphics;
    using namespace LTSE::Core::EntityComponentSystem;
    using namespace LTSE::Core;
    using namespace LTSE::Cuda;
    using namespace math::literals;

    template <typename _Ty>
    struct Dirty
    {
    };

    struct CameraComponent
    {
        math::vec3 Position = math::vec3{ 0.0f, 0.0f, 0.0f };
        float      Pitch    = 0.0f;
        float      Yaw      = 0.0f;
        float      Roll     = 0.0f;

        float Near        = 0.001;
        float Far         = 1000.0f;
        float FieldOfView = 90.0f;
        float AspectRatio = 16.0f / 9.0f;

        CameraComponent()                         = default;
        CameraComponent( const CameraComponent& ) = default;
    };

    struct AnimationChannel
    {
        sImportedAnimationChannel::Channel mChannelID;
        sImportedAnimationSampler          mInterpolation;
        Entity                             mTargetNode;
    };

    struct AnimationComponent
    {
        float Duration       = 0.0f;
        float TickCount      = 0.0f;
        float TicksPerSecond = 0.0f;
        float CurrentTick    = 0.0f;

        std::vector<AnimationChannel> mChannels = {};

        AnimationComponent()                            = default;
        AnimationComponent( const AnimationComponent& ) = default;
    };

    struct AnimationChooser
    {
        std::vector<Entity> Animations = {};

        AnimationChooser()                          = default;
        AnimationChooser( const AnimationChooser& ) = default;
    };

    struct AnimatedTransformComponent
    {
        math::vec3 Translation;
        math::vec3 Scaling;
        math::quat Rotation;

        AnimatedTransformComponent()                                    = default;
        AnimatedTransformComponent( const AnimatedTransformComponent& ) = default;
    };

    // struct TransformComponent
    // {
    //     Ref<transform_t> T = nullptr;

    //     TransformComponent()
    //         : T( New<transform_t>() )
    //     {
    //     }
    //     TransformComponent( const TransformComponent & ) = default;
    //     TransformComponent( Ref<transform_t> a_T ) { T = a_T; };
    //     TransformComponent( math::mat4 a_Matrix ) { T = New<transform_t>( a_Matrix ); };
    // };

    struct LocalTransformComponent
    {
        math::mat4 mMatrix;

        LocalTransformComponent()
            : mMatrix( math::mat4( 1.0f ) )
        {
        }
        LocalTransformComponent( const LocalTransformComponent& ) = default;
        LocalTransformComponent( math::mat4 a_Matrix )
            : mMatrix{ a_Matrix }
        {
        }

        math::vec3 GetScale() { return math::Scaling( mMatrix ); }
        math::vec3 GetTranslation() const { return math::Translation( mMatrix ); }
        math::vec3 GetEulerRotation() const
        {
            math::mat3 lMatrix = math::Rotation( mMatrix );
            return math::vec3{ math::degrees( atan2f( lMatrix[1][2], lMatrix[2][2] ) ),
                math::degrees(
                    atan2f( -lMatrix[0][2], sqrtf( lMatrix[1][2] * lMatrix[1][2] + lMatrix[2][2] * lMatrix[2][2] ) ) ),
                math::degrees( atan2f( lMatrix[0][1], lMatrix[0][0] ) ) };
        }

        LocalTransformComponent( math::vec3 a_Position, math::vec3 a_Rotation, math::vec3 a_Scaling )
        {
            math::mat4 rot = math::Rotation( math::radians( a_Rotation.z ), math::z_axis() ) *
                             math::Rotation( math::radians( a_Rotation.y ), math::y_axis() ) *
                             math::Rotation( math::radians( a_Rotation.x ), math::x_axis() );

            math::vec4 validScale;
            validScale.x = ( fabsf( a_Scaling.x ) < FLT_EPSILON ) ? 0.001f : a_Scaling.x;
            validScale.y = ( fabsf( a_Scaling.y ) < FLT_EPSILON ) ? 0.001f : a_Scaling.y;
            validScale.z = ( fabsf( a_Scaling.z ) < FLT_EPSILON ) ? 0.001f : a_Scaling.z;
            validScale.w = 1.0f;

            math::mat4 l_Scaling     = math::FromDiagonal( validScale );
            math::mat4 l_Translation = math::Translation( a_Position );

            mMatrix = l_Translation * rot * l_Scaling;
        }
    };

    struct TransformMatrixComponent
    {
        math::mat4 Matrix = math::mat4( 1.0f );

        TransformMatrixComponent()                                  = default;
        TransformMatrixComponent( const TransformMatrixComponent& ) = default;
        TransformMatrixComponent( math::mat4 a_Matrix ) { Matrix = a_Matrix; };
    };

    struct StaticMeshComponent
    {
        std::string Name = "";

        ePrimitiveTopology Primitive      = ePrimitiveTopology::TRIANGLES;
        uint32_t           VertexCount    = 0;
        uint32_t           PrimitiveCount = 0;
        Ref<Buffer>        Vertices       = nullptr;
        Ref<Buffer>        Indices        = nullptr;
        uint32_t           mVertexOffset  = 0;
        uint32_t           mVertexCount   = 0;
        uint32_t           mIndexOffset   = 0;
        uint32_t           mIndexCount    = 0;

        StaticMeshComponent()                             = default;
        StaticMeshComponent( const StaticMeshComponent& ) = default;
    };

    struct ParticleSystemComponent
    {
        std::string Name = "";

        uint32_t    ParticleCount = 0;
        float       ParticleSize  = 0.0f;
        Ref<Buffer> Particles;

        ParticleSystemComponent()                                 = default;
        ParticleSystemComponent( const ParticleSystemComponent& ) = default;
    };

    struct ParticleShaderComponent
    {
        float                  LineWidth = 1.0f;
        ParticleSystemRenderer Renderer{};

        ParticleShaderComponent()                                 = default;
        ParticleShaderComponent( const ParticleShaderComponent& ) = default;
    };

    struct SkeletonComponent
    {
        uint32_t                BoneCount;
        std::vector<Entity>     Bones;
        std::vector<math::mat4> InverseBindMatrices;
        std::vector<math::mat4> JointMatrices;

        SkeletonComponent()                           = default;
        SkeletonComponent( const SkeletonComponent& ) = default;
    };

    struct WireframeComponent
    {
        bool IsVisible = false;
        vec3 Color     = 0xffffff_rgbf;

        WireframeComponent()                            = default;
        WireframeComponent( const WireframeComponent& ) = default;
    };

    struct WireframeMeshComponent
    {
        uint32_t    VertexCount  = 0;
        Ref<Buffer> VertexBuffer = nullptr;
        Ref<Buffer> IndexBuffer  = nullptr;

        WireframeMeshComponent()                                = default;
        WireframeMeshComponent( const WireframeMeshComponent& ) = default;
    };

    struct BoundingBoxComponent
    {
        bool        IsVisible    = false;
        bool        Solid        = false;
        vec4        Color        = 0xffffffff_rgbaf;
        uint32_t    VertexCount  = 0;
        Ref<Buffer> VertexBuffer = nullptr;
        Ref<Buffer> IndexBuffer  = nullptr;

        BoundingBoxComponent()                              = default;
        BoundingBoxComponent( const BoundingBoxComponent& ) = default;
    };

    struct RayTracingTargetComponent
    {
        math::mat4 Transform;
        GPUMemory  Vertices;
        GPUMemory  Indices;

        RayTracingTargetComponent()                                   = default;
        RayTracingTargetComponent( const RayTracingTargetComponent& ) = default;
    };

    struct MaterialComponent
    {
        uint32_t mMaterialID;

        MaterialComponent()                           = default;
        MaterialComponent( const MaterialComponent& ) = default;
    };

    enum class MaterialType : uint8_t
    {
        Opaque,
        Mask,
        Blend
    };

    struct RendererComponent
    {
        Entity Material;

        RendererComponent()                           = default;
        RendererComponent( const RendererComponent& ) = default;
    };

    struct MaterialShaderComponent
    {
        MaterialType Type              = MaterialType::Opaque;
        bool         IsTwoSided        = false;
        bool         UseAlphaMask      = true;
        float        LineWidth         = 1.0f;
        float        AlphaMaskTheshold = 0.5;

        MaterialShaderComponent()                                 = default;
        MaterialShaderComponent( const MaterialShaderComponent& ) = default;
    };

    struct BackgroundComponent
    {
        vec3 Color = { 1.0f, 1.0f, 1.0f };

        BackgroundComponent()                             = default;
        BackgroundComponent( const BackgroundComponent& ) = default;
    };

    struct AmbientLightingComponent
    {
        vec3  Color     = { 1.0f, 1.0f, 1.0f };
        float Intensity = 0.0005f;

        AmbientLightingComponent()                                  = default;
        AmbientLightingComponent( const AmbientLightingComponent& ) = default;
    };

    enum class LightType : uint32_t
    {
        DIRECTIONAL = 0,
        SPOTLIGHT   = 1,
        POINT_LIGHT = 2
    };

    struct DirectionalLightComponent
    {
        float Azimuth   = 0.0f;
        float Elevation = 0.0f;
        float Intensity = 0.0f;
        vec3  Color     = { 0.0f, 0.0f, 0.0f };

        bool operator==( const DirectionalLightComponent& a_Other )
        {
            return ( Azimuth == a_Other.Azimuth ) && ( Elevation == a_Other.Elevation ) &&
                   ( Intensity == a_Other.Intensity ) && ( Color == a_Other.Color );
        }
    };

    struct PointLightComponent
    {
        vec3  Position  = { 0.0f, 0.0f, 0.0f };
        vec3  Color     = { 0.0f, 0.0f, 0.0f };
        float Intensity = 0.0f;

        bool operator==( const PointLightComponent& a_Other )
        {
            return ( Position == a_Other.Position ) && ( Color == a_Other.Color ) && ( Intensity == a_Other.Intensity );
        }
    };

    struct SpotlightComponent
    {
        vec3  Position  = { 0.0f, 0.0f, 0.0f };
        float Azimuth   = 0.0f;
        float Elevation = 0.0f;

        vec3  Color     = { 0.0f, 0.0f, 0.0f };
        float Intensity = 0.0f;
        float Cone      = 0.0;

        bool operator==( const SpotlightComponent& a_Other )
        {
            return ( Position == a_Other.Position ) && ( Azimuth == a_Other.Azimuth ) &&
                   ( Elevation == a_Other.Elevation ) && ( Intensity == a_Other.Intensity ) && ( Color == a_Other.Color ) &&
                   ( Cone == a_Other.Cone );
        }
    };

    struct LightComponent
    {
        Entity Light;

        LightComponent()                        = default;
        LightComponent( const LightComponent& ) = default;
    };

} // namespace LTSE::Core::EntityComponentSystem::Components
