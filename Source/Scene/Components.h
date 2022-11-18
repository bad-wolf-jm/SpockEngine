#pragma once

#include <optional>
#include <string>

#include "Core/Logging.h"
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

#include "Mono/Manager.h"

#include "Core/Profiling/BlockTimer.h"

namespace SE::Core::EntityComponentSystem::Components
{

    using namespace math;
    using namespace SE::Graphics;
    using namespace SE::Core::EntityComponentSystem;
    using namespace SE::Core;
    using namespace SE::Cuda;
    using namespace math::literals;

    template <typename _Ty>
    struct Dirty
    {
    };

    struct sCameraComponent
    {
        math::vec3 Position = math::vec3{ 0.0f, 0.0f, 0.0f };
        float      Pitch    = 0.0f;
        float      Yaw      = 0.0f;
        float      Roll     = 0.0f;

        float Near        = 0.001;
        float Far         = 1000.0f;
        float FieldOfView = 90.0f;
        float AspectRatio = 16.0f / 9.0f;

        sCameraComponent()                           = default;
        sCameraComponent( const sCameraComponent & ) = default;
    };

    struct sActorComponent
    {
        std::string mClassFullName = "";

        ScriptClass         mClass;
        ScriptClassInstance mInstance;
        ScriptClassInstance mEntityInstance;

        sActorComponent()                          = default;
        sActorComponent( const sActorComponent & ) = default;

        ~sActorComponent() = default;

        sActorComponent( const std::string &aClassFullName )
            : mClassFullName{ aClassFullName }

        {
            size_t      lSeparatorPos   = aClassFullName.find_last_of( '.' );
            std::string lClassNamespace = aClassFullName.substr( 0, lSeparatorPos );
            std::string lClassName      = aClassFullName.substr( lSeparatorPos + 1 );

            mClass = ScriptClass( lClassNamespace, lClassName, false );
        }

        template <typename T>
        T &Get()
        {
            return mEntity.Get<T>();
        }

        void Initialize( Entity aEntity ) { mEntity = aEntity; }

        void OnCreate()
        {
            // Create Mono side entity object
            auto lEntityID    = static_cast<uint32_t>( mEntity );
            auto lRegistryID  = (size_t)mEntity.GetRegistry();
            auto lEntityClass = ScriptClass( "SpockEngine", "Entity", true );
            mEntityInstance   = lEntityClass.Instantiate( lEntityID, lRegistryID );

            // Instantiate the Mono actor class with the entity object as parameter
            mInstance = mClass.Instantiate();
            mInstance.CallMethod( "Initialize", (size_t)mEntityInstance.GetInstance() );
            mInstance.InvokeMethod( "BeginScenario", 0, nullptr );
        }

        void OnDestroy() { mInstance.InvokeMethod( "EndScenario", 0, nullptr ); }

        void OnUpdate( Timestep ts ) { mInstance.CallMethod( "Tick", ts.GetMilliseconds() ); }

        Entity GetControlledEntity() const { return mEntity; };

      private:
        Entity mEntity;

        ScriptClassMethod mOnUpdate{};
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

        std::vector<sAnimationChannel> mChannels = {};

        sAnimationComponent()                              = default;
        sAnimationComponent( const sAnimationComponent & ) = default;
    };

    struct sAnimationChooser
    {
        std::vector<Entity> Animations = {};

        sAnimationChooser()                            = default;
        sAnimationChooser( const sAnimationChooser & ) = default;
    };

    struct sAnimatedTransformComponent
    {
        math::vec3 Translation;
        math::vec3 Scaling;
        math::quat Rotation;

        sAnimatedTransformComponent()                                      = default;
        sAnimatedTransformComponent( const sAnimatedTransformComponent & ) = default;
    };

    struct sNodeTransformComponent
    {
        math::mat4 mMatrix;

        sNodeTransformComponent()
            : mMatrix( math::mat4( 1.0f ) )
        {
        }
        sNodeTransformComponent( const sNodeTransformComponent & ) = default;
        sNodeTransformComponent( math::mat4 a_Matrix )
            : mMatrix{ a_Matrix }
        {
        }

        math::vec3 GetScale() { return math::Scaling( mMatrix ); }
        math::vec3 GetTranslation() const { return math::Translation( mMatrix ); }
        math::vec3 GetEulerRotation() const
        {
            math::mat3 lMatrix = math::Rotation( mMatrix );
            return math::vec3{
                math::degrees( atan2f( lMatrix[1][2], lMatrix[2][2] ) ),
                math::degrees( atan2f( -lMatrix[0][2], sqrtf( lMatrix[1][2] * lMatrix[1][2] + lMatrix[2][2] * lMatrix[2][2] ) ) ),
                math::degrees( atan2f( lMatrix[0][1], lMatrix[0][0] ) ) };
        }

        sNodeTransformComponent( math::vec3 a_Position, math::vec3 a_Rotation, math::vec3 a_Scaling )
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

    struct sLocalTransformComponent
    {
        math::mat4 mMatrix = math::mat4( 1.0f );

        sLocalTransformComponent()                                   = default;
        sLocalTransformComponent( const sLocalTransformComponent & ) = default;
        sLocalTransformComponent( math::mat4 a_Matrix ) { mMatrix = a_Matrix; };
    };

    struct sTransformMatrixComponent
    {
        math::mat4 Matrix = math::mat4( 1.0f );

        sTransformMatrixComponent()                                    = default;
        sTransformMatrixComponent( const sTransformMatrixComponent & ) = default;
        sTransformMatrixComponent( math::mat4 a_Matrix ) { Matrix = a_Matrix; };
    };

    struct sStaticTransformComponent
    {
        math::mat4 Matrix = math::mat4( 1.0f );

        sStaticTransformComponent()                                    = default;
        sStaticTransformComponent( const sStaticTransformComponent & ) = default;
        sStaticTransformComponent( math::mat4 a_Matrix ) { Matrix = a_Matrix; };
    };

    struct sStaticMeshComponent
    {
        std::string Name = "";

        ePrimitiveTopology Primitive     = ePrimitiveTopology::TRIANGLES;
        uint32_t           mVertexOffset = 0;
        uint32_t           mVertexCount  = 0;
        uint32_t           mIndexOffset  = 0;
        uint32_t           mIndexCount   = 0;

        sStaticMeshComponent()                               = default;
        sStaticMeshComponent( const sStaticMeshComponent & ) = default;
    };

    struct sParticleSystemComponent
    {
        std::string Name = "";

        uint32_t    ParticleCount = 0;
        float       ParticleSize  = 0.0f;
        Ref<Buffer> Particles;

        sParticleSystemComponent()                                   = default;
        sParticleSystemComponent( const sParticleSystemComponent & ) = default;
    };

    struct sParticleShaderComponent
    {
        float LineWidth = 1.0f;
        // ParticleSystemRenderer Renderer{};

        sParticleShaderComponent()                                   = default;
        sParticleShaderComponent( const sParticleShaderComponent & ) = default;
    };

    struct sSkeletonComponent
    {
        uint32_t                BoneCount;
        std::vector<Entity>     Bones;
        std::vector<math::mat4> InverseBindMatrices;
        std::vector<math::mat4> JointMatrices;

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
        math::mat4 Transform;

        sRayTracingTargetComponent()                                     = default;
        sRayTracingTargetComponent( const sRayTracingTargetComponent & ) = default;
    };

    struct sMaterialComponent
    {
        uint32_t mMaterialID;

        sMaterialComponent()                             = default;
        sMaterialComponent( const sMaterialComponent & ) = default;
    };

    enum class eCMaterialType : uint8_t
    {
        Opaque,
        Mask,
        Blend
    };

    struct sMaterialShaderComponent
    {
        eCMaterialType Type              = eCMaterialType::Opaque;
        bool           IsTwoSided        = false;
        bool           UseAlphaMask      = true;
        float          LineWidth         = 1.0f;
        float          AlphaMaskTheshold = 0.5;

        sMaterialShaderComponent()                                   = default;
        sMaterialShaderComponent( const sMaterialShaderComponent & ) = default;
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
        float Intensity = 0.0005f;

        sAmbientLightingComponent()                                    = default;
        sAmbientLightingComponent( const sAmbientLightingComponent & ) = default;
    };

    enum class eLightType : uint32_t
    {
        DIRECTIONAL = 0,
        SPOTLIGHT   = 1,
        POINT_LIGHT = 2
    };

    struct sDirectionalLightComponent
    {
        float Azimuth   = 0.0f;
        float Elevation = 0.0f;
        float Intensity = 0.0f;
        vec3  Color     = { 0.0f, 0.0f, 0.0f };

        bool operator==( const sDirectionalLightComponent &a_Other )
        {
            return ( Azimuth == a_Other.Azimuth ) && ( Elevation == a_Other.Elevation ) && ( Intensity == a_Other.Intensity ) &&
                   ( Color == a_Other.Color );
        }
    };

    struct sPointLightComponent
    {
        vec3  Position  = { 0.0f, 0.0f, 0.0f };
        vec3  Color     = { 0.0f, 0.0f, 0.0f };
        float Intensity = 0.0f;

        bool operator==( const sPointLightComponent &a_Other )
        {
            return ( Position == a_Other.Position ) && ( Color == a_Other.Color ) && ( Intensity == a_Other.Intensity );
        }
    };

    struct sSpotlightComponent
    {
        vec3  Position  = { 0.0f, 0.0f, 0.0f };
        float Azimuth   = 0.0f;
        float Elevation = 0.0f;

        vec3  Color     = { 0.0f, 0.0f, 0.0f };
        float Intensity = 0.0f;
        float Cone      = 0.0;

        bool operator==( const sSpotlightComponent &a_Other )
        {
            return ( Position == a_Other.Position ) && ( Azimuth == a_Other.Azimuth ) && ( Elevation == a_Other.Elevation ) &&
                   ( Intensity == a_Other.Intensity ) && ( Color == a_Other.Color ) && ( Cone == a_Other.Cone );
        }
    };

    struct sLightComponent
    {
        Entity Light;

        sLightComponent()                          = default;
        sLightComponent( const sLightComponent & ) = default;
    };

} // namespace SE::Core::EntityComponentSystem::Components
