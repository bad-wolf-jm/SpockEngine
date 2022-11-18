#include "Animation.h"
#include "Conversion.h"
#include "Core/Logging.h"

void to_keyframe_list( aiVectorKey *a_Keys, uint32_t a_Count, std::vector<KeyFrame<math::vec3>> &o_Out )
{
    o_Out.resize( a_Count );
    for( uint32_t l_Idx = 0; l_Idx < a_Count; l_Idx++ )
    {
        o_Out[l_Idx].Tick  = (float)a_Keys[l_Idx].mTime;
        o_Out[l_Idx].Value = to_vec3( a_Keys[l_Idx].mValue );
    }
}

void to_keyframe_list( aiQuatKey *a_Keys, uint32_t a_Count, std::vector<KeyFrame<math::quat>> &o_Out )
{
    o_Out.resize( a_Count );
    for( uint32_t l_Idx = 0; l_Idx < a_Count; l_Idx++ )
    {
        o_Out[l_Idx].Tick  = (float)a_Keys[l_Idx].mTime;
        o_Out[l_Idx].Value = to_quat( a_Keys[l_Idx].mValue );
    }
}

void LoadAnimations( const aiScene *a_SceneData, std::vector<std::shared_ptr<AnimationSequence>> &a_AnimationTracks, std::map<std::string, int32_t> a_Nodes )
{
    SE::Logging::Info( "Loading animations" );
    a_AnimationTracks.resize( a_SceneData->mNumAnimations );

    for( uint32_t l_AnimationIdx = 0; l_AnimationIdx < a_SceneData->mNumAnimations; l_AnimationIdx++ )
    {
        auto l_AnimationChannel                                 = a_SceneData->mAnimations[l_AnimationIdx];
        std::shared_ptr<AnimationSequence> l_AnimationTrackData = std::make_shared<AnimationSequence>();
        l_AnimationTrackData->Name                              = l_AnimationChannel->mName.C_Str();

        if( l_AnimationTrackData->Name.length() == 0 )
            l_AnimationTrackData->Name = fmt::format( "Unnamed animation {}", l_AnimationIdx );

        l_AnimationTrackData->ID             = l_AnimationIdx;
        l_AnimationTrackData->TickCount      = (float)l_AnimationChannel->mDuration;
        l_AnimationTrackData->TicksPerSecond = (float)l_AnimationChannel->mTicksPerSecond;
        l_AnimationTrackData->Duration       = l_AnimationTrackData->TickCount / l_AnimationTrackData->TicksPerSecond;
        SE::Logging::Info( " - {}: duration={} ticks @ {} ticks/s", l_AnimationTrackData->Name, l_AnimationTrackData->TickCount, l_AnimationTrackData->TicksPerSecond );

        for( uint32_t l_NodeAnimationChannelIdx = 0; l_NodeAnimationChannelIdx < l_AnimationChannel->mNumChannels; l_NodeAnimationChannelIdx++ )
        {
            auto l_NodeAnimationChannel = l_AnimationChannel->mChannels[l_NodeAnimationChannelIdx];
            NodeAnimationTrack l_NodeAnimationChannelData;
            l_NodeAnimationChannelData.TargetNodeName = l_NodeAnimationChannel->mNodeName.C_Str();
            l_NodeAnimationChannelData.TargetNodeID   = a_Nodes[l_NodeAnimationChannelData.TargetNodeName];
            SE::Logging::Info( "   * {}", l_NodeAnimationChannelData.TargetNodeName );

            to_keyframe_list( l_NodeAnimationChannel->mPositionKeys, l_NodeAnimationChannel->mNumPositionKeys, l_NodeAnimationChannelData.TranslationKeyFrames );
            to_keyframe_list( l_NodeAnimationChannel->mScalingKeys, l_NodeAnimationChannel->mNumScalingKeys, l_NodeAnimationChannelData.ScalingKeyFrames );
            to_keyframe_list( l_NodeAnimationChannel->mRotationKeys, l_NodeAnimationChannel->mNumRotationKeys, l_NodeAnimationChannelData.RotationKeyFrames );

            l_AnimationTrackData->NodeAnimationTracks.push_back( l_NodeAnimationChannelData );
        }

        a_AnimationTracks[l_AnimationIdx] = l_AnimationTrackData;
    }
}
