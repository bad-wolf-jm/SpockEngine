#pragma once
#include "Core/Memory.h"

#include "Developer/GraphicContext/Buffer.h"
#include "Developer/GraphicContext/CommandQueue.h"
#include "Developer/GraphicContext/DescriptorSet.h"
#include "Developer/GraphicContext/Device.h"
#include "Developer/GraphicContext/Framebuffer.h"
#include "Developer/GraphicContext/GraphicContext.h"
#include "Developer/GraphicContext/GraphicsPipeline.h"
#include "Developer/GraphicContext/RenderPass.h"
#include "Developer/GraphicContext/Texture2D.h"
#include "Developer/GraphicContext/TextureCubemap.h"

namespace LTSE::Core
{

    using namespace LTSE::Graphics;

    enum IBLCubemapType
    {
        IRRADIANCE     = 0,
        PREFILTEREDENV = 1
    };

    Ref<Texture2D> GenerateBRDFLUT( GraphicContext &a_GraphicContext, int32_t dim = 512 );
    Ref<TextureCubeMap> GenerateIBLCubemap( GraphicContext &a_GraphicContext, Ref<TextureCubeMap> a_Skybox, IBLCubemapType a_Type, int32_t dim = 512 );

} // namespace LTSE::Core
