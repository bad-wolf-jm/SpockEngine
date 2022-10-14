#pragma once
#include "Core/Memory.h"

#include "Core/GraphicContext//Buffer.h"
#include "Core/GraphicContext//CommandQueue.h"
#include "Core/GraphicContext//DescriptorSet.h"
#include "Core/GraphicContext//Device.h"
#include "Core/GraphicContext//Framebuffer.h"
#include "Core/GraphicContext//GraphicContext.h"
#include "Core/GraphicContext//GraphicsPipeline.h"
#include "Core/GraphicContext//RenderPass.h"
#include "Core/GraphicContext//Texture2D.h"
#include "Core/GraphicContext//TextureCubemap.h"

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
