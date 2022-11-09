#pragma once

#include "Core/Math/Types.h"
#include "Core/Platform/ViewportClient.h"
#include <memory>
#include <vulkan/vulkan.h>

#include "VkContext.h"
#include "VkImage.h"
#include "VkAbstractRenderPass.h"

#include "Core/Memory.h"

namespace LTSE::Graphics::Internal
{
    using namespace LTSE::Core;

    struct sVkLightingRenderPassObject : public sVkAbstractRenderPassObject
    {
        sVkLightingRenderPassObject()                        = default;
        sVkLightingRenderPassObject( sVkLightingRenderPassObject & ) = default;
        sVkLightingRenderPassObject( Ref<VkContext> aContext, std::vector<VkAttachmentDescription> aAttachments,
            std::vector<VkSubpassDescription> aSubpasses, std::vector<VkSubpassDependency> aSubpassDependencies );

        sVkLightingRenderPassObject( Ref<VkContext> aContext, VkFormat aFormat, uint32_t aSampleCount, bool aIsSampled, bool aIsPresented,
            math::vec4 aClearColor );

        ~sVkLightingRenderPassObject() = default;
    };

} // namespace LTSE::Graphics::Internal
